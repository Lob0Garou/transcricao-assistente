"""Uso: python transcrever.py <audio> [--idioma pt] [--modelo large-v3] [--saida pasta]

Gera em saidas/<nome-do-audio>/:
  revisao.html     página para revisar ouvindo só os pontos suspeitos
  transcricao.txt  texto com tempos, falante, sobreposições e sons
  dados.json       tudo, palavra por palavra, com confiança e motivos
"""
import argparse
import json
import os
import sys
import time

from ta import env

env.preparar()
sys.stdout.reconfigure(encoding="utf-8")

from ta import acustica, analise, candidatos, deepseek, juiz, relatorio, turnos  # noqa: E402
from dataclasses import asdict  # noqa: E402

from ta.asr import Palavra, Transcritor, carregar_modelo  # noqa: E402
from ta.audio import carregar  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--idioma", default="auto",
                    help="auto (detecta por canal, o padrão), pt, en… — forçar o idioma errado faz o "
                         "Whisper TRADUZIR o que foi falado no outro idioma")
    ap.add_argument("--modelo", default="large-v3")
    ap.add_argument("--saida")
    ap.add_argument("--refazer", action="store_true", help="ignora o cache da transcrição")
    ap.add_argument("--sem-candidatos", action="store_true",
                    help="não re-transcreve as dúvidas para gerar candidatos")
    ap.add_argument("--sem-jev", action="store_true", help="não usa o JEV como juiz de contexto")
    ap.add_argument("--sem-deepseek", action="store_true",
                    help="não usa o DeepSeek V4.1 Flash (API do OpenCode) como revisor de texto")
    ap.add_argument("--sem-locutores", action="store_true",
                    help="não separa turnos nem palpite de locutor (uma fala = um bloco)")
    a = ap.parse_args()

    t0 = time.time()
    audio = carregar(a.audio)
    print(f"{os.path.basename(a.audio)}: {audio.duracao:.1f}s, canais {audio.nomes}, "
          f"independentes={audio.independentes} (corr {audio.correlacao:.2f})")

    quadros = {c: acustica.medir(x) for c, x in zip(audio.nomes, audio.canais)}
    nome = os.path.splitext(os.path.basename(a.audio))[0]
    pasta = a.saida or os.path.join(env.RAIZ, "saidas", nome)

    # A transcrição é a parte lenta: fica em cache para refazer só a análise.
    cache = os.path.join(pasta, f"asr_{a.modelo}_{a.idioma}.json")
    cache_idioma = os.path.join(pasta, "idioma.json")
    dispositivo = "cache"
    tr = None
    if os.path.exists(cache) and not a.refazer:
        with open(cache, encoding="utf-8") as f:
            bruto = json.load(f)
        passadas = {c: {k: [Palavra(**p) for p in v] for k, v in d.items()}
                    for c, d in bruto.items()}
        print("transcrição reaproveitada do cache (use --refazer para rodar de novo)")
    else:
        tr = Transcritor(a.modelo, a.idioma)
        dispositivo = tr.dispositivo
        print(f"modelo {a.modelo} em {tr.dispositivo}")
        passadas = {}
        for c, x in zip(audio.nomes, audio.canais):
            q = quadros[c]
            regs = acustica.regioes(q.db > q.limiar)
            passadas[c] = {"inteira": tr.inteira(x, c), "trechos": tr.trechos(x, c, regs)}
            print(f"  canal {c}: {len(passadas[c]['inteira'])} / "
                  f"{len(passadas[c]['trechos'])} palavras")
        os.makedirs(pasta, exist_ok=True)
        with open(cache, "w", encoding="utf-8") as f:
            json.dump({c: {k: [asdict(p) for p in v] for k, v in d.items()}
                       for c, d in passadas.items()}, f, ensure_ascii=False)

    # Idioma por canal: cada lado de uma call pode falar o seu. Fica gravado junto do
    # cache para a próxima execução não precisar transcrever de novo só para saber.
    if tr is not None:
        idiomas = {c: (tr.idioma_do_canal(c) or (a.idioma if a.idioma != "auto" else "en"))
                   for c in audio.nomes}
        with open(cache_idioma, "w", encoding="utf-8") as f:
            json.dump({"pedido": a.idioma, "por_canal": idiomas}, f, ensure_ascii=False, indent=1)
    else:
        try:
            with open(cache_idioma, encoding="utf-8") as f:
                idiomas = json.load(f).get("por_canal") or {}
        except (OSError, ValueError):
            idiomas = {}
        idiomas = {c: idiomas.get(c) or (a.idioma if a.idioma != "auto" else "en") for c in audio.nomes}
    print("idioma por canal: " + ", ".join(f"{c}={v}" for c, v in idiomas.items()))
    distintos = {v for v in idiomas.values()}
    idioma_geral = "misto" if len(distintos) > 1 else (next(iter(distintos)) if distintos else "en")

    r = analise.analisar(audio, quadros, passadas)

    # Turnos de fala e palpite de locutor: quem falou em cada bloco (você confirma na página).
    info_loc = {"usado": False, "motivo": "desligado (--sem-locutores)"}
    if not a.sem_locutores:
        r["blocos"], info_loc = turnos.analisar(r["falas"], quadros)
        info_loc["usado"] = True
        print(f"locutores: {info_loc['com_palpite']} de {info_loc['pedacos']} pedaços com palpite, "
              f"{len(r['blocos'])} blocos")

    info = {"jev": {"usado": False}, "deepseek": {"usado": False}}
    if not a.sem_candidatos:
        alvos, info = resolver_duvidas(a, audio, r, pasta, tr, idiomas, idioma_geral)
        r["alvos"] = alvos
        for o in r.get("outros_erros", []):
            onde = next((f for f in r["falas"] if o.get("trecho") and analise.norm(o["trecho"])
                         in analise.norm(" ".join(p.texto for p in f["palavras"]))), None)
            if onde:
                r["suspeitas"].append({
                    "tipo": "DeepSeek: possível erro de reconhecimento", "prio": 2,
                    "canal": onde["canal"], "ini": onde["ini"], "fim": onde["fim"],
                    "texto": f'"{o["trecho"]}" → "{o.get("sugestao", "")}" ({o.get("motivo", "")})'})
        r["suspeitas"].sort(key=lambda x: (x["prio"], x["ini"]))
    meta = {"audio": a.audio, "modelo": a.modelo, "idioma": a.idioma, "idiomas": idiomas,
            "idioma_geral": idioma_geral,
            "dispositivo": dispositivo, "segundos": time.time() - t0,
            "independentes": audio.independentes, "correlacao": audio.correlacao,
            "locutores": info_loc, **info}
    pagina = relatorio.salvar(audio, r, meta, pasta)
    print(f"{len(r['falas'])} falas, {len(r['sobreposicoes'])} sobreposições, "
          f"{len(r['sons'])} sons, {len(r['suspeitas'])} suspeitas em {meta['segundos']:.0f}s")
    print(pagina)


def resolver_duvidas(a, audio, r, pasta, tr, idiomas, idioma_geral):
    """Candidatos do áudio → sugestões do DeepSeek → pontuação acústica de todos → JEV. Com cache.

    `idiomas` é o mapa {canal: idioma} da detecção: cada candidato é re-transcrito no
    idioma do seu canal. `idioma_geral` é o idioma do texto para DeepSeek e JEV, ou
    "misto" quando os canais falam idiomas diferentes.
    """
    alvos = candidatos.alvos_da_analise(r)
    if not alvos:
        return [], {"usado": False, "motivo": "sem dúvidas"}
    # Nunca reutiliza sugestões externas ao mudar para o modo local.
    cache = os.path.join(pasta, f"cand_{a.modelo}_{a.idioma}"
                        f"_ds{int(not a.sem_deepseek)}_jev{int(not a.sem_jev)}.json")
    salvos = {}
    if os.path.exists(cache) and not a.refazer:
        with open(cache, encoding="utf-8") as f:
            salvos = json.load(f)

    def assinatura(x):
        return f"{x['tipo']}|{x['canal']}|{x['ini']:.2f}|{x['fim']:.2f}"

    novos = []
    for x in alvos:
        antigo = salvos.get(assinatura(x))
        if antigo:
            x.update({k: antigo[k] for k in ("candidatos", "variantes", "jev", "deepseek_ok",
                                             "motivo_deepseek", "acustica_ok") if k in antigo})
        else:
            novos.append(x)

    modelos = {}

    def carregar(nome):
        if nome not in modelos:
            if tr is not None and nome == a.modelo:
                modelos[nome] = tr.m
            else:
                modelos[nome], _ = carregar_modelo(nome)
        return modelos[nome]

    if novos:
        print(f"gerando candidatos para {len(novos)} dúvidas ({len(alvos) - len(novos)} do cache)")
        candidatos.gerar(audio, novos, carregar, idiomas, r["falas"])

    # DeepSeek revisa o texto inteiro; cada sugestão é conferida no áudio antes do JEV.
    info_ds = {"usado": False, "motivo": "desligado (--sem-deepseek)"}
    if not a.sem_deepseek:
        ds = salvos.get("_deepseek") if not novos else None
        if not ds or not ds.get("usado"):
            ds = deepseek.revisar(r, alvos, idioma_geral,
                                  lambda c: relatorio.rotulo_falante(c, audio.independentes))
        if ds.get("usado"):
            pendentes = [x for x in alvos if not x.get("deepseek_ok")]
            candidatos.adicionar_sugestoes(pendentes, ds["duvidas"])
            for x in pendentes:
                x["deepseek_ok"] = True
                x["motivo_deepseek"] = ds["motivos"].get(x["id"], "")
            r["outros_erros"] = ds.get("outros_erros", [])
        info_ds = {k: v for k, v in ds.items() if k not in ("duvidas", "motivos")}
        salvos["_deepseek"] = ds

    # Todo candidato (do áudio, parônimo ou DeepSeek) passa pelo alinhamento forçado.
    candidatos.pontuar_candidatos(audio, alvos, carregar, idiomas, r["falas"])

    info_jev = {"usado": False, "motivo": "desligado (--sem-jev)"}
    if not a.sem_jev:
        sem_juiz = [x for x in alvos if "jev" not in x]
        info_jev = juiz.ranquear(sem_juiz, idioma_geral) if sem_juiz else {"usado": True, "cache": True}

    for x in alvos:
        x["veredito"] = juiz.veredito(x)
    with open(cache, "w", encoding="utf-8") as f:
        dados = {assinatura(x): x for x in alvos}
        if "_deepseek" in salvos:
            dados["_deepseek"] = salvos["_deepseek"]
        json.dump(dados, f, ensure_ascii=False)
    return alvos, {"jev": info_jev, "deepseek": info_ds}


if __name__ == "__main__":
    main()
