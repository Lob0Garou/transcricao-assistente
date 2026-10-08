"""Candidatos para palavras duvidosas, tirados do ÁUDIO.

Para cada dúvida, re-transcreve só o trecho em volta (±2 s) com várias
variações de decodificação e conta quantas ouviram cada forma. O juiz de
contexto (JEV) só pode escolher entre esses candidatos: nunca inventa palavra.
"""
from difflib import SequenceMatcher

import numpy as np

from .analise import norm
from .asr import PROMPT
from .audio import SR

MARGEM = 2.0

# Pares que soam parecido em PT-BR e que o contexto ajuda a separar.
# Entram com 0 votos de áudio, só quando um dos lados foi ouvido.
PARONIMOS = [
    {"mas", "mais"}, {"a", "há", "à"}, {"pra", "para"}, {"tá", "está"},
    {"tô", "estou"}, {"cê", "você"}, {"por que", "porque", "porquê"},
    {"com certeza", "concerteza"}, {"mal", "mau"}, {"sessão", "seção", "cessão"},
    {"onde", "aonde"}, {"senão", "se não"}, {"afim", "a fim"}, {"eu", "e"},
    {"né", "não é"}, {"tem", "têm"}, {"vem", "vêm"},
]


def _lang(idioma, canal):
    """Idioma de um canal: aceita um idioma só ou o mapa {canal: idioma} da detecção."""
    if isinstance(idioma, dict):
        return idioma.get(canal) or "en"
    return idioma or "en"


def _variantes(idioma, contexto):
    base = PROMPT.get(idioma)
    return [
        ("principal", "large-v3", dict(beam_size=5, initial_prompt=base)),
        ("sem prompt", "large-v3", dict(beam_size=5, initial_prompt=None)),
        ("com contexto", "large-v3", dict(beam_size=5, initial_prompt=(contexto or base))),
        ("amostra 1", "large-v3", dict(beam_size=1, best_of=1, temperature=0.5, initial_prompt=base)),
        ("amostra 2", "large-v3", dict(beam_size=1, best_of=1, temperature=0.8, initial_prompt=base)),
        ("turbo", "large-v3-turbo", dict(beam_size=5, initial_prompt=base)),
    ]


def limpar(t):
    return t.strip().strip(".,!?;:…\"'").strip()


def alvos_da_analise(r):
    """Palavras marcadas e lacunas (voz sem palavra) viram alvos de candidatos."""
    alvos = []
    for fi, f in enumerate(r["falas"]):
        ws = f["palavras"]
        for wi, p in enumerate(ws):
            if not getattr(p, "flags", None):
                continue
            alvos.append({
                "tipo": "palavra", "canal": f["canal"], "ini": p.ini, "fim": p.fim,
                "atual": limpar(p.texto), "fala": fi, "pos": wi,
                "antes": " ".join(w.texto for w in ws[max(0, wi - 12):wi]),
                "depois": " ".join(w.texto for w in ws[wi + 1:wi + 13]),
            })
    for s in r["sons"]:
        if not s["rotulo"].startswith(("voz sem palavra", "fala não reconhecida")):
            continue
        perto = [f for f in r["falas"] if f["canal"] == s["canal"]
                 and f["ini"] - 0.6 < s["fim"] and s["ini"] < f["fim"] + 0.6]
        antes = depois = ""
        if perto:
            f = perto[0]
            antes = " ".join(w.texto for w in f["palavras"] if w.fim <= s["ini"] + 0.05)[-120:]
            depois = " ".join(w.texto for w in f["palavras"] if w.ini >= s["fim"] - 0.05)[:120]
        alvos.append({"tipo": "lacuna", "canal": s["canal"], "ini": s["ini"], "fim": s["fim"],
                      "atual": "", "antes": antes, "depois": depois})
    alvos.sort(key=lambda a: (a["canal"], a["ini"]))
    for i, a in enumerate(alvos):
        a["id"] = f"a{i + 1}"
    return alvos


def _grupos(alvos):
    """Junta alvos próximos do mesmo canal para transcrever o trecho uma vez só."""
    grupos = []
    for a in alvos:
        g = grupos[-1] if grupos else None
        if g and g["canal"] == a["canal"] and a["ini"] - g["fim"] < 3.0:
            g["alvos"].append(a)
            g["fim"] = max(g["fim"], a["fim"])
        else:
            grupos.append({"canal": a["canal"], "ini": a["ini"], "fim": a["fim"], "alvos": [a]})
    return grupos


def _alinhar(base, var):
    """Para cada palavra da base, a fatia [j1, j2) da variante alinhada a ela.

    Alinha por texto, não por tempo: cada re-transcrição desloca os timestamps,
    e recortar por tempo pegava palavras vizinhas ("No, I think" no lugar de "No").
    """
    sm = SequenceMatcher(None, [norm(t) for t, _, _ in base], [norm(t) for t, _, _ in var],
                         autojunk=False)
    mapa = [None] * len(base)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "insert":
            continue
        if op == "equal" or (i2 - i1) == (j2 - j1):
            for k in range(i2 - i1):
                mapa[i1 + k] = (j1 + k, j1 + k + 1)
        elif op == "delete":
            for k in range(i1, i2):
                mapa[k] = (j1, j1)
        else:  # replace com tamanhos diferentes: reparte pelo tempo
            for k in range(i1, i2):
                c = (base[k][1] + base[k][2]) / 2
                js = [j for j in range(j1, j2) if abs((var[j][1] + var[j][2]) / 2 - c) < 0.6]
                mapa[k] = (min(js), max(js) + 1) if js else (j1, j1)
    return mapa


def _candidato(a, base, var, mapa):
    if a["tipo"] == "palavra":
        k = next((i for i, (t, s, e) in enumerate(base)
                  if abs(s - a["ini"]) < 0.01 and abs(e - a["fim"]) < 0.01), None)
        if k is None or mapa[k] is None:
            return None
        j1, j2 = mapa[k]
        return limpar(" ".join(t for t, _, _ in var[j1:j2]))
    # Lacuna: o que a variante pôs entre a palavra anterior e a seguinte da base.
    esq = max((i for i, (t, s, e) in enumerate(base) if e <= a["ini"] + 0.05), default=None)
    dir_ = min((i for i, (t, s, e) in enumerate(base) if s >= a["fim"] - 0.05), default=None)
    j1 = mapa[esq][1] if esq is not None and mapa[esq] else None
    j2 = mapa[dir_][0] if dir_ is not None and mapa[dir_] else None
    if j1 is None:
        j1 = next((j for j, (t, s, e) in enumerate(var) if e > a["ini"] - 0.3), len(var))
    if j2 is None:
        j2 = next((j for j, (t, s, e) in enumerate(var) if s > a["fim"] + 0.3), len(var))
    return limpar(" ".join(t for t, _, _ in var[j1:max(j1, j2)]))


def gerar(audio, alvos, carregar_modelo, idioma, falas, progresso=print):
    """carregar_modelo(nome) -> WhisperModel (com cache do lado de quem chama)."""
    sinais = dict(zip(audio.nomes, audio.canais))
    grupos = _grupos(alvos)
    for a in alvos:
        a["votos"] = {}
        a["variantes"] = 0
    for gi, g in enumerate(grupos):
        x = sinais[g["canal"]]
        a0 = max(0.0, g["ini"] - MARGEM)
        b0 = min(audio.duracao, g["fim"] + MARGEM)
        y = np.ascontiguousarray(x[int(a0 * SR):int(b0 * SR)])
        base = [(p.texto, p.ini, p.fim) for f in falas if f["canal"] == g["canal"]
                for p in f["palavras"] if a0 <= (p.ini + p.fim) / 2 <= b0]
        contexto = " ".join(a["antes"] for a in g["alvos"][:1])[-200:]
        lang = _lang(idioma, g["canal"])
        for nome, modelo, kw in _variantes(lang, contexto):
            m = carregar_modelo(modelo)
            segs, _ = m.transcribe(y, language=lang, word_timestamps=True, vad_filter=False,
                                   condition_on_previous_text=False, **kw)
            var = [(w.word.strip(), w.start + a0, w.end + a0) for s in segs for w in s.words
                   if w.word.strip()]
            mapa = _alinhar(base, var)
            for a in g["alvos"]:
                texto = _candidato(a, base, var, mapa)
                if texto is None:
                    continue
                chave = norm(texto)
                if chave not in a["votos"]:
                    a["votos"][chave] = {"texto": texto or "(nada)", "votos": 0, "fontes": []}
                a["votos"][chave]["votos"] += 1
                a["votos"][chave]["fontes"].append(nome)
                a["variantes"] += 1
        progresso(f"  candidatos: grupo {gi + 1}/{len(grupos)}")

    for a in alvos:
        cands = a.pop("votos")
        if a["atual"] and norm(a["atual"]) not in cands:
            cands[norm(a["atual"])] = {"texto": a["atual"], "votos": 0, "fontes": ["rascunho"]}
        for grupo in PARONIMOS:
            ouvidos = {c["texto"].lower() for c in cands.values()}
            if ouvidos & grupo:
                for p in grupo:
                    if norm(p) not in cands:
                        cands[norm(p)] = {"texto": p, "votos": 0, "fontes": ["parônimo"]}
        a["candidatos"] = sorted(cands.values(), key=lambda c: -c["votos"])
    return alvos


def _chave(texto):
    return "" if texto == "(nada)" else norm(texto)


def adicionar_sugestoes(alvos, sugestoes):
    """Põe as alternativas do revisor de texto (DeepSeek) entre os candidatos.

    Elas entram com 0 votos de áudio; quem diz se o áudio sustenta cada uma é a
    pontuação acústica (pontuar_candidatos), não o próprio revisor.
    """
    for a in alvos:
        alts = sugestoes.get(a["id"], [])
        if not alts:
            continue
        cands = {_chave(c["texto"]): c for c in a.get("candidatos", [])}
        mudou = False
        for alt in alts:
            alt = limpar(alt) or "(nada)"
            c = cands.get(_chave(alt))
            if c is None:
                c = cands[_chave(alt)] = {"texto": alt, "votos": 0, "fontes": []}
                mudou = True
            if "DeepSeek" not in c["fontes"]:
                c["fontes"].append("DeepSeek")
        a["candidatos"] = sorted(cands.values(), key=lambda c: -c["votos"])
        if mudou:
            # Lista nova: a acústica e o juiz precisam ver de novo.
            a.pop("acustica_ok", None)
            a.pop("jev", None)
    return alvos


def pontuar_candidatos(audio, alvos, carregar_modelo, idioma, falas, progresso=print):
    """Alinhamento forçado de cada candidato no áudio (ver ta/pontuacao.py).

    Marca cada candidato como "favorece", "contra" ou "indeciso". Conservador: só
    afirma algo quando a diferença é grande; homófonos ficam "indeciso".
    """
    from . import pontuacao

    sinais = dict(zip(audio.nomes, audio.canais))
    pendentes = [a for a in alvos if not a.get("acustica_ok") and len(a.get("candidatos", [])) >= 2]
    if not pendentes:
        return alvos
    m = carregar_modelo("large-v3")
    for a in pendentes:
        textos = [c["texto"] for c in a["candidatos"]]
        ev = pontuacao.pontuar(m, _lang(idioma, a["canal"]), sinais[a["canal"]], a, falas,
                               audio.duracao, textos)
        rot = pontuacao.classificar(ev, a["tipo"])
        for c in a["candidatos"]:
            c["evidencia"] = round(ev[c["texto"]], 2)
            c["acustica"] = rot.get(c["texto"], "fora da disputa")
        a["acustica_ok"] = True
    progresso(f"  pontuação acústica (alinhamento forçado): {len(pendentes)} dúvidas")
    return alvos
