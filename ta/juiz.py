"""JEV (TypeSafe System One) como juiz de contexto entre candidatos vindos do áudio.

Só o texto do trecho sai da máquina; o áudio fica local. A chave é lida em
tempo de execução de JEV_KEY_FILE e nunca é impressa.
"""
import json
import math
import os
import re
import time
import urllib.error
import urllib.request

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODELO = "jev-latest"
ARQUIVO_CHAVE = ""
POR_REQUISICAO = 12

INSTRUCOES = {
    "pt": (
        "Transcrição LITERAL de fala em português do Brasil. Uma palavra ficou duvidosa no trecho "
        "abaixo, marcada como ⟦?⟧. Escolha qual candidato a pessoa mais provavelmente FALOU. "
        "A fala é espontânea: pode ser informal, agramatical, cortada ou repetida "
        "(\"pra\", \"tá\", \"né\", \"a gente vai\", \"é, é\"). NÃO prefira a forma gramaticalmente "
        "correta só por ser correta; prefira a que cabe no jeito de falar do trecho. "
        "\"(nada)\" significa que não havia palavra ali."
    ),
    "en": (
        "VERBATIM transcription of spontaneous speech. One word in the excerpt below is uncertain, "
        "marked ⟦?⟧. Choose which candidate the speaker most likely SAID. Speech may be informal, "
        "ungrammatical, cut off or repeated. Do NOT prefer the grammatically correct form just "
        "because it is correct. \"(nada)\" means there was no word there."
    ),
    # Conversa que mistura idiomas: o candidato tem que ficar no idioma do trecho.
    "misto": (
        "Transcrição LITERAL de uma conversa que mistura idiomas (numa call, cada lado pode falar o seu). "
        "Uma palavra ficou duvidosa no trecho abaixo, marcada como ⟦?⟧. Escolha qual candidato a pessoa "
        "mais provavelmente FALOU, no idioma original do trecho — não traduza. A fala é espontânea: pode "
        "ser informal, agramatical, cortada ou repetida (\"pra\", \"tá\", \"né\", \"a gente vai\", \"é, é\"). "
        "NÃO prefira a forma gramaticalmente correta só por ser correta. "
        "\"(nada)\" significa que não havia palavra ali."
    ),
}


def carregar_chave():
    caminho = os.environ.get("JEV_KEY_FILE", ARQUIVO_CHAVE)
    try:
        with open(caminho, encoding="utf-8") as f:
            bruto = f.read().strip()
    except OSError:
        return None
    for linha in filter(None, (x.strip() for x in bruto.splitlines())):
        m = re.search(r"(apikey_[A-Za-z0-9_]+)", linha) or re.search(r"(sk-[A-Za-z0-9_\-]{16,})", linha)
        if m:
            return m.group(1)
    linhas = [x for x in bruto.splitlines() if x.strip()]
    return linhas[0].strip() if len(linhas) == 1 else None


def _post(corpo, chave, timeout=25):
    dados = json.dumps(corpo).encode("utf-8")
    for tentativa in range(3):
        req = urllib.request.Request(ENDPOINT, data=dados, method="POST", headers={
            "Authorization": f"Bearer {chave}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 503, 529) and tentativa < 2:
                time.sleep(0.5 * 2 ** tentativa)
                continue
            raise RuntimeError(f"JEV respondeu HTTP {e.code}") from None
        except (urllib.error.URLError, TimeoutError):
            if tentativa < 2:
                continue
            raise RuntimeError("JEV: falha de conexão") from None
        except (ValueError, UnicodeError):
            raise RuntimeError("JEV: resposta inválida") from None
    raise RuntimeError("JEV indisponível")


def _valida(resp, ids):
    try:
        probs = resp["probabilities"]
        if not isinstance(probs, dict):
            return None
        nums = [*probs.values(), resp["confidence"]]
        ok = (resp["choice"] in ids and set(probs) == set(ids)
              and all(isinstance(n, (int, float)) and math.isfinite(n) and 0 <= n <= 1 for n in nums)
              and abs(sum(probs.values()) - 1) < 0.02)
    except (KeyError, TypeError):
        ok = False
    return resp if ok else None


def ranquear(alvos, idioma="pt", progresso=print):
    """Preenche candidato["contexto"] (0–1) e alvo["jev"] para cada alvo com 2+ candidatos."""
    chave = carregar_chave()
    if not chave:
        progresso("  JEV: chave não encontrada (configure JEV_KEY_FILE); pulando juiz")
        return {"usado": False, "motivo": "sem chave"}
    pendentes = [a for a in alvos if len(a.get("candidatos", [])) >= 2]
    instr = INSTRUCOES.get(idioma, INSTRUCOES["en"])
    chamadas, falhas, avaliados = 0, 0, 0
    for i in range(0, len(pendentes), POR_REQUISICAO):
        lote = pendentes[i:i + POR_REQUISICAO]
        perguntas, mapa = {}, {}
        for a in lote:
            ids = {f"c{k + 1}": c for k, c in enumerate(a["candidatos"])}
            mapa[a["id"]] = ids
            trecho = f"…{a['antes']} ⟦?⟧ {a['depois']}…"
            perguntas[a["id"]] = {
                "type": "choice",
                "instructions": f"{instr}\n\nTrecho: {trecho}",
                "criteria": {k: f"\"{c['texto']}\"" for k, c in ids.items()},
            }
        corpo = {"model": MODELO, "state": "Revisão de transcrição de áudio.",
                 "questions": perguntas}
        try:
            resp = _post(corpo, chave)
            chamadas += 1
        except RuntimeError as e:
            falhas += 1
            progresso(f"  JEV: {e}")
            continue
        respostas = resp.get("answers", {}) if isinstance(resp, dict) else {}
        if not isinstance(respostas, dict):
            respostas = {}
        for a in lote:
            ids = mapa[a["id"]]
            r = _valida(respostas.get(a["id"], {}), list(ids))
            if not r:
                continue
            avaliados += 1
            for k, c in ids.items():
                c["contexto"] = round(float(r["probabilities"][k]), 3)
            a["jev"] = {"escolha": ids[r["choice"]]["texto"], "confianca": r["confidence"]}
        progresso(f"  JEV: lote {i // POR_REQUISICAO + 1} ok")
    return {"usado": avaliados > 0, "chamadas": chamadas, "falhas": falhas,
            "alvos": len(pendentes), "avaliados": avaliados}


def veredito(a):
    """Resume as evidências num rótulo curto para a página de revisão.

    Três sinais independentes: votos das re-transcrições (o que o Whisper ouve),
    alinhamento forçado (o quanto o áudio sustenta cada texto) e JEV (contexto).
    """
    cands = a.get("candidatos", [])
    if not cands:
        return ""
    topo = cands[0]
    n = a.get("variantes", 0) or 1
    acust = next((c["texto"] for c in cands if c.get("acustica") == "favorece"), None)
    jev = a.get("jev", {}).get("escolha")
    if a.get("tipo") == "lacuna" and topo["texto"] == "(nada)" and not acust:
        # Sabemos que há voz ali; "nada" só quer dizer que nenhum modelo reconheceu.
        return "voz sem palavra reconhecida: ouvir"
    if acust and jev and acust == jev:
        return f"áudio e contexto concordam: {acust}"
    if acust and jev and acust != jev:
        return "áudio e contexto discordam: ouvir"
    if topo["votos"] / n >= 0.8 and topo["texto"] == a.get("atual"):
        return f"áudio confirma ({topo['votos']}/{n})"
    if acust:
        return f"áudio favorece: {acust}"
    if not jev:
        return "só re-transcrições" if topo["votos"] / n >= 0.7 else "áudio dividido: ouvir"
    if jev == topo["texto"] and topo["votos"] / n >= 0.6:
        return f"re-transcrições e contexto concordam: {jev}"
    return "áudio indeciso: ouvir"
