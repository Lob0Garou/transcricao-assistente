"""Cruza as duas passadas de ASR com a acústica e produz falas, sons e suspeitas."""
import re
import unicodedata

import numpy as np

from . import acustica

# Palavras e frases que o Whisper inventa em silêncio, cliques, tosse e finais de áudio.
ALUC_PALAVRAS = {
    "obrigado", "obrigada", "tchau", "valeu", "legendas", "amara", "amaraorg",
    "inscrevase", "assistir", "thank", "thanks", "bye", "watching", "subscribe",
}
ALUC_FRASES = {
    "obrigado", "obrigada", "obrigadoporassistir", "tchau", "tchautchau", "valeu",
    "legendaspelacomunidadeamaraorg", "inscrevasenocanal", "thankyou", "thanks",
    "thanksforwatching", "thankyouverymuch", "bye", "byebye",
}
# Backchannels curtos ("uhum", "mm-hmm") ficam de fora de propósito: costumam ter
# confiança baixa e mesmo assim são fala real, e importam para sobreposição.

BAIXA_CONF = 0.5


def norm(t):
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", t)


def _perto(p, o, folga=0.25):
    return min(p.fim, o.fim) - max(p.ini, o.ini) > -folga


def _comparar(p, outras):
    """'ok' = a outra passada ouviu a mesma palavra; 'diverge' = ouviu outra coisa
    no mesmo lugar; 'ausente' = a outra passada não ouviu nada ali."""
    vizinhas = [o for o in outras if _perto(p, o)]
    if not vizinhas:
        return "ausente"
    n = norm(p.texto)
    return "ok" if any(norm(o.texto) == n for o in vizinhas) else "diverge"


def _voz(q, ini, fim):
    v = q.vozeado[acustica.fatia(q, ini, fim)]
    return float(v.mean()) if len(v) else 0.0


def montar_falas(palavras, gap=0.8):
    falas = []
    for p in sorted(palavras, key=lambda p: p.ini):
        if falas and falas[-1]["canal"] == p.canal and p.ini - falas[-1]["fim"] < gap:
            falas[-1]["palavras"].append(p)
            falas[-1]["fim"] = max(falas[-1]["fim"], p.fim)
        else:
            falas.append({"canal": p.canal, "ini": p.ini, "fim": p.fim, "palavras": [p]})
    return falas


def _texto(f):
    return " ".join(p.texto for p in f["palavras"])


def _fala(canal, palavras):
    return {"canal": canal, "ini": palavras[0].ini, "fim": max(p.fim for p in palavras),
            "palavras": palavras}


def _tirar_alucinacoes(palavras):
    """Tira "Thank you"/"Obrigado" inventados de dentro de falas reais."""
    limpas, removidas, i = [], [], 0
    while i < len(palavras):
        p = palavras[i]
        if norm(p.texto) in ALUC_PALAVRAS and p.prob < 0.3:
            removidas.append(p)
            prox = palavras[i + 1] if i + 1 < len(palavras) else None
            if prox and norm(p.texto + prox.texto) in ALUC_FRASES:
                removidas.append(prox)
                i += 1
        else:
            limpas.append(p)
        i += 1
    return limpas, removidas


def _agrupar_cliques(sons, gap=1.0):
    """Junta estalos vizinhos num único evento de digitação/cliques."""
    out = []
    for s in sorted(sons, key=lambda s: s["ini"]):
        clique = s["rotulo"] in ("estalo/clique", "cliques/digitação")
        ant = out[-1] if out else None
        if clique and ant and ant.get("_cliques") and s["ini"] - ant["fim"] < gap:
            ant["fim"] = s["fim"]
            ant["_cliques"] += 1
            ant["rotulo"] = f"cliques/digitação ({ant['_cliques']} eventos)"
            continue
        if clique:
            s = {**s, "_cliques": 1}
        out.append(s)
    for s in out:
        s.pop("_cliques", None)
    return out


def _provavel_alucinacao(f, q):
    ps = f["palavras"]
    pmax = max(p.prob for p in ps)
    frase = norm(_texto(f))
    if frase in ALUC_FRASES and min(p.prob for p in ps) < 0.6:
        return True
    if len(ps) <= 3 and pmax < 0.3:  # nada ali tem cara de palavra
        return True
    voz = _voz(q, f["ini"] - 0.1, f["fim"] + 0.1)
    if voz < 0.05:
        return True
    # Trecho curto, pouco vozeado e inseguro: tosse, clique ou respiração lidos como palavra.
    media = sum(p.prob for p in ps) / len(ps)
    if f["fim"] - f["ini"] < 1.0 and voz < 0.3 and media < 0.5:
        return True
    return False


def analisar(audio, quadros, passadas):
    """quadros: {canal: Quadros}; passadas: {canal: {"inteira": [...], "trechos": [...]}}"""
    falas_todas, suspeitas, sons = [], [], []

    for canal in audio.nomes:
        q = quadros[canal]
        base, outra = passadas[canal]["trechos"], passadas[canal]["inteira"]
        for p in base:
            p.flags = []
            if p.prob < BAIXA_CONF:
                p.flags.append(f"confiança {p.prob:.0%}")
            cmp = _comparar(p, outra)
            if cmp == "diverge":
                p.flags.append("passadas divergem")
            if norm(p.texto) in ALUC_PALAVRAS and p.prob < 0.6:
                p.flags.append("palavra típica de alucinação")

        falas = montar_falas(base)
        validas, descartadas = [], []
        for f in falas:
            limpas, removidas = _tirar_alucinacoes(f["palavras"])
            if removidas:
                descartadas.append(_fala(canal, removidas))
            if not limpas:
                continue
            f = _fala(canal, limpas)
            if _provavel_alucinacao(f, q):
                descartadas.append(f)
                continue
            validas.append(f)
            marcadas = [p for p in f["palavras"] if p.flags]
            if marcadas:
                pior = min(p.prob for p in marcadas)
                suspeitas.append({
                    "tipo": f"{len(marcadas)} palavra(s) para conferir", "prio": 3 if pior >= 0.2 else 2,
                    "canal": canal, "ini": min(p.ini for p in marcadas),
                    "fim": max(p.fim for p in marcadas),
                    "texto": " · ".join(f"{p.texto} ({'; '.join(p.flags)})" for p in marcadas)})

        # O que só a passada inteira ouviu (o recorte em trechos pode ter engolido).
        for o in outra:
            if o.prob >= 0.5 and _comparar(o, base) == "ausente":
                suspeitas.append({"tipo": "só a passada inteira ouviu", "prio": 2,
                                  "canal": canal, "ini": o.ini, "fim": o.fim, "texto": o.texto})

        falas_todas += validas

        # Sons: partes com energia que nenhuma palavra válida cobre.
        f0c = acustica.f0_tipico(q)
        coberto = np.zeros(len(q.db), bool)
        for f in validas:
            for p in f["palavras"]:
                coberto[acustica.fatia(q, p.ini - 0.08, p.fim + 0.08)] = True
        bordas = [(f["ini"], f["fim"]) for f in validas]
        novos = []
        for a, b in acustica.regioes((q.db > q.limiar) & ~coberto, junta=0.25, minimo=0.05):
            d = acustica.descrever(q, a, b)
            inventado = [f for f in descartadas if f["ini"] < b and f["fim"] > a]
            colado = any(c - 0.35 < b and a < e + 0.35 for c, e in bordas)
            if inventado and d["voz"] >= 0.5 and d["dur_ativa"] >= 0.15:
                # O ASR inventou texto em cima de voz de verdade: o problema é a palavra.
                rotulo, prio = "fala não reconhecida: ouvir e transcrever", 1
            elif colado and d["voz"] >= 0.5 and d["dur_ativa"] >= 0.2:
                # Voz encostada numa fala, sem palavra: o ASR engoliu um pedaço
                # (foi assim que o "Cool" de "Cool. Sounds good." sumiu nos testes).
                rotulo, prio = "voz sem palavra junto de fala (palavra engolida?)", 2
            else:
                rotulo = acustica.classificar_som(d, b - a, f0c)
                if not rotulo:
                    continue
                prio = 2 if rotulo.startswith(("vocalização", "riso", "tosse")) else 4
            texto = (f"ASR inventou: {' / '.join(_texto(f) for f in inventado)}"
                     if inventado else "")
            novos.append({"canal": canal, "ini": a, "fim": b, "rotulo": rotulo,
                          "prio": prio, "texto": texto, **d})

        for s in _agrupar_cliques(novos):
            sons.append(s)
            suspeitas.append({"tipo": f"som sem palavra: {s['rotulo']}", "prio": s["prio"],
                              "canal": canal, "ini": s["ini"], "fim": s["fim"],
                              "texto": s["texto"]})

    falas_todas.sort(key=lambda f: f["ini"])

    sobreposicoes = []
    if audio.independentes and len(audio.nomes) == 2:
        e, d = (quadros[c] for c in audio.nomes)
        n = min(len(e.db), len(d.db))
        # Exige voz nos dois lados: estalo de um lado com fala do outro não conta.
        for a, b in acustica.regioes(e.vozeado[:n] & d.vozeado[:n], junta=1.0, minimo=0.12):
            quem = [f for f in falas_todas if f["ini"] < b + 0.2 and f["fim"] > a - 0.2]
            sobreposicoes.append({"ini": a, "fim": b,
                                  "textos": {f["canal"]: _texto(f) for f in quem}})
            suspeitas.append({"tipo": "sobreposição", "prio": 2, "canal": "E+D",
                              "ini": a, "fim": b,
                              "texto": " × ".join(f"{f['canal']}: {_texto(f)[:60]}" for f in quem)})

    suspeitas.sort(key=lambda s: (s["prio"], s["ini"]))
    return {"falas": falas_todas, "sons": sons, "sobreposicoes": sobreposicoes,
            "suspeitas": suspeitas}
