"""Pontuação acústica de candidatos por alinhamento forçado (teacher forcing).

Para cada candidato, monta a frase do trecho com o candidato no lugar da dúvida
e pede ao Whisper a probabilidade de cada token dessa frase dado o áudio.
Diferente de "dar uma dica" no prompt (que o modelo simplesmente repete), aqui
o modelo não gera nada: só diz o quanto o áudio sustenta cada texto.
"""
import math

import numpy as np

from .audio import SR

MARGEM = 2.5


def _limpo(t):
    return t.strip().strip(".,!?;:…\"'").strip()


def _trecho(alvo, falas, duracao):
    """Palavras inteiras do mesmo canal em volta da dúvida, e o recorte que as contém."""
    ws = [p for f in falas if f["canal"] == alvo["canal"] for p in f["palavras"]
          if p.ini >= alvo["ini"] - MARGEM and p.fim <= alvo["fim"] + MARGEM]
    ws.sort(key=lambda p: p.ini)
    ini = min([alvo["ini"]] + [p.ini for p in ws]) - 0.15
    fim = max([alvo["fim"]] + [p.fim for p in ws]) + 0.15
    return ws, max(0.0, ini), min(duracao, fim)


def _frase(alvo, ws, texto):
    partes = []
    inserido = False
    for p in ws:
        mesmo = alvo["tipo"] == "palavra" and abs(p.ini - alvo["ini"]) < 0.01 and abs(p.fim - alvo["fim"]) < 0.01
        if alvo["tipo"] == "lacuna" and not inserido and p.ini >= alvo["fim"] - 0.05:
            partes.append(texto)
            inserido = True
        partes.append(texto if mesmo else _limpo(p.texto))
    if alvo["tipo"] == "lacuna" and not inserido:
        partes.append(texto)
    return " ".join(t for t in partes if t and t != "(nada)")


def _logp(modelo, tok, enc, n, frase):
    ids = tok.encode(" " + frase) if frase else []
    if not ids:
        return 0.0
    res = modelo.model.align(enc, tok.sot_sequence, [ids], n)[0]
    return float(sum(math.log(max(p, 1e-9)) for p in res.text_token_probs))


# Peso do desconto do "gosto" do decodificador. Com 1,0 nomes raros ganham bônus
# ("Aziz" passou "as is"); sem desconto, "Thank you" ganha de fala real. 0,75 foi o
# meio-termo nos casos de teste (testes/teste_pontuacao.py). Validação pequena: n=4.
LAMBDA = 0.75
# Diferença mínima para afirmar algo. Abaixo disso o áudio não distingue
# (homófonos como "Aziz" × "as is" ficam ~3 pontos um do outro).
MARGEM_DECISAO = 3.5
MARGEM_CONTRA = 8.0


def pontuar(modelo, idioma, sinal, alvo, falas, duracao, textos):
    """Devolve {texto: evidência acústica} para cada candidato.

    Evidência = log P(frase | áudio) − λ·log P(frase | silêncio). O segundo termo é
    o "gosto" do decodificador por aquela frase: é por ele que o Whisper inventa
    "Thank you" em silêncio. Descontado, sobra o quanto o ÁUDIO sustenta o texto.
    """
    from faster_whisper.audio import pad_or_trim
    from faster_whisper.tokenizer import Tokenizer

    ws, a0, b0 = _trecho(alvo, falas, duracao)
    y = np.ascontiguousarray(sinal[int(a0 * SR):int(b0 * SR)])
    feats = modelo.feature_extractor(y)
    n = min(feats.shape[-1], 3000)
    enc = modelo.encode(pad_or_trim(feats))
    enc_vazio = modelo.encode(pad_or_trim(modelo.feature_extractor(np.zeros_like(y))))
    tok = Tokenizer(modelo.hf_tokenizer, True, task="transcribe", language=idioma)
    out = {}
    for t in textos:
        frase = _frase(alvo, ws, t)
        out[t] = _logp(modelo, tok, enc, n, frase) - LAMBDA * _logp(modelo, tok, enc_vazio, n, frase)
    return out


def classificar(evid, tipo):
    """{texto: "favorece" | "contra" | "indeciso"}. Conservador de propósito.

    Em lacuna, "(nada)" fica fora da disputa: já se sabe que há voz ali.
    """
    disputa = {t: v for t, v in evid.items() if not (tipo == "lacuna" and t == "(nada)")}
    if not disputa:
        return {}
    ordem = sorted(disputa, key=disputa.get, reverse=True)
    melhor = disputa[ordem[0]]
    segundo = disputa[ordem[1]] if len(ordem) > 1 else melhor - 99
    rot = {}
    for t, v in disputa.items():
        if t == ordem[0] and melhor - segundo >= MARGEM_DECISAO:
            rot[t] = "favorece"
        elif melhor - v >= MARGEM_CONTRA:
            rot[t] = "contra"
        else:
            rot[t] = "indeciso"
    return rot


def preferencias(logps):
    """Converte log-probs em preferências que somam 1 (softmax). Só para diagnóstico:
    exagera diferenças pequenas, por isso a página usa classificar()."""
    if not logps:
        return {}
    mx = max(logps.values())
    ex = {t: math.exp(v - mx) for t, v in logps.items()}
    s = sum(ex.values())
    return {t: v / s for t, v in ex.items()}
