"""Testa a divisão em turnos e o palpite de locutor, sem chamar nenhuma API.

Caso: maven-audio de 120 s (inglês), em que as DUAS pessoas estão no mesmo canal
— o caso que a ferramenta não resolvia. As falas do Whisper misturam as duas
vozes ("What do you feed it in? So basically there's...").

Como não há transcrição humana de referência, o teste NÃO afirma quem disse o
quê. Ele verifica o que dá para provar:

  1. a fala que mistura as duas pessoas é cortada em mais de um pedaço;
  2. os dois grupos formados têm tom claramente diferente (>= 40% de diferença
     entre as medianas) — se não tiver, o palpite seria ruído;
  3. o corte por TOM concorda com um corte feito por outra medida, o ESPECTRO
     (formantes/bandas de frequência), calculado aqui de forma independente.
     Concordância alta significa que os grupos são vozes diferentes, e não um
     artefato do estimador de tom (dobra de oitava, por exemplo).

O listão de blocos no fim serve para você conferir quem falou.

Uso: python -s testes/teste_turnos.py
"""
import json
import os
import re
import sys
import unicodedata

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from ta import env  # noqa: E402

env.preparar()
import numpy as np  # noqa: E402
sys.stdout.reconfigure(encoding="utf-8")

from ta import acustica, analise, turnos  # noqa: E402
from ta.asr import Palavra  # noqa: E402
from ta.audio import SR, carregar  # noqa: E402

AUDIO = os.environ.get("TA_AUDIO_TURNOS", "")
CACHE = os.environ.get("TA_CACHE_TURNOS", "")
if not os.path.isfile(AUDIO) or not os.path.isfile(CACHE):
    sys.exit("Configure TA_AUDIO_TURNOS e TA_CACHE_TURNOS com os arquivos de avaliação externos.")
MISTA = "what do you feed it in"   # frase que começa no meio de uma fala com duas vozes
DIFERENCA_MIN = 0.40               # diferença mínima entre as medianas de tom dos grupos
CONCORDANCIA_MIN = 0.85            # concordância mínima com o corte por espectro

falhas = []


def norm(t):
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]+", " ", t.replace("'", "").replace("’", "")).strip()


def bandas(x, ini, fim, n=24):
    """Espectro médio em bandas log — medida independente do tom (formantes).

    É o segundo juiz: se o corte por tom e o corte por espectro concordam, os
    grupos são vozes de verdade.
    """
    seg = x[int(ini * SR):int(fim * SR)]
    if len(seg) < 2048:
        return None
    F = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    fr = np.fft.rfftfreq(len(seg), 1 / SR)
    bordas = np.logspace(np.log10(80), np.log10(min(7000, SR / 2 - 200)), n + 1)
    v = np.array([np.log(F[(fr >= a) & (fr < b)].mean() + 1e-12) if ((fr >= a) & (fr < b)).any()
                  else 0.0 for a, b in zip(bordas[:-1], bordas[1:])])
    return (v - v.mean()) / (v.std() + 1e-9)


def concordancia(a, b):
    """Fração de pares de blocos que os dois cortes colocam do mesmo lado."""
    iguais = total = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            total += 1
            iguais += (a[i] == a[j]) == (b[i] == b[j])
    return iguais / total if total else 0.0


print(f"áudio: {os.path.basename(AUDIO)}")
audio = carregar(AUDIO)
quadros = {c: acustica.medir(x) for c, x in zip(audio.nomes, audio.canais)}
with open(CACHE, encoding="utf-8") as f:
    passadas = {c: {k: [Palavra(**p) for p in v] for k, v in d.items()}
                for c, d in json.load(f).items()}
r = analise.analisar(audio, quadros, passadas)
print(f"{audio.duracao:.0f}s, canais {audio.nomes}, {len(r['falas'])} falas do Whisper")

pedacos = turnos.medir(turnos.dividir(r["falas"]), quadros)
blocos, info = turnos.analisar(r["falas"], quadros)
print(f"{info['pedacos']} pedaços, {info['com_palpite']} com palpite, "
      f"{len(blocos)} blocos depois de agrupar\n")

# Corte independente: espectro dos mesmos blocos, agrupado pelo mesmo k-means
# (a mesma máquina de agrupar, medida diferente — é a medida que está em teste).
usados = [b for b in blocos if b["locutor"]]
X, com_espectro = [], []
for b in usados:
    v = bandas(dict(zip(audio.nomes, audio.canais))[b["canal"]], b["ini"], b["fim"])
    if v is not None:
        X.append(v)
        com_espectro.append(b)
grupos_espectro = (turnos._kmeans(np.array(X), np.ones(len(X))) if len(X) >= 4 else None)

print("blocos (o tom é o palpite; o grupo do espectro é a conferência independente):")
for b in blocos:
    quem = f"Locutor {b['locutor']}" if b["locutor"] else "sem palpite"
    conf = f" ({b['confianca']:.2f})" if b["locutor"] else ""
    tom = acustica.descrever(quadros[b["canal"]], b["ini"], b["fim"])["f0"]
    esp = ""
    if grupos_espectro is not None and b in com_espectro:
        esp = f" espectro=g{grupos_espectro[com_espectro.index(b)]}"
    print(f"  [{b['ini']:6.1f}s] {quem}{conf} f0={tom:3.0f}Hz{esp} · {b['texto'][:60]}")

print("\nverificações:")
mista = [f for f in r["falas"] if norm(MISTA) in norm(" ".join(p.texto for p in f["palavras"]))]
if not mista:
    falhas.append("fala mista não encontrada")
    print("  ERRO corte: fala mista não encontrada")
else:
    f = mista[0]
    dentro = [p for p in pedacos if p["ini"] >= f["ini"] - 0.01 and p["fim"] <= f["fim"] + 0.01]
    ok = len(dentro) >= 2
    if not ok:
        falhas.append(f"a fala de {f['ini']:.1f}s não foi cortada")
    print(f"  {'OK  ' if ok else 'ERRO'} corte: a fala de {f['ini']:.1f}s "
          f"({f['fim'] - f['ini']:.1f}s) virou {len(dentro)} pedaços")

medianas = {}
for loc in (1, 2):
    f0s = [acustica.descrever(quadros[b["canal"]], b["ini"], b["fim"])["f0"]
           for b in usados if b["locutor"] == loc]
    f0s = [x for x in f0s if x > 0]
    medianas[loc] = float(np.median(f0s)) if f0s else 0.0
    print(f"  tom do locutor {loc}: {len(f0s)} blocos, mediana {medianas[loc]:.0f} Hz")
grave, agudo = sorted(medianas.values())
dif = (agudo / grave - 1) if grave else 0.0
ok = grave > 0 and dif >= DIFERENCA_MIN
if not ok:
    falhas.append(f"tom dos grupos parecido demais ({dif:.0%})")
print(f"  {'OK  ' if ok else 'ERRO'} separação de tom: {dif:.0%} de diferença "
      f"({grave:.0f} Hz contra {agudo:.0f} Hz)")

if grupos_espectro is None:
    falhas.append("espectro: poucos blocos para conferir")
    print("  ERRO espectro: poucos blocos para conferir")
else:
    a = [b["locutor"] for b in com_espectro]
    c = list(grupos_espectro)
    conc = concordancia(a, c)
    ok = conc >= CONCORDANCIA_MIN
    if not ok:
        falhas.append(f"corte por tom e por espectro discordam (concordância {conc:.0%})")
    print(f"  {'OK  ' if ok else 'ERRO'} conferência independente: o corte por tom concorda "
          f"{conc:.0%} com o corte por espectro ({len(com_espectro)} blocos)")

print("\nRESULTADO:", "PASSOU" if not falhas else "FALHOU")
for f in falhas:
    print("  -", f)
sys.exit(1 if falhas else 0)
