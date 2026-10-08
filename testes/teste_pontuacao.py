"""Testa a pontuação acústica por alinhamento forçado, sem API.

Casos com resposta clara (análise manual + sentido da frase):
  curto 8,95 s  lacuna "Cool" antes de "Sounds good"
  longo 110 s   "as is" (o rascunho tem "Aziz")
  longo 191,6 s "calendar view" (o rascunho tem "for")
  longo 17,9 s  "at the latest if we want" (o rascunho tem "that")
O teste exige: "Cool" marcado como "favorece" e NENHUMA forma errada marcada como
"favorece" nos outros casos (ficar "indeciso" é aceitável: são quase homófonos).

Uso: python -s testes/teste_pontuacao.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ta import env  # noqa: E402

env.preparar()
sys.stdout.reconfigure(encoding="utf-8")

from faster_whisper import WhisperModel  # noqa: E402

from ta import acustica, analise, candidatos, pontuacao  # noqa: E402
from ta.asr import Palavra  # noqa: E402
from ta.audio import carregar  # noqa: E402

UNIDADES = {
    "curto": os.environ.get("TA_AUDIO_CURTO", ""),
    "longo": os.environ.get("TA_AUDIO_LONGO", ""),
}
if not all(os.path.isfile(p) for p in UNIDADES.values()):
    sys.exit("Configure TA_AUDIO_CURTO e TA_AUDIO_LONGO com os arquivos de avaliação externos.")
CASOS = [
    ("curto", 8.95, "Cool", ["Cool", "Okay", "Banana", "Thank you", "(nada)"]),
    ("longo", 109.9, "as is", ["as is", "Aziz", "as easy", "(nada)"]),
    ("longo", 191.6, "view", ["view", "for", "few", "(nada)"]),
    ("longo", 17.9, "if", ["if", "that", "when", "(nada)"]),
]

m = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")
_cache = {}


def preparar(nome):
    if nome not in _cache:
        audio = carregar(UNIDADES[nome])
        pasta = os.path.join(env.RAIZ, "saidas", os.path.splitext(os.path.basename(UNIDADES[nome]))[0])
        quadros = {c: acustica.medir(x) for c, x in zip(audio.nomes, audio.canais)}
        with open(os.path.join(pasta, "asr_large-v3_en.json"), encoding="utf-8") as f:
            passadas = {c: {k: [Palavra(**p) for p in v] for k, v in d.items()}
                        for c, d in json.load(f).items()}
        r = analise.analisar(audio, quadros, passadas)
        _cache[nome] = (audio, r, candidatos.alvos_da_analise(r))
    return _cache[nome]


ok = True
for unidade, ini, certo, textos in CASOS:
    audio, r, alvos = preparar(unidade)
    alvo = min(alvos, key=lambda x: abs(x["ini"] - ini))
    sinal = dict(zip(audio.nomes, audio.canais))[alvo["canal"]]
    ev = pontuacao.pontuar(m, "en", sinal, alvo, r["falas"], audio.duracao, textos)
    rot = pontuacao.classificar(ev, alvo["tipo"])
    favorecido = [t for t, v in rot.items() if v == "favorece"]
    caso_ok = (favorecido == [certo]) if certo == "Cool" else (favorecido in ([], [certo]))
    ok &= caso_ok
    print(f"\n{unidade} {alvo['tipo']} {alvo['ini']:.1f}s atual={alvo['atual']!r} (certo: {certo!r}) "
          f"→ {'OK' if caso_ok else 'ERRO'}")
    for t in sorted(ev, key=ev.get, reverse=True):
        print(f"   {t!r:12} evidência {ev[t]:7.2f}   {rot.get(t, 'fora da disputa')}")
print("\nRESULTADO:", "PASSOU" if ok else "FALHOU")
sys.exit(0 if ok else 1)
