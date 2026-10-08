"""Carrega o áudio com ffmpeg e descreve a estrutura de canais."""
import json
import subprocess
from dataclasses import dataclass

import numpy as np

SR = 16000


@dataclass
class Audio:
    caminho: str
    canais: list          # lista de arrays float32 a 16 kHz
    nomes: list           # "E"/"D" para estéreo, "M" para mono
    duracao: float
    independentes: bool   # True = um microfone por canal (sem vazamento)
    correlacao: float


def _probe(caminho):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=channels", "-of", "json", caminho],
        capture_output=True, text=True, check=True).stdout
    return json.loads(out)["streams"][0]["channels"]


def carregar(caminho):
    n = _probe(caminho)
    n = 2 if n >= 2 else 1
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", caminho, "-ac", str(n), "-ar", str(SR),
         "-f", "f32le", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.float32).reshape(-1, n)
    dur = len(x) / SR
    if n == 1:
        return Audio(caminho, [x[:, 0].copy()], ["M"], dur, False, 1.0)
    e, d = x[:, 0].copy(), x[:, 1].copy()
    if np.std(e) < 1e-6 or np.std(d) < 1e-6:
        corr = 0.0
    else:
        corr = float(np.corrcoef(e, d)[0, 1])
    if abs(corr) > 0.98:  # estéreo falso: os dois canais são iguais
        return Audio(caminho, [(e + d) / 2], ["M"], dur, False, corr)
    # Correlação baixa = microfones separados. Correlação média = mixagem estéreo
    # comum (a mesma voz aparece nos dois lados), que deve ser tratada como mono.
    indep = abs(corr) < 0.2
    if not indep:
        return Audio(caminho, [(e + d) / 2], ["M"], dur, False, corr)
    return Audio(caminho, [e, d], ["E", "D"], dur, True, corr)
