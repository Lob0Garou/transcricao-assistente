"""Medidas acústicas por quadro: energia, vozeamento, F0 e planura espectral.

Separa "tem som" de "tem voz". Foi essa distinção que desmascarou os
"Thank you" falsos nos testes: cliques e digitação têm energia, mas não têm
harmônicos.
"""
from dataclasses import dataclass

import numpy as np

from .audio import SR

JAN = 640   # 40 ms
PASSO = 160  # 10 ms -> 100 quadros por segundo
FPS = SR // PASSO


@dataclass
class Quadros:
    db: np.ndarray        # energia em dBFS
    vozeado: np.ndarray   # bool: tem periodicidade de voz
    f0: np.ndarray        # Hz (0 quando não vozeado)
    planura: np.ndarray   # 0 = tonal, 1 = ruído branco
    piso: float           # nível de ruído de fundo (dB)
    limiar: float         # acima disso conta como atividade


def medir(x):
    n = max(0, (len(x) - JAN) // PASSO + 1)
    idx = np.arange(JAN)[None, :] + PASSO * np.arange(n)[:, None]
    F = x[idx] * np.hanning(JAN)[None, :]
    rms = np.sqrt((F ** 2).mean(1)) + 1e-9
    db = 20 * np.log10(rms)

    spec = np.abs(np.fft.rfft(F, n=2 * JAN, axis=1)) ** 2
    ac = np.fft.irfft(spec, axis=1)[:, :JAN]
    ac = ac / (ac[:, :1] + 1e-12)
    lo, hi = SR // 400, SR // 70           # F0 entre 70 e 400 Hz
    k = np.argmax(ac[:, lo:hi], axis=1) + lo
    pico = ac[np.arange(n), k]

    mag = np.sqrt(spec[:, : JAN + 1]) + 1e-12
    planura = np.exp(np.log(mag).mean(1)) / mag.mean(1)

    piso = float(np.percentile(db, 10))
    limiar = max(piso + 15, -50.0)
    vozeado = (pico > 0.5) & (db > limiar)
    f0 = np.where(vozeado, SR / k, 0.0)
    return Quadros(db, vozeado, f0, planura, piso, limiar)


def regioes(ativo, junta=0.3, minimo=0.1):
    """Converte um vetor booleano por quadro em intervalos [ini, fim] em segundos."""
    segs, i, n = [], 0, len(ativo)
    while i < n:
        if ativo[i]:
            j = i
            while j < n and ativo[j]:
                j += 1
            segs.append([i / FPS, j / FPS])
            i = j
        else:
            i += 1
    unidos = []
    for s in segs:
        if unidos and s[0] - unidos[-1][1] < junta:
            unidos[-1][1] = s[1]
        else:
            unidos.append(s)
    return [s for s in unidos if s[1] - s[0] >= minimo]


def fatia(q, ini, fim):
    a = max(0, int(ini * FPS))
    return slice(a, max(a + 1, int(fim * FPS)))


def descrever(q, ini, fim):
    """Resumo de um trecho: quanto é voz, F0 mediano, planura, impulsividade."""
    s = fatia(q, ini, fim)
    db = q.db[s]
    ativo = db > q.limiar
    if not ativo.any():
        return {"voz": 0.0, "f0": 0.0, "f0_var": 0.0, "planura": 0.0, "pulsos": 0,
                "dur_ativa": 0.0}
    voz = float(q.vozeado[s][ativo].mean())
    f0s = q.f0[s][q.vozeado[s]]
    # Pulsos: subidas rápidas de energia (cliques, teclas, tosse, riso).
    subidas = np.diff(db) > 12
    med = float(np.median(f0s)) if len(f0s) else 0.0
    return {
        "voz": round(voz, 2),
        "f0": round(med, 0),
        # Voz humana varia o tom; zumbido de máquina fica parado.
        "f0_var": round(float(np.std(f0s) / med), 3) if len(f0s) > 5 else 0.0,
        "planura": round(float(np.median(q.planura[s][ativo])), 3),
        "pulsos": int(subidas.sum()),
        "dur_ativa": round(float(ativo.sum()) / FPS, 2),
    }


def f0_tipico(q):
    v = q.f0[q.vozeado]
    return float(np.median(v)) if len(v) else 0.0


def classificar_som(d, dur, f0_canal=0.0):
    """Palpite para um trecho com energia, mas sem palavra transcrita.

    São heurísticas. A página de revisão mostra o palpite como candidato,
    nunca como fato.
    """
    if d["dur_ativa"] < 0.05:
        return None
    if d["dur_ativa"] < 0.15:
        return "estalo/clique"  # curto demais para sílaba; teclas também "vibram"
    if d["dur_ativa"] > 1.5 and d["voz"] >= 0.3 and 0 < d["f0_var"] < 0.06:
        return "zumbido/ruído tonal"
    if d["voz"] >= 0.6:
        # Riso: pulsos vozeados e tom acima do normal da pessoa.
        if d["pulsos"] >= 3 and dur < 3 and f0_canal and d["f0"] > 1.3 * f0_canal:
            return "riso?"
        return "vocalização sem palavra (hum/murmúrio/filler?)"
    if d["dur_ativa"] < 0.25 and d["pulsos"] <= 2 and d["voz"] < 0.3:
        return "estalo/clique"
    if d["pulsos"] >= 4 and d["dur_ativa"] / max(dur, 1e-3) < 0.5:
        return "cliques/digitação"
    if 0.15 <= d["dur_ativa"] <= 1.2 and d["planura"] >= 0.07 and d["voz"] < 0.5:
        return "tosse/pigarro?"
    if d["dur_ativa"] > 2:
        return "ruído contínuo/fundo"
    return "ruído"
