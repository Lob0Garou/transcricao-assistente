"""Transcrição com faster-whisper em duas passadas independentes.

- Passada "inteira": o canal todo, com VAD do Whisper. Tem mais contexto.
- Passada "trechos": cada região de atividade isolada, sem VAD. Pega falas
  curtas que a passada inteira engole (backchannels, "é", "uhum").

Onde as duas discordam, a página de revisão manda ouvir.
"""
from dataclasses import dataclass

import numpy as np

from .audio import SR

# Prompt com disfluências: puxa o Whisper para o literal, em vez de limpar
# "é", "hã", "né" e repetições (o padrão dele é apagar tudo isso).
PROMPT = {
    "pt": "Então, é... hã, tipo, né? Aí, hum, é, é isso. Uhum. Ah, tá.",
    "en": "Um, uh, yeah, mm-hmm. So, like, you know.",
    # Sem idioma escolhido: empurrão para o literal nos dois idiomas, sem forçar nenhum
    # (medido: com este prompt a detecção continuou 99% correta).
    "misto": "É, hã, né, tipo... Um, uh, yeah, mm-hmm, you know.",
}


@dataclass
class Palavra:
    texto: str
    ini: float
    fim: float
    prob: float
    canal: str
    passada: str


def carregar_modelo(modelo):
    """Usa GPU quando disponível; CPU também serve candidatos e alinhamento."""
    from faster_whisper import WhisperModel
    try:
        return WhisperModel(modelo, device="cuda", compute_type="int8_float16"), "cuda"
    except (RuntimeError, ValueError):
        return WhisperModel(modelo, device="cpu", compute_type="int8"), "cpu"


class Transcritor:
    """idioma=None (ou "auto") faz o Whisper detectar — por canal, separadamente.

    Numa gravação de call cada lado pode falar um idioma. Forçar um só faz o
    Whisper TRADUZIR o outro lado: medido no áudio do Yuri, forçar "pt" num canal
    que é inglês devolveu 22 marcadores de português e ZERO de inglês.
    """

    def __init__(self, modelo="large-v3", idioma=None):
        self.m, self.dispositivo = carregar_modelo(modelo)
        self.idioma = None if idioma in (None, "", "auto") else idioma
        self.detectados = {}   # canal -> {idioma: segundos de fala}

    def _rodar(self, x, vad, canal=None):
        segs, info = self.m.transcribe(
            x, language=self.idioma, word_timestamps=True, vad_filter=vad,
            beam_size=5, condition_on_previous_text=False,
            initial_prompt=PROMPT.get(self.idioma or "misto"), no_speech_threshold=0.9)
        # Só a passada inteira vota no idioma: trecho curto erra a detecção.
        if self.idioma is None and canal is not None and getattr(info, "language", None):
            votos = self.detectados.setdefault(canal, {})
            votos[info.language] = votos.get(info.language, 0.0) + len(x) / SR
        return [(w.word.strip(), w.start, w.end, w.probability)
                for s in segs for w in s.words if w.word.strip()]

    def idioma_do_canal(self, canal):
        """Idioma detectado no canal (o de maior tempo de fala). None se não deu."""
        if self.idioma:
            return self.idioma
        votos = self.detectados.get(canal) or {}
        return max(votos, key=votos.get) if votos else None

    def idiomas(self, canais):
        return {c: self.idioma_do_canal(c) for c in canais}

    def inteira(self, x, canal):
        return [Palavra(t, a, b, p, canal, "inteira") for t, a, b, p in self._rodar(x, True, canal)]

    def trechos(self, x, canal, regioes):
        blocos, cur = [], None
        for a, b in regioes:
            if cur and a - cur[1] < 1.0 and b - cur[0] < 15:
                cur[1] = b
            else:
                if cur:
                    blocos.append(cur)
                cur = [a, b]
        if cur:
            blocos.append(cur)
        out = []
        dur = len(x) / SR
        for a, b in blocos:
            a0, b0 = max(0.0, a - 0.3), min(dur, b + 0.3)
            y = np.ascontiguousarray(x[int(a0 * SR):int(b0 * SR)])
            for t, s, e, p in self._rodar(y, False):
                out.append(Palavra(t, s + a0, e + a0, p, canal, "trechos"))
        return out
