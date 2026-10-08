"""Turnos de fala e palpite de locutor.

O Whisper junta palavras em "falas" com pausa de 0,8 s. Numa conversa isso dá
dois erros opostos: duas pessoas caem no mesmo bloco (a pausa entre a pergunta e
a resposta é menor que 0,8 s) e uma resposta longa de uma pessoa só vira um bloco
gigante (ela pausa 0,5 s no meio da frase). Por isso a pausa sozinha não separa
nada: aqui a fala é cortada em pedaços finos e cada pedaço recebe um palpite de
locutor pelas características da voz, e os pedaços vizinhos do MESMO locutor são
juntados de volta.

O palpite é palpite. Só é afirmado quando os dois grupos são claros e o tom
difere o bastante; fora disso o pedaço fica sem locutor ("não deu para separar")
e quem decide é você, na página de revisão.

Sem dependência nova: usa as medidas que ta/acustica.py já calcula por quadro.
"""
import numpy as np

from . import acustica

PAUSA = 0.35          # s: pausa que corta um pedaço
MIN_VOZ = 0.50        # s de voz para o pedaço valer como evidência E receber palpite
MIN_PEDACOS = 4       # menos que isso não dá para separar com segurança
SEPARACAO_MIN = 0.14  # diferença mínima de tom entre os grupos (log natural ≈ 15%)
JUNTA = 1.2           # s: pedaços do mesmo locutor até esta pausa viram um bloco
JUNTA_SEM = 0.8       # s: sem palpite, mantém o agrupamento antigo (pausa de 0,8 s)


def dividir(falas, pausa=PAUSA):
    """Corta cada fala nas pausas >= pausa. Devolve pedaços com canal, tempos e palavras."""
    pedacos = []
    for f in falas:
        atual = []
        for p in f["palavras"]:
            if atual and p.ini - atual[-1].fim >= pausa:
                pedacos.append(_pedaco(f["canal"], atual))
                atual = []
            atual.append(p)
        if atual:
            pedacos.append(_pedaco(f["canal"], atual))
    return pedacos


def _pedaco(canal, palavras):
    return {"canal": canal, "ini": palavras[0].ini, "fim": max(p.fim for p in palavras),
            "palavras": list(palavras), "texto": " ".join(p.texto for p in palavras),
            "locutor": None, "confianca": 0.0, "motivo": ""}


def medir(pedacos, quadros):
    """Preenche as medidas de voz de cada pedaço (as mesmas usadas para os sons)."""
    for x in pedacos:
        q = quadros[x["canal"]]
        d = acustica.descrever(q, x["ini"], x["fim"])
        x["voz"], x["f0"], x["f0_var"] = d["voz"], d["f0"], d["f0_var"]
        x["planura"], x["dur_ativa"] = d["planura"], d["dur_ativa"]
        # Quantos quadros realmente vozeados: é o peso do pedaço na separação e o
        # aviso de que o tom dele é confiável (pedaço curto erra oitava).
        x["dur_voz"] = round(float(q.vozeado[acustica.fatia(q, x["ini"], x["fim"])].sum())
                             / acustica.FPS, 2)
    return pedacos


def _kmeans(X, pesos, iteracoes=30):
    """k-means 2-D determinístico (k=2), ponderado, inicializado pelos quartis do tom."""
    ordem = np.argsort(X[:, 0])
    centros = np.stack([X[ordem[len(ordem) // 4]], X[ordem[(3 * len(ordem)) // 4]]])
    grupos = np.zeros(len(X), dtype=int)
    for _ in range(iteracoes):
        d = ((X[:, None, :] - centros[None, :, :]) ** 2).sum(2)
        novos_grupos = d.argmin(1)
        novos = np.stack([np.average(X[novos_grupos == k], axis=0, weights=pesos[novos_grupos == k])
                          if (novos_grupos == k).any() else centros[k] for k in range(2)])
        if np.array_equal(novos_grupos, grupos) and np.allclose(novos, centros):
            break
        grupos, centros = novos_grupos, novos
    return grupos


def _perto_do_centro(f0, centros):
    """Distância em log do tom, com correção de oitava só quando o pedaço é curto.

    Em pedaço curto a autocorrelação às vezes trava no harmônico e devolve o dobro
    do tom real. Como o erro só aparece onde há pouca voz para medir, a correção
    fica restrita a esses casos — em pedaço longo o tom medido é confiável.
    """
    alvo = np.log(f0)
    d = np.abs(alvo - centros)
    if f0 > 1.8 * np.exp(centros.min()):
        d = np.minimum(d, np.abs(np.log(f0 / 2) - centros))
    return d


def palpitar(pedacos):
    """Agrupa os pedaços em 2 locutores pelo tom da voz. Devolve quantos receberam palpite.

    Conservador de propósito: exige pedaços suficientes, os dois grupos com tamanho
    razoável e uma diferença de tom clara. Sem isso, ninguém recebe palpite.
    """
    com_tom = [x for x in pedacos if x.get("f0", 0) > 0]
    # Só pedaço com voz suficiente vira evidência E recebe palpite. Abaixo disso o
    # tom medido é instável (medido: 0,3 s de voz deu 269 Hz e 333 Hz na mesma voz
    # que em pedaço longo mede ~200 Hz), então o pedaço fica sem locutor e a página
    # pede a sua decisão em vez de chutar.
    bons = [x for x in com_tom if x.get("dur_voz", 0) >= MIN_VOZ]
    for x in com_tom:
        if x.get("dur_voz", 0) < MIN_VOZ:
            x["motivo"] = "pouca voz para medir o tom: decida você"
    if len(bons) < MIN_PEDACOS:
        return 0

    f0 = np.array([x["f0"] for x in bons])
    pesos = np.array([x["dur_voz"] for x in bons])
    # Só o tom decide. A planura espectral foi testada como segunda característica e
    # PIOROU: ela varia 0,027–0,048 entre pedaços (é ruído) e arrastava pedaços de
    # uma voz para o grupo da outra, derrubando a pureza de 1,00 para 0,73. Com o
    # tom sozinho as duas vozes deste áudio ficam sem sobreposição (máx. 134 Hz num
    # grupo, mín. 176 Hz no outro).
    X = np.column_stack([np.log(f0)])
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)

    grupos = _kmeans(X, pesos)
    n0, n1 = int((grupos == 0).sum()), int((grupos == 1).sum())
    if min(n0, n1) < max(2, int(0.25 * len(bons))):
        return 0

    med0, med1 = float(np.median(f0[grupos == 0])), float(np.median(f0[grupos == 1]))
    grave, agudo = sorted([med0, med1])
    separacao = float(np.log(agudo / grave)) if grave > 0 else 0.0
    if separacao < SEPARACAO_MIN:
        return 0

    # Quão puro é o corte: cada pedaço tem que estar mais perto do seu grupo.
    limite = (np.log(grave) + np.log(agudo)) / 2
    do_lado = (np.log(f0) < limite).astype(int)
    if min((do_lado == 0).sum(), (do_lado == 1).sum()) == 0:
        return 0
    puro = max((do_lado == grupos).mean(), (do_lado != grupos).mean())
    if puro < 0.85:
        return 0

    # Grupo 1 é sempre o mais grave, para o rótulo não trocar de áudio para áudio.
    do_grave = 0 if med0 <= med1 else 1
    centros = np.array([np.log(med0), np.log(med1)])
    confianca = round(min(1.0, separacao / 0.45) * puro, 2)
    motivo = f"tom mais grave ({int(grave)} Hz) contra mais agudo ({int(agudo)} Hz)"
    for x in bons:
        g = int(np.argmin(_perto_do_centro(x["f0"], centros)))
        x["locutor"] = 1 if g == do_grave else 2
        x["confianca"] = confianca
        x["motivo"] = motivo
    return len(bons)


def agrupar(pedacos, junta=JUNTA, junta_sem=JUNTA_SEM):
    """Junta pedaços vizinhos do mesmo locutor (ou, sem palpite, os do mesmo canal).

    Sem isso a página mostraria migalhas de 1 s em vez de falas.
    """
    blocos = []
    for x in pedacos:
        ant = blocos[-1] if blocos else None
        mesmo = ant is not None and ant["canal"] == x["canal"]
        if mesmo and ant["locutor"] is not None and ant["locutor"] == x["locutor"]:
            mesmo = x["ini"] - ant["fim"] < junta
        elif mesmo and ant["locutor"] is None and x["locutor"] is None:
            mesmo = x["ini"] - ant["fim"] < junta_sem
        else:
            mesmo = False
        if mesmo:
            ant["fim"] = x["fim"]
            ant["palavras"] += x["palavras"]
            ant["texto"] += " " + x["texto"]
            ant["pedacos"] += 1
            ant["confianca"] = min(ant["confianca"], x["confianca"])
        else:
            blocos.append({"canal": x["canal"], "ini": x["ini"], "fim": x["fim"],
                           "palavras": list(x["palavras"]), "texto": x["texto"],
                           "locutor": x["locutor"], "confianca": x["confianca"],
                           "motivo": x["motivo"], "pedacos": 1})
    return blocos


def analisar(falas, quadros, pausa=PAUSA):
    """Atalho: corta, mede, palpita e agrupa. Devolve (blocos, pedacos)."""
    pedacos = medir(dividir(falas, pausa), quadros)
    com_palpite = palpitar(pedacos)
    return agrupar(pedacos), {"pedacos": len(pedacos), "com_palpite": com_palpite}
