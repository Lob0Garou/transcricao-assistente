"""DeepSeek V4.1 Flash como revisor de TEXTO, com rotas alternativas.

O DeepSeek não aceita áudio: ele não transcreve. Ele lê a transcrição do
Whisper com as dúvidas marcadas e sugere alternativas. Cada sugestão depois é
conferida contra o áudio (ta/candidatos.verificar_sugestoes) antes de virar
candidata, e o JEV ranqueia tudo.

Rotas, na ordem em que são tentadas (todas servem o mesmo modelo, V4.1 Flash):

  1. OpenCode Zen    https://opencode.ai/zen/v1                deepseek-v4.1-flash
  2. OpenRouter      https://openrouter.ai/api/v1              deepseek/deepseek-v4.1-flash
  3. NVIDIA          https://integrate.api.nvidia.com/v1       deepseek-ai/deepseek-v4.1-flash

A primeira que responder vale; as outras existem porque uma conta pode estar
sem saldo, um modelo pode sair do ar ou um endpoint pode dar timeout. Erros
conhecidos: 402 "Insufficient account funds" (conta Zen sem saldo), 403
"free tier can only be used from within OpenCode" (modelo grátis do Zen não
aceita chamada de fora), 402 "requires more credits, or fewer max_tokens"
(OpenRouter reserva crédito pelo max_tokens; o código rebaixa e tenta de novo).

Configuração (variáveis de ambiente, todas opcionais):
  DEEPSEEK_BASE            força UMA rota (padrão: tenta as três na ordem acima)
  DEEPSEEK_MODELO          modelo da rota forçada (padrão deepseek-v4.1-flash)
  DEEPSEEK_CHAVE_PROVEDOR  entrada do auth.json da rota forçada (padrão "opencode")
  DEEPSEEK_ROTAS           ordem das rotas, por nome: "opencode,openrouter,nvidia"
  DEEPSEEK_ESFORCO         padrão max
  DEEPSEEK_MAX_TOKENS      padrão 16000 (o OpenRouter reserva crédito por este número)
  DEEPSEEK_KEY_FILE        arquivo com a chave da PRIMEIRA rota, tentado antes do auth.json
  OPENCODE_AUTH            caminho do auth.json (padrão ~/.local/share/opencode/auth.json)

Nenhuma chave é impressa.
"""
import json
import os
import re
import urllib.error
import urllib.request
import uuid

ESFORCO = os.environ.get("DEEPSEEK_ESFORCO", "max")
MAX_TOKENS = int(os.environ.get("DEEPSEEK_MAX_TOKENS", "64000"))
MAX_CORTE = 128000   # teto do orçamento de saída quando a resposta vem cortada
AUTH = os.environ.get("OPENCODE_AUTH") or os.path.join(
    os.path.expanduser("~"), ".local", "share", "opencode", "auth.json")
ARQUIVO_CHAVE = os.environ.get("DEEPSEEK_KEY_FILE")

# "esforco"/"max_tokens" por rota: o OpenRouter RESERVA crédito pelo max_tokens, então
# lá o teto é menor de propósito (e o código reduz mais se a conta não cobrir).
ROTAS = [
    {"nome": "OpenCode Zen", "base": "https://opencode.ai/zen/v1",
     "modelo": "deepseek-v4.1-flash", "chave": "opencode",
     "esforco": ESFORCO, "max_tokens": MAX_TOKENS},
    {"nome": "OpenRouter", "base": "https://openrouter.ai/api/v1",
     "modelo": "deepseek/deepseek-v4.1-flash", "chave": "openrouter",
     "esforco": "high", "max_tokens": 16000},
    {"nome": "NVIDIA", "base": "https://integrate.api.nvidia.com/v1",
     "modelo": "deepseek-ai/deepseek-v4.1-flash", "chave": "nvidia",
     "esforco": "high", "max_tokens": 16000},
]

SISTEMA = {
    "pt": (
        "Você revisa transcrições automáticas (Whisper) de fala espontânea em português do Brasil. "
        "A transcrição deve ser LITERAL: mantenha \"é\", \"hã\", \"né\", \"tipo\", repetições, gírias e "
        "frases cortadas. Seu trabalho NÃO é melhorar o texto: é achar onde o reconhecedor provavelmente "
        "ouviu errado (palavra que não faz sentido ali, nome próprio deformado, homófono trocado, "
        "\"Obrigado\"/\"Legendas\" inventados) e propor o que a pessoa provavelmente DISSE."
    ),
    "en": (
        "You review automatic (Whisper) transcripts of spontaneous speech. The transcript must be "
        "VERBATIM: keep fillers, repetitions, slang and cut-off sentences. Your job is NOT to improve the "
        "text: find where the recognizer probably misheard (word that makes no sense there, mangled proper "
        "name, swapped homophone, invented \"Thank you\") and propose what the speaker most likely SAID."
    ),
    # Conversa com mais de um idioma (numa call, cada lado pode falar o seu).
    "misto": (
        "Você revisa transcrições automáticas (Whisper) de uma conversa que MISTURA idiomas — numa call, "
        "cada lado pode falar o seu. Cada trecho tem que ficar no idioma em que foi falado: NÃO traduza, "
        "nem normalize para um idioma só. A transcrição deve ser LITERAL: mantenha \"é\", \"hã\", \"né\", "
        "repetições, gírias e frases cortadas. Seu trabalho NÃO é melhorar o texto: é achar onde o "
        "reconhecedor provavelmente ouviu errado (palavra que não faz sentido ali, nome próprio deformado, "
        "homófono trocado, \"Obrigado\"/\"Thank you\" inventados) e propor o que a pessoa provavelmente "
        "DISSE, no idioma original do trecho."
    ),
}

PEDIDO = """Transcrição abaixo. Cada linha: [tempo] falante: texto.
Palavras duvidosas aparecem como [[ID:palavra]]; trechos com voz mas sem palavra reconhecida como [[ID:?]].

Para CADA ID, proponha até 3 alternativas do que foi dito, da mais para a menos provável. Inclua a
palavra atual entre as alternativas se ela parecer certa. Use "(nada)" se ali provavelmente não havia
palavra. Também aponte trechos SEM marcação que pareçam erro de reconhecimento.

Responda SOMENTE com JSON neste formato:
{"duvidas": [{"id": "a3", "alternativas": ["...", "..."], "motivo": "curto"}],
 "outros_erros": [{"trecho": "texto exato da transcrição", "sugestao": "...", "motivo": "curto"}]}

TRANSCRIÇÃO:
"""


def _chave_arquivo():
    """Chave avulsa (DEEPSEEK_KEY_FILE), no mesmo formato tolerante do JEV."""
    if not ARQUIVO_CHAVE:
        return None
    try:
        with open(ARQUIVO_CHAVE, encoding="utf-8") as f:
            bruto = f.read().strip()
    except OSError:
        return None
    for linha in filter(None, (x.strip() for x in bruto.splitlines())):
        m = re.search(r"(sk-[A-Za-z0-9_\-]{16,})", linha) or re.search(r"([A-Za-z0-9_\-]{24,})", linha)
        if m:
            return m.group(1)
    return None


def _chave_auth(entrada):
    try:
        with open(AUTH, encoding="utf-8") as f:
            return json.load(f).get(entrada, {}).get("key")
    except (OSError, ValueError):
        return None


def rotas(progresso=None):
    """Rotas a tentar, já com as chaves resolvidas (sem repetir a mesma chave)."""
    if os.environ.get("DEEPSEEK_BASE"):
        escolhidas = [{"nome": "rota fixada por DEEPSEEK_BASE",
                       "base": os.environ["DEEPSEEK_BASE"],
                       "modelo": os.environ.get("DEEPSEEK_MODELO", "deepseek-v4.1-flash"),
                       "chave": os.environ.get("DEEPSEEK_CHAVE_PROVEDOR", "opencode")}]
    else:
        pedidas = [x.strip() for x in os.environ.get("DEEPSEEK_ROTAS", "").split(",") if x.strip()]
        if pedidas:
            por_nome = {r["nome"].lower(): r for r in ROTAS}
            por_nome.update({r["chave"].lower(): r for r in ROTAS})
            escolhidas = [por_nome[p.lower()] for p in pedidas if p.lower() in por_nome]
            if not escolhidas:
                escolhidas = list(ROTAS)
        else:
            escolhidas = list(ROTAS)

    avulsa = _chave_arquivo()
    saida = []
    for i, r in enumerate(escolhidas):
        chaves = []
        if i == 0 and avulsa:
            chaves.append(("chave de DEEPSEEK_KEY_FILE", avulsa))
        do_auth = _chave_auth(r["chave"])
        if do_auth and all(do_auth != k for _, k in chaves):
            chaves.append((f"chave '{r['chave']}' do auth.json", do_auth))
        if not chaves:
            if progresso:
                progresso(f"  DeepSeek: sem chave para {r['nome']} ({r['chave']} ausente no auth.json)")
            continue
        saida.append({**r, "chaves": chaves})
    return saida


def montar_texto(r, alvos, rotulo):
    """Transcrição em texto corrido com as dúvidas marcadas [[ID:palavra]]."""
    por_palavra = {(x["fala"], x["pos"]): x for x in alvos if x["tipo"] == "palavra"}
    lacunas = [x for x in alvos if x["tipo"] == "lacuna"]
    linhas = []
    for fi, f in enumerate(r["falas"]):
        itens = [(p.ini, 0, por_palavra.get((fi, wi)), p.texto) for wi, p in enumerate(f["palavras"])]
        itens += [(x["ini"], 1, x, None) for x in lacunas
                  if x["canal"] == f["canal"] and f["ini"] - 0.6 < x["fim"] and x["ini"] < f["fim"] + 0.6]
        partes = []
        for _, _, alvo, texto in sorted(itens, key=lambda z: (z[0], z[1])):
            if alvo and alvo["tipo"] == "lacuna":
                partes.append(f"[[{alvo['id']}:?]]")
            elif alvo:
                partes.append(f"[[{alvo['id']}:{texto}]]")
            else:
                partes.append(texto)
        m, s = divmod(f["ini"], 60)
        linhas.append(f"[{int(m):02d}:{s:04.1f}] {rotulo(f['canal'])}: {' '.join(partes)}")
    return "\n".join(linhas)


def _extrair_json(texto):
    texto = re.sub(r"^```(?:json)?|```$", "", texto.strip(), flags=re.M).strip()
    ini, fim = texto.find("{"), texto.rfind("}")
    if ini < 0:
        # Sem isto, uma resposta vazia virava "0 dúvidas" e parecia sucesso.
        raise ValueError("resposta sem JSON")
    dados = json.loads(texto[ini:fim + 1])
    if not isinstance(dados, dict):
        raise ValueError("resposta não é um objeto")
    duvidas = dados.get("duvidas")
    outros = dados.get("outros_erros", [])
    if not isinstance(duvidas, list) or not isinstance(outros, list):
        raise ValueError("listas de revisão ausentes ou inválidas")
    for d in duvidas:
        if (not isinstance(d, dict) or not isinstance(d.get("id"), str)
                or not isinstance(d.get("alternativas", []), list)
                or not all(isinstance(a, str) for a in d.get("alternativas", []))
                or not isinstance(d.get("motivo", ""), str)):
            raise ValueError("dúvida inválida")
    for erro in outros:
        if not isinstance(erro, dict) or not all(isinstance(erro.get(k, ""), str)
                                               for k in ("trecho", "sugestao", "motivo")):
            raise ValueError("erro de revisão inválido")
    return dados


def _corpo(modelo, sistema, pedido, max_tokens, esforco):
    return {
        "model": modelo,
        "reasoning_effort": esforco,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": sistema}, {"role": "user", "content": pedido}],
    }


def _post(rota, chave, corpo, timeout):
    """Uma chamada. Devolve (dados, None) ou (None, motivo) — sem levantar exceção."""
    req = urllib.request.Request(
        rota["base"].rstrip("/") + "/chat/completions", data=json.dumps(corpo).encode("utf-8"),
        method="POST",
        headers={"Authorization": f"Bearer {chave}", "Content-Type": "application/json",
                 # Sem User-Agent próprio a Cloudflare do OpenCode recusa (erro 1010).
                 "User-Agent": "transcricao-assistente/0.1",
                 "x-opencode-session": str(uuid.uuid4())})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            dados = json.loads(resp.read().decode("utf-8"))
            escolhas = dados.get("choices") if isinstance(dados, dict) else None
            if (not isinstance(escolhas, list) or not escolhas
                    or not isinstance(escolhas[0], dict)
                    or not isinstance(escolhas[0].get("message"), dict)
                    or not isinstance(escolhas[0]["message"].get("content") or "", str)):
                return None, "resposta do provedor com formato inválido"
            if not isinstance(dados.get("usage") or {}, dict):
                dados["usage"] = {}
            return dados, None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}"
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        return None, f"falha de conexão ({e})"


def _explicar(motivo):
    if "funds" in motivo or "Balance" in motivo:
        return "conta sem saldo"
    if "free tier" in motivo:
        return "modelo grátis do Zen só funciona dentro do OpenCode"
    if "fewer max_tokens" in motivo:
        return "crédito insuficiente para o max_tokens pedido"
    return motivo


def revisar(r, alvos, idioma, rotulo, progresso=print, timeout=900):
    """Uma chamada para o áudio inteiro. Devolve {"duvidas": {...}, "outros_erros": [...]} ou erro.

    Percorre as rotas na ordem; a primeira que responder é usada. O motivo de cada
    falha volta em "tentativas" para aparecer no relatório.
    """
    sistema = SISTEMA.get(idioma, SISTEMA["en"])
    pedido = PEDIDO + montar_texto(r, alvos, rotulo)
    lista = rotas(progresso)
    if not lista:
        return {"usado": False, "motivo": "nenhuma chave de DeepSeek encontrada "
                                          f"(auth.json: {AUTH}; DEEPSEEK_KEY_FILE: {ARQUIVO_CHAVE or 'não definido'})"}

    tentativas = []
    for rota in lista:
        esforco = rota.get("esforco") or ESFORCO
        for origem, chave in rota["chaves"]:
            if len(lista) > 1 or len(rota["chaves"]) > 1:
                progresso(f"  DeepSeek: {rota['nome']} ({rota['modelo']}, esforço {esforco}, {origem})…")
            max_tokens = rota.get("max_tokens") or MAX_TOKENS
            for _ in range(4):
                dados, motivo = _post(rota, chave,
                                      _corpo(rota["modelo"], sistema, pedido, max_tokens, esforco), timeout)
                if dados is None:
                    # OpenRouter reserva crédito pelo max_tokens: ele diz quanto cabe.
                    m = re.search(r"afford (\d+)", motivo or "")
                    if m and int(m.group(1)) < max_tokens:
                        novo = max(1024, int(m.group(1)) - 64)
                        progresso(f"  DeepSeek: {rota['nome']} só cobre {m.group(1)} tokens de saída; "
                                  f"reduzindo de {max_tokens} para {novo}")
                        max_tokens = novo
                        if max_tokens < 12000 and esforco != "low":
                            # Pouco espaço de saída: com esforço máximo o raciocínio come tudo.
                            esforco = "low"
                            progresso(f"  DeepSeek: pouco espaço para raciocínio; baixando o esforço para {esforco}")
                        continue
                    curto = _explicar(motivo)
                    progresso(f"  DeepSeek: {rota['nome']} falhou ({curto})")
                    tentativas.append({"rota": rota["nome"], "origem": origem, "motivo": curto})
                    break

                escolha = dados["choices"][0]
                conteudo = escolha["message"].get("content") or ""
                uso = dados.get("usage") or {}
                # Resposta cortada: com esforço alto o raciocínio consome o orçamento inteiro
                # e o JSON nunca sai. Medido: 16000 tokens de saída, 16000 de raciocínio, zero JSON.
                if escolha.get("finish_reason") == "length" or (not conteudo.strip() and "{" not in conteudo):
                    if max_tokens < MAX_CORTE:
                        novo = min(MAX_CORTE, max_tokens * 2)
                        progresso(f"  DeepSeek: resposta cortada (gastou {uso.get('completion_tokens')} tokens, "
                                  f"{uso.get('completion_tokens_details', {}).get('reasoning_tokens')} raciocinando); "
                                  f"repetindo com {novo}")
                        max_tokens = novo
                        continue
                    if esforco != "low":
                        esforco = "low"
                        progresso(f"  DeepSeek: sem espaço para aumentar; baixando o esforço para {esforco}")
                        continue
                    progresso(f"  DeepSeek: {rota['nome']} não devolveu JSON")
                    tentativas.append({"rota": rota["nome"], "origem": origem,
                                       "motivo": "resposta cortada sem JSON"})
                    break

                try:
                    saida = _extrair_json(conteudo)
                except ValueError:
                    progresso(f"  DeepSeek: {rota['nome']} respondeu sem JSON válido")
                    tentativas.append({"rota": rota["nome"], "origem": origem,
                                       "motivo": "resposta sem JSON válido"})
                    break
                duvidas = {d["id"]: [a.strip() for a in d.get("alternativas", []) if a and a.strip()][:3]
                           for d in saida.get("duvidas", []) if isinstance(d, dict) and "id" in d}
                motivos = {d["id"]: d.get("motivo", "") for d in saida.get("duvidas", [])
                           if isinstance(d, dict) and "id" in d}
                progresso(f"  DeepSeek: {len(duvidas)} dúvidas comentadas via {rota['nome']}, "
                          f"{len(saida.get('outros_erros', []))} outros possíveis erros")
                return {"usado": True, "rota": rota["nome"], "base": rota["base"], "modelo": rota["modelo"],
                        "esforco": esforco, "max_tokens": max_tokens, "origem_chave": origem,
                        "duvidas": duvidas, "motivos": motivos,
                        "outros_erros": saida.get("outros_erros", []),
                        "uso": uso, "tentativas": tentativas}

    motivo = "; ".join(f"{t['rota']}: {t['motivo']}" for t in tentativas) or "nenhuma rota disponível"
    progresso(f"  DeepSeek: nenhuma rota respondeu — {motivo}")
    return {"usado": False, "motivo": motivo, "tentativas": tentativas}
