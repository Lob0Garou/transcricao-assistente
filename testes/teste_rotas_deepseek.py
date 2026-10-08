"""Testa a escolha de rotas do revisor DeepSeek, sem chamar nenhuma API.

Verifica: ordem padrão das rotas, chave avulsa (DEEPSEEK_KEY_FILE) tentada antes
do auth.json, DEEPSEEK_ROTAS reordenando, DEEPSEEK_BASE fixando uma rota só, rota
sem chave sendo descartada, e a tradução dos erros reais já vistos em produção
(402 "Insufficient account funds", 403 do modelo grátis do Zen, 402 do OpenRouter
por max_tokens).

Uso: python -s testes/teste_rotas_deepseek.py
"""
import importlib
import json
import os
import sys
import tempfile

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.stdout.reconfigure(encoding="utf-8")

from ta import deepseek  # noqa: E402

VARIAS = ("DEEPSEEK_BASE", "DEEPSEEK_MODELO", "DEEPSEEK_CHAVE_PROVEDOR",
          "DEEPSEEK_ROTAS", "DEEPSEEK_KEY_FILE", "OPENCODE_AUTH")

tmp = tempfile.mkdtemp(prefix="rotas-deepseek-")
auth_falso = os.path.join(tmp, "auth.json")
with open(auth_falso, "w", encoding="utf-8") as f:
    json.dump({"opencode": {"type": "api", "key": "sk-falsa-opencode-000000000000"},
               "openrouter": {"type": "api", "key": "sk-or-falsa-0000000000000000"},
               "nvidia": {"type": "api", "key": "nvapi-falsa-0000000000000000"}}, f)

arquivo_chave = os.path.join(tmp, "chave.txt")
with open(arquivo_chave, "w", encoding="utf-8") as f:
    f.write("sk-chave-avulsa-com-tamanho-suficiente\n")

falhas = []


def carregar(**env):
    for k in VARIAS:
        os.environ.pop(k, None)
    os.environ["OPENCODE_AUTH"] = auth_falso
    for k, v in env.items():
        if v is not None:
            os.environ[k] = v
    return importlib.reload(deepseek)


def checar(rotulo, obtido, esperado):
    ok = obtido == esperado
    if not ok:
        falhas.append(f"{rotulo}: obtido {obtido!r}, esperado {esperado!r}")
    print(f"  {'OK  ' if ok else 'ERRO'} {rotulo}: {obtido!r}")


print("1. ordem padrão")
m = carregar()
r = m.rotas()
checar("nomes", [x["nome"] for x in r], ["OpenCode Zen", "OpenRouter", "NVIDIA"])
checar("modelos", [x["modelo"] for x in r],
       ["deepseek-v4.1-flash", "deepseek/deepseek-v4.1-flash", "deepseek-ai/deepseek-v4.1-flash"])
checar("uma chave por rota", [len(x["chaves"]) for x in r], [1, 1, 1])
checar("origem das chaves", [x["chaves"][0][0] for x in r],
       ["chave 'opencode' do auth.json", "chave 'openrouter' do auth.json", "chave 'nvidia' do auth.json"])

print("\n2. DEEPSEEK_KEY_FILE entra antes do auth.json, só na primeira rota")
m = carregar(DEEPSEEK_KEY_FILE=arquivo_chave)
r = m.rotas()
checar("chaves da 1a rota", [o for o, _ in r[0]["chaves"]],
       ["chave de DEEPSEEK_KEY_FILE", "chave 'opencode' do auth.json"])
checar("valor da chave avulsa", r[0]["chaves"][0][1], "sk-chave-avulsa-com-tamanho-suficiente")
checar("2a rota sem chave avulsa", [o for o, _ in r[1]["chaves"]], ["chave 'openrouter' do auth.json"])

print("\n3. DEEPSEEK_ROTAS muda a ordem")
m = carregar(DEEPSEEK_ROTAS="openrouter,opencode")
checar("nomes", [x["nome"] for x in m.rotas()], ["OpenRouter", "OpenCode Zen"])

print("\n4. DEEPSEEK_BASE fixa uma rota só")
m = carregar(DEEPSEEK_BASE="https://exemplo.test/v1")
r = m.rotas()
checar("quantidade", len(r), 1)
checar("base", r[0]["base"], "https://exemplo.test/v1")
checar("modelo padrão", r[0]["modelo"], "deepseek-v4.1-flash")

print("\n5. rota sem chave no auth.json é descartada com aviso")
sem_nvidia = os.path.join(tmp, "auth-sem-nvidia.json")
with open(sem_nvidia, "w", encoding="utf-8") as f:
    json.dump({"opencode": {"type": "api", "key": "sk-falsa-opencode-000000000000"}}, f)
m = carregar(OPENCODE_AUTH=sem_nvidia)
avisos = []
r = m.rotas(progresso=avisos.append)
checar("rotas restantes", [x["nome"] for x in r], ["OpenCode Zen"])
checar("aviso emitido", len(avisos), 2)

print("\n6. tradução dos erros reais")
casos = [
    ('HTTP 402: {"error":{"type":"server_error","message":"Upstream request failed: Insufficient account funds"}}',
     "conta sem saldo"),
    ('HTTP 402: {"error":{"message":"Insufficient Balance (request_id: abc)"}}', "conta sem saldo"),
    ('HTTP 403: {"error":{"type":"FreeTierError","message":"OpenCode\'s free tier can only be used from within OpenCode"}}',
     "modelo grátis do Zen só funciona dentro do OpenCode"),
    ('HTTP 402: {"error":{"message":"This request requires more credits, or fewer max_tokens. You requested up to 64000 tokens, but can only afford 8416."}}',
     "crédito insuficiente para o max_tokens pedido"),
]
for corpo, esperado in casos:
    checar(esperado, m._explicar(corpo), esperado)

print("\n7. quanto o OpenRouter diz que cabe é lido do erro")
import re  # noqa: E402
achado = re.search(r"afford (\d+)", casos[3][0])
checar("tokens que cabem", int(achado.group(1)) if achado else None, 8416)

print("\nRESULTADO:", "PASSOU" if not falhas else "FALHOU")
for f in falhas:
    print("  -", f)
sys.exit(1 if falhas else 0)
