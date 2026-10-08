"""Página local para anexar áudio e acompanhar a transcrição.

Uso: python -s app.py   (ou dois cliques em "Abrir transcricao.bat")
Abre em http://127.0.0.1:8790. Só aceita conexões da própria máquina.
"""
import html
import hashlib
import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RAIZ = os.path.dirname(os.path.abspath(__file__))
ENTRADAS = os.path.join(RAIZ, "entradas")
SAIDAS = os.path.join(RAIZ, "saidas")
PORTA = 8790
MAX_BYTES = 1024 * 1024 * 1024  # 1 GB

mimetypes.add_type("audio/mp4", ".m4a")
JOBS = {}
FILA = threading.Lock()  # uma transcrição por vez: a GPU tem 6 GB
HASHES = {}              # sha256 do áudio -> {"feito": bool, "url": str} — evita transcrever o mesmo arquivo duas vezes


def _sha256(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        while bloco := f.read(1 << 20):
            h.update(bloco)
    return h.hexdigest()


def _lembrar_resultados():
    """Reconstrói a memória de áudios já transcritos a partir das pastas de saída."""
    if not os.path.isdir(SAIDAS):
        return
    for d in os.listdir(SAIDAS):
        p = os.path.join(SAIDAS, d, "origem.json")
        try:
            with open(p, encoding="utf-8") as f:
                origem = json.load(f)
        except (OSError, ValueError):
            continue
        if origem.get("hash"):
            HASHES[origem["hash"]] = {"feito": True, "pasta": d,
                                      "url": f"/saidas/{urllib.parse.quote(d)}/revisao.html"}


def _seguro(nome):
    base, ext = os.path.splitext(os.path.basename(nome))
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")[:80] or "audio"
    return base + re.sub(r"[^A-Za-z0-9.]", "", ext)[:8]


def _rodar(job_id, caminho, idioma, deepseek, jev, pasta=None, refazer=False):
    job = JOBS[job_id]
    if FILA.locked():
        na_frente = sum(1 for j in JOBS.values() if not j.get("feito") and j is not job)
        job["log"].append(f"na fila: {na_frente} trabalho(s) à frente — o app transcreve um áudio por vez "
                          f"para não estourar a GPU. Pode deixar a aba aberta.")
    if pasta:
        job["log"].append("mesmo áudio de antes: reaproveitando a transcrição já feita"
                          + (" — refazendo do zero, como pedido" if refazer else " (deve ficar pronto em segundos)"))
    with FILA:
        job["iniciado"] = time.time()
        job["log"].append("começando…")
        cmd = [sys.executable, "-s", "-u", os.path.join(RAIZ, "transcrever.py"), caminho, "--idioma", idioma]
        if pasta:
            # Caminho ABSOLUTO: o subprocesso roda com cwd na raiz do projeto, então um
            # nome relativo criaria a pasta no lugar errado e perderia o cache.
            cmd += ["--saida", os.path.join(SAIDAS, pasta)]
        if refazer:
            cmd.append("--refazer")
        if not deepseek:
            cmd.append("--sem-deepseek")
        if not jev:
            cmd.append("--sem-jev")
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        t0 = time.time()
        proc = subprocess.Popen(cmd, cwd=RAIZ, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", env=env)
        for linha in proc.stdout:
            linha = linha.rstrip()
            if not linha or "Warning" in linha or "warn(" in linha:
                continue
            job["log"].append(linha)
        proc.wait()
        job["segundos"] = round(time.time() - t0)
        pagina = next((l for l in reversed(job["log"]) if l.endswith("revisao.html")), None)
        if proc.returncode == 0 and pagina:
            pasta = os.path.basename(os.path.dirname(pagina))
            job["url"] = f"/saidas/{urllib.parse.quote(pasta)}/revisao.html"
            # Guarda o que gerou esta pasta: permite reconhecer o mesmo áudio depois.
            try:
                with open(os.path.join(SAIDAS, pasta, "origem.json"), "w", encoding="utf-8") as f:
                    json.dump({"hash": job["hash"], "arquivo": job["nome"], "idioma": idioma,
                               "segundos": job["segundos"], "quando": time.strftime("%Y-%m-%d %H:%M:%S")},
                              f, ensure_ascii=False, indent=1)
            except OSError:
                pass
        else:
            job["erro"] = f"o processo terminou com código {proc.returncode}"
        job["feito"] = True
        HASHES[job["hash"]] = {"feito": True, "pasta": pasta if job.get("url") else None,
                               "url": job.get("url")}


def _resultados():
    if not os.path.isdir(SAIDAS):
        return []
    itens = []
    for d in os.listdir(SAIDAS):
        p = os.path.join(SAIDAS, d, "revisao.html")
        if os.path.exists(p):
            itens.append((os.path.getmtime(p), d))
    return [d for _, d in sorted(itens, reverse=True)[:30]]


PAGINA = """<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Transcrição Assistente</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--tx:#1d1d1b;--mut:#6b6b66;--bd:#e3e3de;--ac:#2f5bd3;--ok:#2e8b57;--err:#c0392b}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#151514;--card:#1e1e1c;--tx:#ecece8;
--mut:#9a9a93;--bd:#33332f;--ac:#7d9bff;--ok:#5cc28a;--err:#ff7b6e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
.wrap{max-width:760px;margin:0 auto;padding:24px 16px}h1{font-size:20px;margin:0 0 4px}.mut{color:var(--mut);font-size:13px}
#zona{margin:18px 0;border:2px dashed var(--bd);border-radius:12px;padding:36px 16px;text-align:center;background:var(--card);cursor:pointer}
#zona.sobre{border-color:var(--ac)}#zona b{font-size:16px}
.op{display:flex;flex-wrap:wrap;gap:14px;align-items:center;margin:8px 0 4px}
select,button{font:inherit;border:1px solid var(--bd);background:var(--card);color:var(--tx);border-radius:6px;padding:4px 10px}
button.pri{background:var(--ac);border-color:var(--ac);color:#fff}button:disabled{opacity:.5}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:12px 14px;margin:14px 0}
pre{margin:6px 0 0;white-space:pre-wrap;font:12px/1.45 ui-monospace,Consolas,monospace;color:var(--mut);max-height:280px;overflow:auto}
a{color:var(--ac)}ul{padding-left:18px;margin:6px 0}.ok{color:var(--ok)}.err{color:var(--err)}
</style></head><body><div class="wrap">
<h1>Transcrição Assistente</h1>
<div class="mut">Whisper large-v3 (local, GPU) transcreve · DeepSeek V4.1 Flash revisa o texto · JEV dá o contexto ·
o áudio decide. O áudio nunca sai do computador; só o texto da transcrição sai, para o DeepSeek (OpenCode Zen ou
OpenRouter, o primeiro que responder) e para o JEV.<br>
O idioma é detectado <b>por canal</b> — numa call cada lado pode falar o seu. Forçar o idioma errado faz o
Whisper <b>traduzir</b> em vez de transcrever, então deixe em "detectar automaticamente" a menos que precise.</div>
<div id="zona"><b>Arraste o áudio aqui</b><br><span class="mut">ou clique para escolher (ogg, mp3, wav, m4a, mp4…)</span>
<input id="arq" type="file" accept="audio/*,video/*" hidden></div>
<div class="op">
<label>Idioma <select id="idioma"><option value="auto" selected>Detectar automaticamente</option><option value="pt">Português</option><option value="en">Inglês</option></select></label>
<label><input type="checkbox" id="deepseek"> DeepSeek (envia a transcrição como texto)</label>
<label><input type="checkbox" id="jev"> JEV (envia trechos e candidatos como texto)</label>
<label><input type="checkbox" id="refazer"> refazer do zero</label>
<button class="pri" id="ir" disabled>Transcrever</button></div>
<div class="mut" id="escolhido">Nenhum arquivo escolhido.</div>
<div class="card" id="job" hidden><b id="titulo">Enviando…</b><pre id="log"></pre></div>
<div class="card"><b>Resultados anteriores</b><ul id="lista">__LISTA__</ul></div>
</div><script>
const z=document.getElementById('zona'),arq=document.getElementById('arq'),ir=document.getElementById('ir');let f=null;
function escolher(x){f=x;document.getElementById('escolhido').textContent=x?x.name+' · '+(x.size/1048576).toFixed(1)+' MB':'';ir.disabled=!x}
z.onclick=()=>arq.click();arq.onchange=()=>escolher(arq.files[0]);
z.ondragover=e=>{e.preventDefault();z.classList.add('sobre')};z.ondragleave=()=>z.classList.remove('sobre');
z.ondrop=e=>{e.preventDefault();z.classList.remove('sobre');if(e.dataTransfer.files[0])escolher(e.dataTransfer.files[0])};
ir.onclick=async()=>{if(!f)return;ir.disabled=true;const job=document.getElementById('job'),log=document.getElementById('log'),
tit=document.getElementById('titulo');job.hidden=false;tit.textContent='Enviando '+f.name+'…';log.textContent='';
const q=new URLSearchParams({nome:f.name,idioma:document.getElementById('idioma').value,
deepseek:document.getElementById('deepseek').checked?1:0,jev:document.getElementById('jev').checked?1:0,
refazer:document.getElementById('refazer').checked?1:0});
let r;try{r=await fetch('/enviar?'+q,{method:'POST',body:f});r=await r.json()}catch(e){tit.textContent='Falha no envio';tit.className='err';ir.disabled=false;return}
if(r.erro){tit.textContent=r.erro;if(r.url)tit.innerHTML+=' <a href="'+r.url+'" target="_blank">abrir revisão</a>';tit.className='err';ir.disabled=false;return}
const t=setInterval(async()=>{const s=await (await fetch('/status?id='+r.id)).json();log.textContent=s.log.join('\\n');log.scrollTop=log.scrollHeight;
if(s.feito){clearInterval(t);ir.disabled=false;if(s.url){tit.innerHTML='Pronto em '+s.segundos+' s · <a href="'+s.url+'" target="_blank">abrir revisão</a>';tit.className='ok';window.open(s.url,'_blank')}
else{tit.textContent='Erro: '+s.erro;tit.className='err'}}
else{tit.textContent=(s.decorrido?'Processando '+f.name+' · '+Math.floor(s.decorrido/60)+'min '+(s.decorrido%60)+'s':'Na fila: '+f.name)}},1000)};
</script></body></html>"""


class App(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        dados = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(dados)))
        self.end_headers()
        self.wfile.write(dados)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        if url.path == "/":
            lista = "".join(f'<li><a href="/saidas/{urllib.parse.quote(d)}/revisao.html" target="_blank">'
                            f'{html.escape(d)}</a></li>' for d in _resultados()) or '<li class="mut">nenhum ainda</li>'
            dados = PAGINA.replace("__LISTA__", lista).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(dados)))
            self.end_headers()
            self.wfile.write(dados)
        elif url.path == "/status":
            job = JOBS.get(urllib.parse.parse_qs(url.query).get("id", [""])[0])
            if job and not job.get("feito") and job.get("iniciado"):
                job = {**job, "decorrido": int(time.time() - job["iniciado"])}
            self._json(job or {"feito": True, "erro": "trabalho não encontrado", "log": []})
        elif url.path.startswith("/saidas/"):
            rel = urllib.parse.unquote(url.path[len("/saidas/"):])
            caminho = os.path.realpath(os.path.join(SAIDAS, rel))
            if not caminho.startswith(os.path.realpath(SAIDAS) + os.sep) or not os.path.isfile(caminho):
                self.send_error(404)
                return
            tipo = mimetypes.guess_type(caminho)[0] or "application/octet-stream"
            if tipo.startswith("text/") or tipo == "application/json":
                tipo += "; charset=utf-8"
            tam = os.path.getsize(caminho)
            self.send_response(200)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(tam))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with open(caminho, "rb") as f:
                while bloco := f.read(1 << 20):
                    self.wfile.write(bloco)
        else:
            self.send_error(404)

    def _salvar_locutores(self, url):
        """Grava os nomes/atribuições da revisão em saidas/<pasta>/locutores.json e o .txt com nomes."""
        q = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
        pasta = os.path.realpath(os.path.join(SAIDAS, os.path.basename(q.get("pasta", ""))))
        if not pasta.startswith(os.path.realpath(SAIDAS) + os.sep) or not os.path.isdir(pasta):
            self._json({"erro": "pasta desconhecida"}, 404)
            return
        tam = int(self.headers.get("Content-Length") or 0)
        if not tam or tam > (4 << 20):
            self._json({"erro": "corpo vazio ou grande demais"}, 400)
            return
        try:
            dados = json.loads(self.rfile.read(tam).decode("utf-8"))
        except ValueError:
            self._json({"erro": "json inválido"}, 400)
            return
        with open(os.path.join(pasta, "locutores.json"), "w", encoding="utf-8") as f:
            json.dump({"nomes": dados.get("nomes", {}), "atribuicao": dados.get("atribuicao", {}),
                       "confirmados": dados.get("confirmados", {})}, f, ensure_ascii=False, indent=1)
        if isinstance(dados.get("txt"), str) and dados["txt"].strip():
            with open(os.path.join(pasta, "transcricao.txt"), "w", encoding="utf-8") as f:
                f.write(dados["txt"])
        self._json({"ok": True})

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        if url.path == "/locutores":
            self._salvar_locutores(url)
            return
        if url.path != "/enviar":
            self.send_error(404)
            return
        q = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
        tam = int(self.headers.get("Content-Length") or 0)
        if not tam or tam > MAX_BYTES:
            self._json({"erro": "arquivo vazio ou maior que 1 GB"}, 400)
            return
        os.makedirs(ENTRADAS, exist_ok=True)
        nome = time.strftime("%Y%m%d-%H%M%S_") + _seguro(q.get("nome", "audio"))
        caminho = os.path.join(ENTRADAS, nome)
        restante = tam
        with open(caminho, "wb") as f:
            while restante:
                bloco = self.rfile.read(min(restante, 1 << 20))
                if not bloco:
                    break
                f.write(bloco)
                restante -= len(bloco)
        # Reenviar o mesmo áudio é permitido: reaproveita a pasta e o cache da transcrição,
        # então roda em segundos em vez de minutos. "refazer" ignora o cache.
        digest = _sha256(caminho)
        anterior = HASHES.get(digest) or {}
        refazer = q.get("refazer") == "1"
        pasta = anterior.get("pasta")   # mesma pasta do envio anterior, se já conhecemos este áudio
        job_id = uuid.uuid4().hex[:12]
        HASHES[digest] = {"feito": False, "pasta": pasta, "url": None}
        JOBS[job_id] = {"log": [f"recebido: {nome} ({tam / 1048576:.1f} MB)"], "feito": False,
                        "hash": digest, "nome": q.get("nome", "audio")}
        idioma = q.get("idioma") if q.get("idioma") in ("auto", "pt", "en") else "auto"
        threading.Thread(target=_rodar, daemon=True, args=(
            job_id, caminho, idioma, q.get("deepseek") == "1", q.get("jev") == "1",
            pasta, refazer)).start()
        self._json({"id": job_id})


if __name__ == "__main__":
    _lembrar_resultados()
    srv = ThreadingHTTPServer(("127.0.0.1", PORTA), App)
    print(f"Transcrição Assistente em http://127.0.0.1:{PORTA}  (Ctrl+C para fechar)")
    if "--sem-navegador" not in sys.argv:
        webbrowser.open(f"http://127.0.0.1:{PORTA}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
