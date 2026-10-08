"""Gera a página de revisão (HTML), o texto (.txt) e os dados brutos (.json)."""
import html
import json
import os
import subprocess


def mmss(t):
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:04.1f}"


def _menor(som):
    """Estalos, cliques e ruídos curtos: escondidos por padrão para não poluir a revisão."""
    return som["rotulo"].startswith(("estalo", "cliques", "ruído")) and         not som["rotulo"].startswith("ruído contínuo")


def rotulo_falante(canal, independentes):
    if canal == "M":
        return "Falante ?"
    return f"Canal {canal}" if independentes else f"Falante {canal}"


def _sobrepoe(f, sobreposicoes):
    return any(o["ini"] < f["fim"] and o["fim"] > f["ini"] for o in sobreposicoes)


def gerar_txt(audio, r, locutores=None):
    """Texto exportado. Havendo locutor, sai o nome dele em vez de "Canal E"."""
    locutores = locutores or {}

    def quem_de(x):
        n = x.get("locutor")
        if n:
            return locutores.get(str(n)) or locutores.get(n) or f"Locutor {n}"
        return rotulo_falante(x["canal"], audio.independentes)

    falas = r.get("blocos") or r["falas"]
    linhas = []
    eventos = [("fala", f["ini"], f) for f in falas] + \
              [("som", s["ini"], s) for s in r["sons"]
               if s["rotulo"].startswith(("fala não", "vocalização", "riso", "tosse", "ruído contínuo"))]
    for tipo, _, x in sorted(eventos, key=lambda e: e[1]):
        quem = quem_de(x)
        if tipo == "fala":
            texto = x.get("texto") or " ".join(p.texto for p in x["palavras"])
            ov = " [sobreposição]" if _sobrepoe(x, r["sobreposicoes"]) else ""
            linhas.append(f"[{mmss(x['ini'])}–{mmss(x['fim'])}] {quem}:{ov} {texto}")
        else:
            linhas.append(f"[{mmss(x['ini'])}–{mmss(x['fim'])}] {quem}: [{x['rotulo']}]")
    return "\n".join(linhas) + "\n"


def ler_locutores(pasta):
    """Nomes já salvos numa revisão anterior desta pasta (locutores.json)."""
    try:
        with open(os.path.join(pasta, "locutores.json"), encoding="utf-8") as f:
            return json.load(f).get("nomes", {})
    except (OSError, ValueError):
        return {}


def gerar_json(audio, r, meta, locutores=None):
    def pal(p):
        return {"texto": p.texto, "ini": round(p.ini, 2), "fim": round(p.fim, 2),
                "prob": round(p.prob, 3), "flags": getattr(p, "flags", [])}
    dados = {
        "meta": meta,
        "falas": [{"canal": f["canal"], "ini": round(f["ini"], 2), "fim": round(f["fim"], 2),
                   "texto": " ".join(p.texto for p in f["palavras"]),
                   "palavras": [pal(p) for p in f["palavras"]]} for f in r["falas"]],
        "sons": r["sons"], "sobreposicoes": r["sobreposicoes"], "suspeitas": r["suspeitas"],
    }
    if r.get("blocos"):
        dados["blocos"] = [{"canal": b["canal"], "ini": round(b["ini"], 2), "fim": round(b["fim"], 2),
                            "texto": b["texto"], "locutor": b.get("locutor"),
                            "confianca": b.get("confianca", 0.0), "motivo": b.get("motivo", ""),
                            "pedacos": b.get("pedacos", 1)} for b in r["blocos"]]
    if locutores:
        dados["locutores"] = locutores
    return dados


CSS = """
:root{--bg:#f7f7f5;--card:#fff;--tx:#1d1d1b;--mut:#6b6b66;--bd:#e3e3de;--ac:#2f5bd3;
--warn:#fff1c2;--bad:#ffd9d6;--ov:#e5ecff;--snd:#e9f6ec}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#151514;--card:#1e1e1c;
--tx:#ecece8;--mut:#9a9a93;--bd:#33332f;--ac:#7d9bff;--warn:#4a3f17;--bad:#552622;--ov:#22305a;--snd:#1d3a25}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);
font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:16px}
header{position:sticky;top:0;z-index:5;background:var(--bg);padding:10px 0;border-bottom:1px solid var(--bd)}
h1{font-size:18px;margin:0 0 6px}h2{font-size:15px;margin:22px 0 8px}
.meta{color:var(--mut);font-size:13px}.ctl{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-top:8px}
button{font:inherit;font-size:13px;border:1px solid var(--bd);background:var(--card);color:var(--tx);
border-radius:6px;padding:3px 9px;cursor:pointer}button.on{background:var(--ac);color:#fff;border-color:var(--ac)}
audio{width:100%;max-width:520px;height:34px}
.grid{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:16px}
@media (max-width:860px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:10px 12px}
.fala{padding:6px 0;border-bottom:1px solid var(--bd)}.fala:last-child{border:0}
.fala.ov{background:linear-gradient(90deg,var(--ov),transparent 60%)}
.quem{font-weight:600;font-size:13px}.t{color:var(--mut);font-size:12px;font-variant-numeric:tabular-nums}
.w{cursor:pointer;border-radius:3px;padding:0 1px}.w:hover{outline:1px solid var(--ac)}
.w.f{background:var(--warn)}.w.x{background:var(--bad)}.w.now{outline:2px solid var(--ac)}
.som{font-size:13px;background:var(--snd);border-radius:4px;padding:2px 6px;display:inline-block;margin:4px 0}
table{width:100%;border-collapse:collapse;font-size:13px}td{padding:4px 4px;border-bottom:1px solid var(--bd);vertical-align:top}
.p1{color:#c0392b;font-weight:600}.p2{color:#b7791f;font-weight:600}.p3,.p4{color:var(--mut)}
.dd{cursor:pointer;font-size:11px;color:var(--ac);padding:0 2px;font-weight:700}
.dd.ok{color:#2e8b57}.dd.no{color:#c0392b}.lac{background:var(--bad);font-weight:600}.w.ed{background:var(--snd);text-decoration:underline dotted}
#pop{position:absolute;z-index:20;background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:6px;
box-shadow:0 6px 24px rgba(0,0,0,.18);min-width:220px;max-width:min(360px,92vw)}#pop .vd{font-size:12px;color:var(--mut);margin:2px 4px 6px}
#pop .cand{display:block;width:100%;text-align:left;margin:2px 0}#pop small{color:var(--mut)}
.menor{display:none}body.todos .menor{display:block}body.todos tr.menor{display:table-row}
.leg span{display:inline-block;margin-right:10px;font-size:12px}.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
.loc{font-weight:600;font-size:12px;border-radius:10px;padding:1px 8px;border:1px solid transparent;cursor:pointer}
.loc.l1{background:#dbe7ff;color:#1c3f9e}.loc.l2{background:#ffe3d0;color:#9a4a12}
.loc.zero{background:transparent;border:1px dashed var(--bd);color:var(--mut)}
.loc.pal{border-style:dashed;border-color:currentColor}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]) .loc.l1{background:#22305a;color:#bcd0ff}
:root:not([data-theme=light]) .loc.l2{background:#4a2f1c;color:#ffd0a8}}
.lb{font-size:11px;padding:0 6px;margin-left:2px}.lb.on{background:var(--ac);color:#fff;border-color:var(--ac)}
#linha{position:relative;height:24px;background:var(--card);border:1px solid var(--bd);border-radius:6px;margin:8px 0;cursor:pointer;overflow:hidden}
#linha i{position:absolute;top:0;bottom:0;display:block}
#linha i.l1{background:#7d9bff}#linha i.l2{background:#f0a35e}#linha i.zero{background:#c9c9c2}
#linha i.ov{background:repeating-linear-gradient(45deg,#c0392b,#c0392b 3px,transparent 3px,transparent 6px);opacity:.5}
#quemquem{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:6px 0;font-size:13px}
#quemquem input{font:inherit;font-size:13px;border:1px solid var(--bd);background:var(--card);color:var(--tx);
border-radius:6px;padding:3px 8px;width:150px}
.salvo{color:#2e8b57;font-size:12px}.salvo.erro{color:#c0392b}
"""

JS = """
const a=document.getElementById('au');let fimTrecho=null;
// Servidores sem suporte a Range deixam o áudio sem busca; carregar como blob resolve.
// Aberto como arquivo local (file://) o fetch falha e o src direto já funciona.
fetch(a.getAttribute('src')).then(r=>r.blob()).then(b=>{a.src=URL.createObjectURL(b)}).catch(()=>{});
function tocar(i,f){a.currentTime=Math.max(0,i);fimTrecho=f==null?null:f;a.play()}
a.addEventListener('timeupdate',()=>{if(fimTrecho!=null&&a.currentTime>=fimTrecho){a.pause();fimTrecho=null}
const t=a.currentTime;document.querySelectorAll('.w.now').forEach(e=>e.classList.remove('now'));
for(const e of document.querySelectorAll('.w')){if(+e.dataset.i<=t&&t<=+e.dataset.f){e.classList.add('now');break}}});
document.querySelectorAll('[data-vel]').forEach(b=>b.onclick=()=>{a.playbackRate=+b.dataset.vel;
document.querySelectorAll('[data-vel]').forEach(x=>x.classList.toggle('on',x===b))});
let ctx,gE,gD;function rota(m){try{if(!ctx){ctx=new AudioContext();const s=ctx.createMediaElementSource(a);
const sp=ctx.createChannelSplitter(2),mg=ctx.createChannelMerger(2);gE=ctx.createGain();gD=ctx.createGain();
s.connect(sp);sp.connect(gE,0);sp.connect(gD,1);for(const g of[gE,gD]){g.connect(mg,0,0);g.connect(mg,0,1)}mg.connect(ctx.destination)}
gE.gain.value=m!=='D'?1:0;gD.gain.value=m!=='E'?1:0;
document.querySelectorAll('[data-can]').forEach(x=>x.classList.toggle('on',x.dataset.can===m))}catch(e){}}
document.querySelectorAll('[data-can]').forEach(b=>b.onclick=()=>rota(b.dataset.can));
const bm=document.getElementById('menores');bm.onclick=()=>{bm.classList.toggle('on',document.body.classList.toggle('todos'))};
const ALVOS=JSON.parse(document.getElementById('alvos').textContent),pop=document.getElementById('pop');
const esc=t=>String(t).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'})[c]);
function abrir(b){const id=b.dataset.alvo,x=ALVOS[id],w=document.querySelector('.w[data-alvo="'+id+'"]');
let h='<div class="vd">'+esc(x.veredito||'')+' · ▶ tocando o trecho'+(x.motivo_deepseek?'<br>DeepSeek: '+esc(x.motivo_deepseek):'')+'</div>';
for(const c of x.candidatos){const ctx=c.contexto!=null?' · contexto '+Math.round(c.contexto*100)+'%':'';
h+='<button class="cand" data-t="'+esc(c.texto)+'">'+esc(c.texto)+' <small>ouvido '+c.votos+'/'+(x.variantes||'?')+ctx+
(c.fontes.includes('parônimo')?' · parônimo':'')+(c.fontes.includes('DeepSeek')?' · DeepSeek':'')+(c.acustica==='favorece'?' · ✓ áudio favorece':c.acustica==='contra'?' · ✗ áudio contra':'')+'</small></button>'}
h+='<button class="cand" data-livre="1">escrever outra…</button>';pop.innerHTML=h;
const r=b.getBoundingClientRect();pop.style.left=Math.max(4,Math.min(window.scrollX+r.left,window.scrollX+innerWidth-370))+'px';
pop.style.top=(window.scrollY+r.bottom+4)+'px';pop.hidden=false;
pop.querySelectorAll('.cand').forEach(c=>c.onclick=ev=>{ev.stopPropagation();let t=c.dataset.t;
if(c.dataset.livre){t=prompt('Palavra:',w.textContent);if(t==null)return}
const pont=(w.dataset.orig||'').match(/[.,!?;:…]+$/);t=(t==='(nada)')?'':t+(t&&pont?pont[0]:'');
w.textContent=t;w.classList.remove('lac','f');w.classList.add('ed');pop.hidden=true});
tocar(x.ini-1.0,x.fim+0.6)}
document.querySelectorAll('.dd').forEach(b=>b.onclick=ev=>{ev.stopPropagation();abrir(b)});
pop.onclick=ev=>ev.stopPropagation();document.addEventListener('click',()=>{pop.hidden=true});
document.getElementById('baixar').onclick=()=>{
const u=URL.createObjectURL(new Blob([texto()],{type:'text/plain'}));const l=document.createElement('a');
l.href=u;l.download='transcricao_revisada.txt';l.click()};
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;
if(e.code==='Space'){e.preventDefault();a.paused?a.play():a.pause()}
if(e.code==='ArrowLeft'){a.currentTime-=2}if(e.code==='ArrowRight'){a.currentTime+=2}});

// ---------- quem falou ----------
const DADOS=JSON.parse(document.getElementById('dados-loc').textContent);
const NOMES={1:'',2:''},LOC={},CONF={};
DADOS.blocos.forEach((b,i)=>{LOC[i]=b.loc||0;CONF[i]=false});
Object.assign(NOMES,DADOS.nomes||{});
const CHAVE='loc:'+DADOS.pasta;
try{const g=JSON.parse(localStorage.getItem(CHAVE)||'null');
    if(g&&g.atribuicao){Object.keys(g.atribuicao).forEach(i=>{if(LOC[i]!==undefined)LOC[i]=g.atribuicao[i]});
      Object.assign(NOMES,g.nomes||{});Object.keys(g.confirmados||{}).forEach(i=>{CONF[i]=g.confirmados[i]})}}catch(e){}
const nome=n=>NOMES[n]||('Locutor '+n);
function pinta(){
  document.querySelectorAll('.fala[data-bloco]').forEach(f=>{
    const i=f.dataset.bloco,n=LOC[i],b=DADOS.blocos[i],palpite=b.palpite&&!CONF[i],chip=f.querySelector('.loc');
    chip.className='loc '+(n?('l'+n):'zero')+(palpite?' pal':'');
    chip.textContent=n?nome(n):'quem?';
    chip.title=n?(palpite?('palpite: '+b.motivo+' — clique para confirmar'):'confirmado'):'sem palpite: escolha 1 ou 2';
    f.dataset.quem=n?nome(n):'?';
    f.querySelectorAll('.lb').forEach(x=>x.classList.toggle('on',+x.dataset.n===n));
  });
  document.querySelectorAll('#linha i').forEach(s=>{
    const n=LOC[s.dataset.bloco];s.className=(n?('l'+n):'zero')+(s.dataset.ov?' ov':'')});
}
function texto(){
  const L=[];
  document.querySelectorAll('[data-quem]').forEach(f=>{
    const ws=[...f.querySelectorAll('.w')].map(w=>w.textContent.trim()).filter(t=>t&&t!=='⟨?⟩');
    if(ws.length)L.push('['+f.dataset.ini+'–'+f.dataset.fim+'] '+f.dataset.quem+':'+(f.classList.contains('ov')?' [sobreposição]':'')+' '+ws.join(' '));
    else if(f.dataset.som)L.push('['+f.dataset.ini+'–'+f.dataset.fim+'] '+f.dataset.quem+': ['+f.dataset.som+']')});
  return L.join('\\n')+'\\n';
}
function aviso(t,erro){const s=document.getElementById('salvo');s.textContent=t;s.className='salvo'+(erro?' erro':'')}
function salvar(){
  const dados={nomes:{1:NOMES[1],2:NOMES[2]},atribuicao:LOC,confirmados:CONF};
  try{localStorage.setItem(CHAVE,JSON.stringify(dados))}catch(e){}
  if(!location.protocol.startsWith('http')){aviso('salvo neste navegador (abra pelo app para gravar na pasta)');return}
  fetch('/locutores?pasta='+encodeURIComponent(DADOS.pasta),{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(Object.assign({txt:texto()},dados))})
    .then(r=>r.ok?aviso('salvo em '+DADOS.pasta):aviso('não consegui salvar na pasta',1))
    .catch(()=>aviso('não consegui salvar na pasta',1));
}
document.querySelectorAll('.lb').forEach(b=>b.onclick=ev=>{ev.stopPropagation();
  const i=b.dataset.lb,n=+b.dataset.n;LOC[i]=(LOC[i]===n?0:n);CONF[i]=LOC[i]!==0;pinta();salvar()});
document.querySelectorAll('.loc').forEach(c=>c.onclick=ev=>{ev.stopPropagation();
  const i=c.closest('.fala').dataset.bloco;if(LOC[i]){CONF[i]=true;pinta();salvar()}});
['1','2'].forEach(n=>{const i=document.getElementById('nome'+n);
  i.value=NOMES[n]||'';i.oninput=()=>{NOMES[n]=i.value.trim();pinta()};i.onchange=salvar});
document.getElementById('aceitar').onclick=()=>{DADOS.blocos.forEach((b,i)=>{if(b.palpite){LOC[i]=b.loc;CONF[i]=true}});pinta();salvar()};
document.getElementById('limpar').onclick=()=>{DADOS.blocos.forEach((b,i)=>{LOC[i]=b.palpite?b.loc:0;CONF[i]=false});pinta();salvar()};
document.getElementById('linha').onclick=ev=>{const r=ev.currentTarget.getBoundingClientRect();
  tocar((ev.clientX-r.left)/r.width*a.duration-0.2)};
pinta();
"""


def gerar_html(audio, r, meta, nome_audio, pasta="", locutores=None):
    e = html.escape
    ind = audio.independentes
    locutores = locutores or {}
    alvos = r.get("alvos", [])
    # Blocos de fala com locutor (ta/turnos.py). Sem eles, cada fala é um bloco.
    blocos = r.get("blocos") or [
        {"canal": f["canal"], "ini": f["ini"], "fim": f["fim"], "palavras": f["palavras"],
         "texto": " ".join(p.texto for p in f["palavras"]), "locutor": None,
         "confianca": 0.0, "motivo": "", "pedacos": 1} for f in r["falas"]]
    # O bloco reusa as MESMAS palavras da fala, então o alvo é achado por identidade.
    por_pos = {(x["fala"], x["pos"]): x for x in alvos if x["tipo"] == "palavra"}
    por_palavra = {}
    for fi, f in enumerate(r["falas"]):
        for wi, p in enumerate(f["palavras"]):
            if (fi, wi) in por_pos:
                por_palavra[id(p)] = por_pos[(fi, wi)]
    lacunas, soltas = {}, {}
    for x in alvos:
        if x["tipo"] != "lacuna":
            continue
        bi = next((i for i, b in enumerate(blocos) if b["canal"] == x["canal"]
                   and b["ini"] - 0.6 < x["fim"] and x["ini"] < b["fim"] + 0.6), None)
        if bi is None:
            soltas[(x["canal"], round(x["ini"], 2))] = x
        else:
            lacunas.setdefault(bi, []).append(x)

    def nome_de(n):
        return locutores.get(str(n)) or locutores.get(n) or f"Locutor {n}"

    def chip(bi, b):
        n = b.get("locutor")
        texto = nome_de(n) if n else "quem?"
        return (f'<span class="loc {"l" + str(n) if n else "zero"}">{e(texto)}</span>'
                f'<button class="lb" data-lb="{bi}" data-n="1" title="esta fala é do locutor 1">1</button>'
                f'<button class="lb" data-lb="{bi}" data-n="2" title="esta fala é do locutor 2">2</button>')

    def seletor(x):
        v = x.get("veredito", "")
        cor = " ok" if ("concordam" in v or v.startswith("áudio confirma")) else (" no" if "ouvir" in v else "")
        return f'<b class="dd{cor}" data-alvo="{x["id"]}" title="{e(v)}">▾</b>'

    def lacuna(x):
        return (f'<span class="w lac" data-alvo="{x["id"]}" data-i="{x["ini"]:.2f}" data-f="{x["fim"]:.2f}" '
                f'title="voz sem palavra: escolha um candidato" '
                f'onclick="tocar({x["ini"]-0.6:.2f},{x["fim"]+0.4:.2f})">⟨?⟩</span>{seletor(x)}')

    falas_html = []
    eventos = [("fala", b["ini"], (i, b)) for i, b in enumerate(blocos)] + \
              [("som", s["ini"], s) for s in r["sons"]]
    for tipo, _, x in sorted(eventos, key=lambda z: z[1]):
        if tipo == "som":
            quem = e(rotulo_falante(x["canal"], ind))
            solta = soltas.get((x["canal"], round(x["ini"], 2)))
            menor = " menor" if _menor(x) and not solta else ""
            # Vira linha no texto exportado: lacuna solta (se preenchida) ou som relevante.
            # "Voz sem palavra" não entra como som: ela já aparece como ⟨?⟩ dentro da fala.
            exporta_som = not solta and not _menor(x) and                 not x["rotulo"].startswith(("voz sem palavra", "fala não"))
            attrs = (f' data-quem="{quem}" data-ini="{mmss(x["ini"])}" data-fim="{mmss(x["fim"])}"'
                     if solta or exporta_som else "")
            if exporta_som:
                attrs += f' data-som="{e(x["rotulo"])}"'
            extra = f" {lacuna(solta)}" if solta else ""
            falas_html.append(
                f'<div class="fala{menor}"{attrs}><span class="t">{mmss(x["ini"])}</span> '
                f'<span class="som" onclick="tocar({x["ini"]-0.4:.2f},{x["fim"]+0.4:.2f})" '
                f'style="cursor:pointer">▶ {quem}: {e(x["rotulo"])} · voz {x["voz"]:.0%}</span>{extra}</div>')
            continue
        bi, x = x
        n = x.get("locutor")
        quem = e(nome_de(n)) if n else e(rotulo_falante(x["canal"], ind))
        itens = [(p.ini, "p", wi, p) for wi, p in enumerate(x["palavras"])] + \
                [(lc["ini"], "l", -1, lc) for lc in lacunas.get(bi, [])]
        ws = []
        for _, k, wi, p in sorted(itens, key=lambda z: z[0]):
            if k == "l":
                ws.append(lacuna(p))
                continue
            fl = getattr(p, "flags", [])
            cls = "w x" if any("aluc" in f or "sem voz" in f for f in fl) else ("w f" if fl else "w")
            tip = e(f'{p.prob:.0%}' + (" · " + "; ".join(fl) if fl else ""))
            alvo = por_palavra.get(id(p))
            da = f' data-alvo="{alvo["id"]}" data-orig="{e(p.texto)}"' if alvo else ""
            ws.append(f'<span class="{cls}"{da} data-i="{p.ini:.2f}" data-f="{p.fim:.2f}" title="{tip}" '
                      f'onclick="tocar({p.ini-0.3:.2f})">{e(p.texto)}</span>' + (seletor(alvo) if alvo else ""))
        ov = " ov" if _sobrepoe(x, r["sobreposicoes"]) else ""
        falas_html.append(
            f'<div class="fala{ov}" data-bloco="{bi}" data-quem="{quem}" '
            f'data-ini="{mmss(x["ini"])}" data-fim="{mmss(x["fim"])}">'
            f'{chip(bi, x)} '
            f'<span class="t" style="cursor:pointer" onclick="tocar({x["ini"]-0.3:.2f},{x["fim"]+0.3:.2f})">'
            f'▶ {mmss(x["ini"])}–{mmss(x["fim"])}</span>{" · sobreposição" if ov else ""}<br>{" ".join(ws)}</div>')

    sus = []
    for s in r["suspeitas"]:
        sus.append(f'<tr{" class=menor" if s["prio"] >= 4 else ""}><td><button onclick="tocar({s["ini"]-0.6:.2f},{s["fim"]+0.6:.2f})">▶ {mmss(s["ini"])}</button></td>'
                   f'<td class="p{s["prio"]}">{e(s["tipo"])}</td><td>{e(s["canal"])}</td><td>{e(s["texto"])}</td></tr>')

    canais_btn = ""
    if ind:
        canais_btn = ('<span class="meta">ouvir:</span><button data-can="ambos" class="on">ambos</button>'
                      '<button data-can="E">só E</button><button data-can="D">só D</button>')
    n_sob = len(r["sobreposicoes"])
    campos = ("ini", "fim", "candidatos", "variantes", "veredito", "jev", "motivo_deepseek")
    alvos_json = json.dumps({x["id"]: {k: x.get(k) for k in campos} for x in alvos},
                            ensure_ascii=False).replace("</", "<\\/")
    n_menor = sum(1 for x in r["sons"] if _menor(x))

    # Linha do tempo: uma faixa por bloco, colorida pelo locutor. O palpite já pinta;
    # a página repinta conforme você confirma.
    def pct(t):
        return max(0.0, min(100.0, 100.0 * t / max(audio.duracao, 0.01)))

    linha = "".join(
        f'<i data-bloco="{i}" class="{(("l" + str(b["locutor"])) if b["locutor"] else "zero")}'
        f'{" ov" if _sobrepoe(b, r["sobreposicoes"]) else ""}" '
        f'style="left:{pct(b["ini"]):.3f}%;width:{max(0.2, pct(b["fim"]) - pct(b["ini"])):.3f}%" '
        f'title="{e(b["texto"][:70])}"></i>' for i, b in enumerate(blocos))

    dados_loc = json.dumps({
        "pasta": os.path.basename(pasta.rstrip("\\/")) if pasta else "",
        "nomes": {str(k): v for k, v in locutores.items()},
        "blocos": [{"loc": b.get("locutor") or 0, "palpite": bool(b.get("locutor")),
                    "motivo": b.get("motivo", ""), "ini": round(b["ini"], 2),
                    "fim": round(b["fim"], 2), "canal": b["canal"]} for b in blocos],
    }, ensure_ascii=False).replace("</", "<\\/")

    painel = (
        '<div id="quemquem"><span class="meta">quem é quem:</span>'
        '<span class="loc l1">1</span><input id="nome1" placeholder="nome do locutor 1">'
        '<span class="loc l2">2</span><input id="nome2" placeholder="nome do locutor 2">'
        '<button id="aceitar" title="marca todos os palpites como confirmados">aceitar os palpites</button>'
        '<button id="limpar" title="volta aos palpites originais">desfazer</button>'
        '<span class="salvo" id="salvo"></span></div>')
    estrutura = (f"estéreo, 1 microfone por canal (correlação {audio.correlacao:.2f})" if ind
                 else f"mono (correlação entre canais {audio.correlacao:.2f})")
    aviso = ("Estéreo, 1 microfone por canal. Onde as duas pessoas caem no mesmo canal, a ferramenta separa "
             "por turnos e tom de voz — confira os palpites (chip tracejado)."
             if ind else "Áudio mono: os locutores são separados por turnos e tom de voz, como palpite. "
                         "Confirme ou corrija no chip de cada fala.")
    # Qual rota de DeepSeek realmente respondeu (Zen, OpenRouter ou NVIDIA) e o que falhou antes.
    ds = meta.get("deepseek") or {}
    if ds.get("usado"):
        rota_deepseek = (f"revisor: DeepSeek {e(ds.get('modelo', ''))} via {e(ds.get('rota', ''))} "
                         f"(esforço {e(ds.get('esforco', ''))}, {len(alvos)} dúvidas enviadas)")
        falhas = [f"{e(t['rota'])}: {e(t['motivo'])}" for t in ds.get("tentativas", [])]
        if falhas:
            rota_deepseek += " · rotas descartadas antes: " + "; ".join(falhas)
    else:
        rota_deepseek = f"revisor: DeepSeek não usado ({e(ds.get('motivo', 'desligado'))})"
    return f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Revisão de transcrição</title>
<style>{CSS}</style></head><body><div class="wrap">
<header><h1>{e(os.path.basename(audio.caminho))}</h1>
<div class="meta">{mmss(audio.duracao)} · {estrutura} · {len(r["falas"])} falas do Whisper em {len(blocos)} blocos ·
{n_sob} sobreposições · {len(r["sons"])} sons sem palavra · {len(r["suspeitas"])} pontos para ouvir ·
modelo {e(meta["modelo"])} ({e(meta["dispositivo"])}) · {meta["segundos"]:.0f}s de processamento</div>
<div class="meta">{rota_deepseek}</div>
<div class="meta">{e(aviso)}</div>
<div class="ctl"><audio id="au" controls preload="auto" src="{e(nome_audio)}"></audio>
<span class="meta">velocidade:</span><button data-vel="0.5">0,5×</button><button data-vel="0.75">0,75×</button>
<button data-vel="1" class="on">1×</button>{canais_btn}<button id="menores">mostrar estalos e ruídos ({n_menor})</button>
<button id="baixar">baixar texto revisado</button>
<span class="meta">espaço = tocar/pausar · ←/→ = 2 s · ▾ = candidatos</span></div>
{painel}
<div id="linha" title="linha do tempo: quem falou quando. Clique para tocar dali.">{linha}</div>
<div class="leg meta"><span><i class="sw" style="background:var(--warn)"></i>conferir</span>
<span><i class="sw" style="background:var(--bad)"></i>provável alucinação</span>
<span><i class="sw" style="background:var(--ov)"></i>sobreposição</span>
<span><i class="sw" style="background:var(--snd)"></i>som sem palavra</span>
<span><i class="sw" style="background:#7d9bff"></i>locutor 1</span>
<span><i class="sw" style="background:#f0a35e"></i>locutor 2</span>
<span><i class="sw" style="background:#c9c9c2"></i>sem palpite</span></div></header>
<div class="grid"><section><h2>Transcrição (clique numa palavra para tocar)</h2><div class="card">{"".join(falas_html)}</div></section>
<section><h2>Onde ouvir, por prioridade</h2><div class="card"><table>{"".join(sus) or "<tr><td>nada suspeito</td></tr>"}</table></div></section></div>
</div><div id="pop" hidden></div>
<script type="application/json" id="alvos">{alvos_json}</script>
<script type="application/json" id="dados-loc">{dados_loc}</script><script>{JS}</script></body></html>"""


def salvar(audio, r, meta, pasta, locutores=None):
    os.makedirs(pasta, exist_ok=True)
    if locutores is None:
        locutores = ler_locutores(pasta)
    # O Chrome erra a duração de alguns .ogg (leu 64 s num arquivo de 104 s) e aí
    # o clique na palavra pula para o lugar errado. AAC em .m4a busca com precisão.
    nome_audio = "audio.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", audio.caminho, "-c:a", "aac",
                    "-b:a", "160k", "-movflags", "+faststart",
                    os.path.join(pasta, nome_audio)], check=True)
    with open(os.path.join(pasta, "revisao.html"), "w", encoding="utf-8") as f:
        f.write(gerar_html(audio, r, meta, nome_audio, pasta, locutores))
    with open(os.path.join(pasta, "transcricao.txt"), "w", encoding="utf-8") as f:
        f.write(gerar_txt(audio, r, locutores))
    with open(os.path.join(pasta, "dados.json"), "w", encoding="utf-8") as f:
        json.dump(gerar_json(audio, r, meta, locutores), f, ensure_ascii=False, indent=1, default=float)
    return os.path.join(pasta, "revisao.html")
