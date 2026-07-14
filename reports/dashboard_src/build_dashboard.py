#!/usr/bin/env python3
import json, csv
ROOT="/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet"
dd=json.load(open(f"{ROOT}/reports/downstream_data.json"))
curves=json.load(open(f"{ROOT}/reports/training_bpb_curves.json"))
cfg=json.load(open(f"{ROOT}/reports/model_config.json"))
bpb=list(csv.DictReader(open(f"{ROOT}/reports/scaling_bpb.csv")))
robust=json.load(open(f"{ROOT}/reports/robustness.json"))
despace=json.load(open(f"{ROOT}/reports/despace.json"))
ablation=json.load(open(f"{ROOT}/reports/parsing_ablation_100M.json"))

D=dd["data"]
# 1.3B avg-all spread
av=[D[s]["1.3B"]["_avgall"] for s in D if D[s].get("1.3B")]
tiles=[
 ("Model scales","100M → 1.3B","4 points · ratio ~210 B/param"),
 ("Families","5","Llama · AU-Net · BPEByte-rg · Hybrid leafQ/greedyQ"),
 ("1.3B Avg-all","%.1f–%.1f"%(min(av),max(av)),"4-way tie (Hybrid leafQ trails)"),
 ("Best BPB @1.3B","0.827","Hybrid (leaf/B3) · vs Llama 0.840"),
]
payload={"downstream":dd,"curves":curves,"config":cfg,"bpb":bpb,"robust":robust,"despace":despace,"ablation":ablation}
J=json.dumps(payload,separators=(",",":"))

HTML=r'''<title>Tokenization Scaling — AU-Net / BPEByte / Hybrid</title>
<style>
:root{
 --ground:#f6f8fb; --surface:#ffffff; --surface2:#eef2f8; --ink:#152232; --ink2:#4a596c;
 --muted:#6b7a8d; --border:#dbe3ee; --hair:#e8edf4; --accent:#3b6fe0; --accent-soft:#e7eefc;
 --good:#12805c; --warn:#b45309;
 --s-llama:#0072B2; --s-aunet:#E69F00; --s-rg:#009E73; --s-leaf:#D55E00; --s-greedy:#CC79A7;
 --font-sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial;
 --font-mono:ui-monospace,"SF Mono","Cascadia Code","Roboto Mono",Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){:root{
 --ground:#0e141c; --surface:#161e29; --surface2:#1d2733; --ink:#e6edf6; --ink2:#aab7c7;
 --muted:#8494a6; --border:#273341; --hair:#202b38; --accent:#5b8bff; --accent-soft:#1a2740;
 --good:#3fbf92; --warn:#e0a24a;}}
:root[data-theme="light"]{
 --ground:#f6f8fb; --surface:#ffffff; --surface2:#eef2f8; --ink:#152232; --ink2:#4a596c;
 --muted:#6b7a8d; --border:#dbe3ee; --hair:#e8edf4; --accent:#3b6fe0; --accent-soft:#e7eefc; --good:#12805c; --warn:#b45309;}
:root[data-theme="dark"]{
 --ground:#0e141c; --surface:#161e29; --surface2:#1d2733; --ink:#e6edf6; --ink2:#aab7c7;
 --muted:#8494a6; --border:#273341; --hair:#202b38; --accent:#5b8bff; --accent-soft:#1a2740; --good:#3fbf92; --warn:#e0a24a;}
*{box-sizing:border-box}
body{margin:0}
.wrap{background:var(--ground);color:var(--ink);font-family:var(--font-sans);
 line-height:1.5;-webkit-font-smoothing:antialiased;min-height:100vh}
.mono{font-family:var(--font-mono);font-variant-numeric:tabular-nums}
.page{max-width:1180px;margin:0 auto;padding:0 24px 80px}
header.top{padding:40px 0 22px;border-bottom:1px solid var(--border)}
.kick{font-family:var(--font-mono);font-size:11.5px;letter-spacing:.16em;text-transform:uppercase;color:var(--accent);margin:0 0 10px}
h1{font-size:clamp(26px,3.4vw,38px);line-height:1.08;margin:0;letter-spacing:-.02em;text-wrap:balance;font-weight:680}
.sub{color:var(--ink2);margin:12px 0 0;max-width:70ch;font-size:15px}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:26px 0 4px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px 16px 14px}
.tile .t-lab{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
.tile .t-val{font-family:var(--font-mono);font-size:25px;font-weight:600;margin:7px 0 3px;letter-spacing:-.01em}
.tile .t-note{font-size:12px;color:var(--ink2)}
nav.sticky{position:sticky;top:0;z-index:20;background:color-mix(in srgb,var(--ground) 88%,transparent);
 backdrop-filter:blur(8px);border-bottom:1px solid var(--border);margin-top:26px}
nav.sticky ul{list-style:none;display:flex;gap:4px;margin:0 auto;padding:8px 24px;max-width:1180px;flex-wrap:wrap}
nav.sticky a{font-family:var(--font-mono);font-size:12.5px;color:var(--ink2);text-decoration:none;padding:7px 13px;border-radius:8px;letter-spacing:.02em}
nav.sticky a:hover{background:var(--surface2);color:var(--ink)}
nav.sticky a.on{background:var(--accent-soft);color:var(--accent)}
section{padding:38px 0 8px;scroll-margin-top:52px}
h2{font-size:20px;margin:0 0 4px;letter-spacing:-.01em;font-weight:640}
.lede{color:var(--ink2);font-size:14px;margin:0 0 18px;max-width:74ch}
.card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:18px}
.chartbar{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px}
.chip{font-family:var(--font-mono);font-size:12px;border:1px solid var(--border);background:var(--surface);
 color:var(--ink2);padding:5px 11px;border-radius:999px;cursor:pointer}
.chip:hover{border-color:var(--accent)}
.chip.on{background:var(--accent);border-color:var(--accent);color:#fff}
.chart{width:100%;overflow-x:auto}
svg .grid{stroke:var(--hair)} svg .axis{stroke:var(--border)}
svg text{fill:var(--muted);font-family:var(--font-mono);font-size:11px}
svg text.ttl{fill:var(--ink);font-family:var(--font-sans);font-weight:640;font-size:13px}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin-top:10px;font-size:12.5px;color:var(--ink2)}
.legend span{display:inline-flex;align-items:center;gap:7px}
.legend i{width:16px;height:3px;border-radius:2px;display:inline-block}
.legend i.dash{background:repeating-linear-gradient(90deg,currentColor 0 5px,transparent 5px 9px)!important}
.tablewrap{overflow-x:auto;border:1px solid var(--border);border-radius:14px;margin-top:6px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:9px 12px;text-align:right;white-space:nowrap}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
thead th{font-family:var(--font-mono);font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);
 border-bottom:1px solid var(--border);background:var(--surface2);position:sticky;top:0}
td{font-family:var(--font-mono);font-variant-numeric:tabular-nums;border-bottom:1px solid var(--hair);color:var(--ink2)}
td.fam{font-family:var(--font-sans);color:var(--ink)}
tr.scale-lead td{border-top:2px solid var(--border)}
td.avg{color:var(--ink);font-weight:600;background:var(--surface2)}
.dot{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:7px;vertical-align:middle}
.note{font-size:12.5px;color:var(--muted);margin-top:12px;max-width:80ch}
.foot{margin-top:40px;padding-top:18px;border-top:1px solid var(--border);color:var(--muted);font-size:12px}
@media (max-width:760px){.tiles{grid-template-columns:repeat(2,1fr)}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>

<div class="wrap"><div class="page">
<header class="top">
 <p class="kick">Scaling Report · 0-shot · ratio-matched ladder</p>
 <h1>Tokenization scaling: AU-Net, BPEByte-rg &amp; Hybrid</h1>
 <p class="sub">Downstream accuracy, bits-per-byte, and training curves for five tokenization schemes across four model scales (100M→1.3B). Byte families and Llama are token-budget matched; the hybrid is evaluated in two question regimes (leafQ / greedyQ).</p>
 <div class="tiles" id="tiles"></div>
</header>
<nav class="sticky"><ul>
 <li><a href="#downstream" class="on">Downstream</a></li>
 <li><a href="#ablation">100M ablation</a></li>
 <li><a href="#robust">Robustness</a></li>
 <li><a href="#despace">Despace</a></li>
 <li><a href="#bpb">Bits-per-byte</a></li>
 <li><a href="#training">Training curves</a></li>
 <li><a href="#config">Configuration</a></li>
</ul></nav>

<section id="downstream">
 <h2>Downstream accuracy</h2>
 <p class="lede">Pick a benchmark to see how each family scales. Solid = base families; dashed = hybrid regimes. Hybrid has no 760M run and no ARC-C below 1.3B. <b style="color:var(--warn)">100M &amp; 300M use the AU-Net-law recipe; the 300M base families (Llama/AU-Net/rg) are PARTIAL — law @~21 B tokens, still training — so they read low vs the 63 B hybrid.</b></p>
 <div class="card">
  <div class="chartbar" id="dbar"></div>
  <div class="chart" id="dchart"></div>
  <div class="legend" id="dlegend"></div>
 </div>
 <div class="tablewrap" id="dtable"></div>
 <p class="note">Avg3 = mean(HS, ARC-E, PIQA); Avg-all = mean of all benchmark columns present. 760M = ratio-matched cmp_g10 (BoolQ/WinoG main-lineage). Hybrid = canonical leaf/B3 checkpoints, full eval.</p>
</section>

<section id="ablation">
 <h2>100M parsing / boundary-rule ablation</h2>
 <p class="lede">Seven tokenization schemes at a fixed <b>100M / 21 GB</b> budget (AU-Net scaling-law recipe — small-batch/long, distinct from the g10 ladder above). BPB vs 0-shot downstream — the headline is <b>BPB rank ≠ downstream rank</b>: Hybrid leaf_mid wins BPB, Llama leads downstream.</p>
 <div class="card">
  <div class="chart" id="achart"></div>
  <div class="legend" id="alegend"></div>
 </div>
 <div class="tablewrap" id="atable"></div>
 <p class="note">Avg3 = HS/ARC-E/PIQA; all-6 adds ARC-C/BoolQ/WinoGrande. Hybrids at their native question regime (leaf_mid→leafQ, bt→btQ). Source: <span class="mono">reports/leaderboard_100M.md</span>.</p>
</section>

<section id="robust">
 <h2>Robustness &mdash; input perturbation (1.3B)</h2>
 <p class="lede"><b>Noise</b> = HellaSwag under 21 text corruptions (mean acc_norm; bar shows the drop from clean HS). <b>PBP</b> (prompt-boundary problem) = accuracy / BPC change when the space at the prompt boundary is shifted &mdash; byte families are near-immune, Llama BPE is not.</p>
 <div class="card">
  <div class="chart" id="rchart"></div>
  <div class="legend" id="rlegend"></div>
 </div>
 <h3 style="font-size:15px;margin:26px 0 6px;letter-spacing:-.01em">PBP-mc by benchmark — <code>acc</code> vs <code>acc_norm</code> under the same boundary shift</h3>
 <p class="lede" style="margin-bottom:10px">Full test sets (ARC-E 2,376 · ARC-C 1,172 · HellaSwag 10,042; Curated = 10-item controlled diagnostic). Each cell = <b>canonical → space-shifted</b> MC accuracy with the drop (Δ) in parentheses. Two heatmaps, same runs, different scoring metric:</p>
 <p class="lede" style="margin:0 0 6px"><b>① <code>acc</code> — raw total log-prob (the honest cut-invariance metric).</b> A trailing space shifts every option's log-prob by the <i>same</i> constant for a byte model, so the argmax cannot change → byte families read exactly 0. Llama BPE re-tokenizes the tail (a per-option offset) and flips: sensitivity is <b>concentrated</b> in the space-heavy benchmarks (Curated 70→10, ARC-Easy 69→48); ARC-Challenge / HellaSwag barely move.</p>
 <div class="tablewrap" id="rpbp"></div>
 <p class="lede" style="margin:16px 0 6px"><b>② <code>acc_norm</code> — log-prob ÷ continuation byte-length (what the downstream table reports).</b> The per-option length normalization is <b>nonlinear</b>, so even a genuinely cut-invariant byte model drifts ~1 pp (AU-Net ARC-E −1.47, rg −0.93, Hybrid −0.84) — a <b>measurement artifact, not a real boundary sensitivity</b>. Llama's true flip persists but is partly masked (ARC-E −16.46 vs the acc −21.46). This is why PBP is scored on <code>acc</code>.</p>
 <div class="tablewrap" id="rpbpnorm"></div>
 <div class="tablewrap" id="rtable" style="margin-top:18px"></div>
 <p class="note">PBP Δacc = multiple-choice accuracy change (pp) when a space is shifted at the prompt boundary (0 = robust), full test sets. Both heatmaps are the <b>same evaluation runs</b> re-scored under <code>acc</code> vs <code>acc_norm</code>. Avg = macro-mean of Δ over the three real MC benchmarks (excludes the curated diagnostic). Δbpc = bits-per-byte change. Noise Δ = clean HS − noisy HS. Llama PBP via <span class="mono">apps.main.eval</span>, byte families via <span class="mono">apps.aunet.eval</span>. Hybrid noise not evaluated.</p>
</section>

<section id="despace">
 <h2>Despace &mdash; ALL spaces stripped from the prompt (1.3B)</h2>
 <p class="lede">A harsher cousin of PBP: instead of shifting one boundary space, remove <b>every</b> intra-line space from the prompt context &mdash; <span class="mono">"I have a boy" &rarr; "Ihaveaboy"</span> &mdash; leaving the answer choices and the "Answer:" cue intact, then re-score the full test set. This <b>inverts</b> the PBP story: here the byte families are not uniformly immune, and one of them fails hardest of all.</p>
 <div class="card">
  <div class="chart" id="dchart"></div>
  <div class="legend" id="dlegend"></div>
 </div>
 <h3 style="font-size:15px;margin:26px 0 6px;letter-spacing:-.01em">Despace by benchmark — <code>acc_norm</code> (clean → despaced, Δ)</h3>
 <p class="lede" style="margin-bottom:10px">Six benchmarks, full sets (ARC-E 2,376 · ARC-C 1,172 · PIQA 1,838 · BoolQ 3,270 · HellaSwag 10,042 · WinoGrande 1,267). <b>Primary metric is <code>acc_norm</code></b> (matches the downstream table). Each cell = <b>clean → despaced</b> accuracy with the drop (Δ). <b>AU-Net (word) is worst on all six</b> (avg −12.6, ~2&times; the others) — because its level-2 pooling boundaries <i>are</i> the whitespace: strip the spaces and the whole prompt fuses into one un-segmentable "word". Llama (BPE sub-word merges), BPEByte-rg and Hybrid (content / online-BPE segmentation) don't hinge on spaces for structure, so they lose only the space-bytes. WinoGrande shows <code>acc_norm</code>&equiv;<code>acc</code> exactly (all options share one continuation → identical byte-length → normalization is constant).</p>
 <div class="tablewrap" id="dpbp"></div>
 <p class="lede" style="margin:16px 0 6px"><code>acc</code> (clean → despaced, Δ) — same runs, raw total log-prob. The un-normalized view sharpens AU-Net's collapse on the space-structured tasks: ARC-Easy 69.4→<b>35.3</b> (−34.1, near the 4-way chance floor) and BoolQ 61.2→<b>43.9</b> (−17.3, below the majority baseline).</p>
 <div class="tablewrap" id="dpbpnorm"></div>
 <p class="note">Despace Δ = accuracy change (pp) when all prompt spaces are removed (0 = robust), full test sets. Avg = macro-mean of Δ over the six benchmarks. Context-only strip (choices left intact), so the delta isolates prompt-reading. Byte families via <span class="mono">apps.aunet.eval</span>, Llama via <span class="mono">apps.main.eval</span>; sentinel <span class="mono">despace_mc</span>.</p>
</section>

<section id="bpb">
 <h2>Bits-per-byte (final)</h2>
 <p class="lede">Lower is better. Training-loss converted to bits-per-byte (Llama tokens ×4.5483 B/token); the shared measure across subword and byte models.</p>
 <div class="card">
  <div class="chart" id="bchart"></div>
  <div class="legend" id="blegend"></div>
 </div>
 <div class="tablewrap" id="btable"></div>
</section>

<section id="training">
 <h2>Training BPB curves</h2>
 <p class="lede">Bits-per-byte over training bytes (block-averaged, log y, capped at 3). Llama's BPE-token count is converted to bytes (×4.55) so all families share one data axis. Choose a scale.</p>
 <div class="card">
  <div class="chartbar" id="tbar"></div>
  <div class="chart" id="tchart"></div>
  <div class="legend" id="tlegend"></div>
 </div>
</section>

<section id="config">
 <h2>Configuration</h2>
 <p class="lede">Trunk dimensions/layers, tokenizer, training budget, and final BPB per run. AU-Net/byte models are two-stage (encoder × trunk).</p>
 <div class="tablewrap" id="ctable"></div>
</section>

<p class="foot">Generated from <span class="mono">reports/downstream_data.json</span>, <span class="mono">scaling_bpb.csv</span>, per-run <span class="mono">metrics.jsonl</span> and <span class="mono">config.yaml</span>. Hybrid provenance: <span class="mono">runs/main/HYBRID_CANONICAL.md</span>.</p>
</div></div>

<script>
const DATA=__JSON__;
const TILES=__TILES__;
const SER=[["llama","Llama","--s-llama",false],["aunet(word)","AU-Net","--s-aunet",false],
 ["bpebyte_rg","BPEByte-rg","--s-rg",false],["hyb_leafQ","Hybrid leafQ","--s-leaf",true],
 ["hyb_greedyQ","Hybrid greedyQ","--s-greedy",true]];
const SCALES=["100M","300M","760M","1.3B"], SX={"100M":1e8,"300M":3e8,"760M":7.6e8,"1.3B":1.3e9};
const cssv=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function el(t,a,h){const e=document.createElementNS("http://www.w3.org/2000/svg",t);for(const k in a)e.setAttribute(k,a[k]);if(h!=null)e.textContent=h;return e;}
// tiles
document.getElementById("tiles").innerHTML=TILES.map(t=>`<div class="tile"><div class="t-lab">${t[0]}</div><div class="t-val">${t[1]}</div><div class="t-note">${t[2]}</div></div>`).join("");

function smoothPath(P){ // Catmull-Rom -> cubic bezier
 if(P.length<3) return P.map((p,i)=>(i?"L":"M")+p[0]+" "+p[1]).join(" ");
 let d="M"+P[0][0]+" "+P[0][1];
 for(let i=0;i<P.length-1;i++){const p0=P[i-1]||P[i],p1=P[i],p2=P[i+1],p3=P[i+2]||p2;
  const c1x=p1[0]+(p2[0]-p0[0])/6,c1y=p1[1]+(p2[1]-p0[1])/6,c2x=p2[0]-(p3[0]-p1[0])/6,c2y=p2[1]-(p3[1]-p1[1])/6;
  d+=`C${c1x} ${c1y} ${c2x} ${c2y} ${p2[0]} ${p2[1]}`;}
 return d;
}
function lineChart(host,{series,xType,xVals,xLabels,yMin,yMax,yLabel,chance,xLabel,yType,yTicks,smooth}){
 host.innerHTML="";const W=host.clientWidth||820,H=360,m={l:52,r:18,t:16,b:46};
 const iw=W-m.l-m.r, ih=H-m.t-m.b;
 const xs = xType==="log" ? xVals.map(v=>Math.log10(v)) : xVals;
 const xmin=Math.min(...xs), xmax=Math.max(...xs);
 const X=v=>m.l+ (( (xType==="log"?Math.log10(v):v) - xmin)/(xmax-xmin||1))*iw;
 const yl=yType==="log",L=Math.log10;
 const Y=v=>yl?m.t+ih-((L(v)-L(yMin))/(L(yMax)-L(yMin)))*ih:m.t+ih-((v-yMin)/(yMax-yMin))*ih;
 const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%",height:H,preserveAspectRatio:"xMidYMid meet"});
 // y grid
 const yt=yTicks||Array.from({length:6},(_,i)=>yMin+(yMax-yMin)*i/5);
 yt.forEach(v=>{if(v<yMin||v>yMax)return;svg.appendChild(el("line",{class:"grid",x1:m.l,x2:W-m.r,y1:Y(v),y2:Y(v)}));svg.appendChild(el("text",{x:m.l-8,y:Y(v)+3,"text-anchor":"end"},v.toFixed(v<1?2:(v<3?1:0))));});
 if(chance!=null){const l=el("line",{x1:m.l,x2:W-m.r,y1:Y(chance),y2:Y(chance),stroke:cssv("--muted"),"stroke-dasharray":"2 3","stroke-width":1});svg.appendChild(l);svg.appendChild(el("text",{x:m.l+2,y:Y(chance)-4},"chance "+chance));}
 // x ticks
 xVals.forEach((v,i)=>{svg.appendChild(el("text",{x:X(v),y:H-24,"text-anchor":"middle"},xLabels[i]));});
 svg.appendChild(el("text",{x:m.l+iw/2,y:H-6,"text-anchor":"middle"},xLabel||""));
 svg.appendChild(el("text",{x:14,y:m.t+ih/2,"text-anchor":"middle",transform:`rotate(-90 14 ${m.t+ih/2})`},yLabel||""));
 // series
 series.forEach(s=>{
  const pts=s.pts.filter(p=>p[1]!=null);if(!pts.length)return;
  const P=pts.map(p=>[X(p[0]),Y(p[1])]);
  const d=smooth? smoothPath(P) : P.map((p,i)=>(i?"L":"M")+p[0]+" "+p[1]).join(" ");
  svg.appendChild(el("path",{d,fill:"none",stroke:s.color,"stroke-width":2.2,"stroke-dasharray":s.dash?"7 5":"none","stroke-linejoin":"round","stroke-linecap":"round"}));
  if(pts.length<=14) pts.forEach(p=>{svg.appendChild(el("circle",{cx:X(p[0]),cy:Y(p[1]),r:3.4,fill:s.color,stroke:cssv("--surface"),"stroke-width":1.4}));});
 });
 host.appendChild(svg);
}
function legend(host,items){host.innerHTML=items.map(s=>`<span><i class="${s.dash?'dash':''}" style="background:${s.color};color:${s.color}"></i>${s.label}</span>`).join("");}

// ---- downstream ----
const BENCHES=[["hellaswag","HellaSwag",25],["arc_easy","ARC-Easy",25],["arc_challenge","ARC-Challenge",25],["piqa","PIQA",50],["boolq","BoolQ",50],["winogrande","WinoGrande",50],["mmlu_text","MMLU-text",25],["_avg3","Avg3 (HS/ARC-E/PIQA)",null],["_avgall","Avg-all",null]];
let curBench="_avg3";
function drawDown(){
 const dd=DATA.downstream.data;
 const series=SER.map(([k,lab,cv,dash])=>({label:lab,color:cssv(cv),dash,pts:SCALES.map(s=>[SX[s],(dd[k][s]&&dd[k][s][curBench]!=null)?dd[k][s][curBench]:null])}));
 const all=series.flatMap(s=>s.pts.map(p=>p[1])).filter(v=>v!=null);
 let yMin=Math.floor(Math.min(...all,curBench.startsWith("_")?35:22)/5)*5, yMax=Math.ceil(Math.max(...all)/5)*5+2;
 const b=BENCHES.find(b=>b[0]===curBench);
 lineChart(document.getElementById("dchart"),{series,xType:"log",xVals:SCALES.map(s=>SX[s]),xLabels:SCALES,yMin,yMax,yLabel:"accuracy (%)",chance:b?b[2]:null,xLabel:"model scale (params, log)"});
 legend(document.getElementById("dlegend"),series);
}
document.getElementById("dbar").innerHTML=BENCHES.map(b=>`<button class="chip${b[0]===curBench?' on':''}" data-b="${b[0]}">${b[1]}</button>`).join("");
document.getElementById("dbar").addEventListener("click",e=>{if(!e.target.dataset.b)return;curBench=e.target.dataset.b;document.querySelectorAll("#dbar .chip").forEach(c=>c.classList.toggle("on",c.dataset.b===curBench));drawDown();});
// downstream table
(function(){
 const dd=DATA.downstream.data,cols=["hellaswag","arc_easy","arc_challenge","piqa","boolq","winogrande","mmlu_text"];
 const head=["Scale","Family","HS","ARC-E","ARC-C","PIQA","BoolQ","WinoG","MMLU","Avg3","Avg-all"];
 let h="<table><thead><tr>"+head.map(x=>`<th>${x}</th>`).join("")+"</tr></thead><tbody>";
 SCALES.forEach(s=>{let first=true;SER.forEach(([k,lab,cv])=>{const c=dd[k][s];if(!c&&!(k.startsWith('hyb')&&s==='760M'))return;
  const cells=c||{};const row=cols.map(b=>`<td>${cells[b]!=null?cells[b].toFixed(1):'—'}</td>`).join("");
  h+=`<tr class="${first?'scale-lead':''}"><td>${first?'<b>'+s+'</b>':''}</td><td class="fam"><span class="dot" style="background:${cssv(cv)}"></span>${lab}</td>${row}<td class="avg">${cells._avg3!=null?cells._avg3.toFixed(1):'—'}</td><td class="avg">${cells._avgall!=null?cells._avgall.toFixed(1):'—'}</td></tr>`;first=false;});});
 h+="</tbody></table>";document.getElementById("dtable").innerHTML=h;
})();

// ---- bpb ----
(function(){
 const rows=DATA.bpb, fam={"llama":["Llama","--s-llama"],"aunet(word)":["AU-Net","--s-aunet"],"bpebyte_rg":["BPEByte-rg","--s-rg"],"hybrid":["Hybrid","--s-leaf"]};
 const series=Object.entries(fam).map(([m,[lab,cv]])=>({label:lab,color:cssv(cv),dash:m==="hybrid",pts:SCALES.map(s=>{const r=rows.find(r=>r.model===m&&r.scale===s);return [SX[s],r?+r.bpb:null];})}));
 lineChart(document.getElementById("bchart"),{series,xType:"log",xVals:SCALES.map(s=>SX[s]),xLabels:SCALES,yMin:0.8,yMax:1.25,yLabel:"bits per byte (↓)",xLabel:"model scale (params, log)"});
 legend(document.getElementById("blegend"),series);
 const head=["Model","Scale","Params","Final BPB","Last step"];
 let h="<table><thead><tr>"+head.map(x=>`<th>${x}</th>`).join("")+"</tr></thead><tbody>";
 rows.sort((a,b)=>(+a.params-+b.params)||a.model.localeCompare(b.model)).forEach(r=>{h+=`<tr><td class="fam">${(fam[r.model]||[r.model])[0]}</td><td>${r.scale}</td><td>${(+r.params/1e6).toFixed(0)}M</td><td class="avg">${(+r.bpb).toFixed(4)}</td><td>${(+r.laststep).toLocaleString()}</td></tr>`;});
 h+="</tbody></table>";document.getElementById("btable").innerHTML=h;
})();

// ---- training curves ----
let curScale="1.3B";
function drawTrain(){
 const c=DATA.curves, famcol={"llama":"--s-llama","aunet(word)":"--s-aunet","bpebyte_rg":"--s-rg","hybrid":"--s-leaf"};
 const labs={"llama":"Llama","aunet(word)":"AU-Net","bpebyte_rg":"BPEByte-rg","hybrid":"Hybrid"};
 const YCAP=3.0, BPT=4.5483;   // Llama BPE tokens -> bytes for a common x-axis
 const series=Object.keys(famcol).map(m=>{const k=m+"|"+curScale,cv=c[k];if(!cv)return null;
   const mul=(m==="llama")?BPT:1;
   const pts=cv.tokens.map((t,i)=>[t==null?null:t*mul,cv.bpb[i]]).filter(p=>p[0]!=null&&p[1]<=YCAP);
   return pts.length?{label:labs[m],color:cssv(famcol[m]),dash:m==="hybrid",pts}:null;}).filter(Boolean);
 if(!series.length){document.getElementById("tchart").innerHTML="<p class='note'>No curve at this scale.</p>";document.getElementById("tlegend").innerHTML="";return;}
 const maxtok=Math.max(...series.flatMap(s=>s.pts.map(p=>p[0])));
 const fmt=t=>t>=1e9?(t/1e9).toFixed(0)+"B":(t/1e6).toFixed(0)+"M";
 lineChart(document.getElementById("tchart"),{series,smooth:true,xType:"linear",xVals:[0,maxtok*.25,maxtok*.5,maxtok*.75,maxtok],xLabels:["0",fmt(maxtok*.25),fmt(maxtok*.5),fmt(maxtok*.75),fmt(maxtok)],yType:"log",yMin:0.79,yMax:YCAP,yTicks:[0.8,1,1.5,2,3],yLabel:"bits per byte (↓, log)",xLabel:"training bytes"});
 legend(document.getElementById("tlegend"),series);
}
document.getElementById("tbar").innerHTML=SCALES.map(s=>`<button class="chip${s===curScale?' on':''}" data-s="${s}">${s}</button>`).join("");
document.getElementById("tbar").addEventListener("click",e=>{if(!e.target.dataset.s)return;curScale=e.target.dataset.s;document.querySelectorAll("#tbar .chip").forEach(c=>c.classList.toggle("on",c.dataset.s===curScale));drawTrain();});

// ---- config table ----
(function(){
 const c=DATA.config, order=["llama","aunet(word)","bpebyte_rg","hybrid"], labs={"llama":"Llama","aunet(word)":"AU-Net","bpebyte_rg":"BPEByte-rg","hybrid":"Hybrid"};
 const head=["Scale","Model","Params","Trunk dims","Layers","Tokenizer","Vocab","Steps","Batch×GA","LR","Prefill/Boundary","Final BPB","Checkpoint"];
 let h="<table><thead><tr>"+head.map(x=>`<th>${x}</th>`).join("")+"</tr></thead><tbody>";
 SCALES.forEach(s=>{let first=true;order.forEach(m=>{const v=c[m+"|"+s];if(!v)return;
  const bg=v.batch&&v.grad_acc?`${v.batch}×${v.grad_acc}`:(v.batch||"—");
  const pb=(v.prefill||v.boundary)?`${v.prefill||"—"} / ${v.boundary||"—"}`:"—";
  const ck=`<span class="mono" style="font-size:10.5px;color:var(--ink2)">${v.ckpt}</span>`+(v.ckpt_local?'':' <em style="font-style:normal;color:var(--warn)">· on ece</em>');
  h+=`<tr class="${first?'scale-lead':''}"><td>${first?'<b>'+s+'</b>':''}</td><td class="fam">${labs[m]}</td><td>${(+v.params/1e6).toFixed(0)}M</td><td>${v.dims||"—"}</td><td>${v.layers||"—"}</td><td>${v.tokenizer||"—"}</td><td>${v.vocab||"—"}</td><td>${v.steps?(+v.steps).toLocaleString():"—"}</td><td>${bg}</td><td>${v.lr||"—"}</td><td>${pb}</td><td class="avg">${(+v.final_bpb).toFixed(4)}</td><td style="text-align:left">${ck}</td></tr>`;first=false;});});
 h+="</tbody></table>";document.getElementById("ctable").innerHTML=h;
})();

// ---- robustness ----
function drawRobust(){
 const R=DATA.robust, fam={"llama":["Llama","--s-llama"],"aunet(word)":["AU-Net","--s-aunet"],"bpebyte_rg":["BPEByte-rg","--s-rg"],"hybrid":["Hybrid","--s-leaf"]};
 const host=document.getElementById("rchart");const W=host.clientWidth||820,H=210,m={l:104,r:28,t:22,b:32};
 const iw=W-m.l-m.r,ih=H-m.t-m.b;
 const keys=Object.keys(fam).filter(k=>R[k]&&R[k].pbp_mc_dacc!=null);
 const vals=keys.map(k=>R[k].pbp_mc_dacc); const lo=Math.min(-10,...vals), hi=Math.max(1,...vals);
 const X=v=>m.l+((v-lo)/(hi-lo))*iw; const bw=Math.min(30,ih/keys.length*0.55);
 const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%",height:H});
 svg.appendChild(el("text",{class:"ttl",x:m.l-2,y:12},"PBP · MC accuracy change under prompt-boundary shift (pp)"));
 svg.appendChild(el("line",{x1:X(0),x2:X(0),y1:m.t,y2:m.t+ih,stroke:cssv("--border")}));
 svg.appendChild(el("text",{x:X(0),y:H-6,"text-anchor":"middle"},"0 = robust"));
 keys.forEach((k,i)=>{const v=R[k].pbp_mc_dacc,cy=m.t+ih*(i+0.5)/keys.length,col=cssv(fam[k][1]);
  svg.appendChild(el("rect",{x:Math.min(X(0),X(v)),y:cy-bw/2,width:Math.max(1.5,Math.abs(X(v)-X(0))),height:bw,fill:col,rx:3}));
  svg.appendChild(el("text",{x:m.l-10,y:cy+4,"text-anchor":"end",fill:cssv("--ink")},fam[k][0]));
  svg.appendChild(el("text",{x:X(v)+(v<0?-6:6),y:cy+4,"text-anchor":v<0?"end":"start"},v.toFixed(2)));
 });
 host.appendChild(svg);
 document.getElementById("rlegend").innerHTML="<span>Byte families (AU-Net · BPEByte-rg · Hybrid) are near-immune to boundary shifts; Llama BPE drops ~9 pp on average.</span>";
 // per-benchmark PBP-mc heatmaps — ① acc, ② acc_norm (same runs, different metric)
 const BEN=[["curated","Curated"],["arc_easy","ARC-Easy"],["arc_challenge","ARC-Chal."],["hellaswag","HellaSwag"]];
 const REAL=["arc_easy","arc_challenge","hellaswag"];
 function pbpTable(cf,df,lf){                        // canon-field, degraded-field, delta-field
  let hh="<table><thead><tr><th>Model</th>"+BEN.map(b=>`<th>${b[1]}</th>`).join("")+"<th>Avg</th></tr></thead><tbody>";
  Object.keys(fam).forEach(k=>{const r=R[k];if(!r||!r.pbp_by_bench)return;
   hh+=`<tr><td class="fam"><span class="dot" style="background:${cssv(fam[k][1])}"></span>${fam[k][0]}</td>`;
   BEN.forEach(([bk])=>{const o=r.pbp_by_bench[bk]||{},dl=o[lf],dg=o[df],cn=o[cf];
    const a=dl==null?0:Math.min(0.88,Math.abs(dl)/60*0.9+(dl<=-0.3?0.06:0));
    const txt=dg==null?'—':`${cn!=null?cn.toFixed(1)+'→':''}${dg.toFixed(1)} <span style="opacity:.7">(${dl>0?'+':''}${dl.toFixed(2)})</span>`;
    hh+=`<td style="${dl==null?'':'background:rgba(213,94,0,'+a.toFixed(3)+');color:'+(a>0.45?'#fff':'var(--ink2)')}">${txt}</td>`;});
   const av=REAL.map(b=>(r.pbp_by_bench[b]||{})[lf]).filter(v=>v!=null);
   const avg=av.length?(av.reduce((s,v)=>s+v,0)/av.length):null;
   hh+=`<td class="avg">${avg==null?'—':(avg>0?'+':'')+avg.toFixed(2)}</td></tr>`;});
  return hh+"</tbody></table>";
 }
 document.getElementById("rpbp").innerHTML=pbpTable("canon","degraded","delta");
 document.getElementById("rpbpnorm").innerHTML=pbpTable("canon_norm","degraded_norm","delta_norm");
 const head=["Model","Clean HS","Noise HS","Noise Δ","PBP Δacc","PBP Δbpc"];
 let h="<table><thead><tr>"+head.map(x=>`<th>${x}</th>`).join("")+"</tr></thead><tbody>";
 Object.keys(fam).forEach(k=>{const r=R[k];if(!r)return;const nd=(r.clean_hs!=null&&r.noise_hs!=null)?(r.clean_hs-r.noise_hs).toFixed(1):"—";
  h+=`<tr><td class="fam"><span class="dot" style="background:${cssv(fam[k][1])}"></span>${fam[k][0]}</td><td>${r.clean_hs??"—"}</td><td>${r.noise_hs??"—"}</td><td>${nd}</td><td class="avg">${r.pbp_mc_dacc??"—"}</td><td>${r.pbp_dbpc??"—"}</td></tr>`;});
 h+="</tbody></table>";document.getElementById("rtable").innerHTML=h;
}

// ---- despace (space-strip) ----
function drawDespace(){
 const R=DATA.despace, fam={"llama":["Llama","--s-llama"],"aunet(word)":["AU-Net","--s-aunet"],"bpebyte_rg":["BPEByte-rg","--s-rg"],"hybrid":["Hybrid","--s-leaf"]};
 const keys=Object.keys(fam).filter(k=>R[k]&&R[k].avg_delta!=null);
 // avg Δacc bar chart
 const host=document.getElementById("dchart");const W=host.clientWidth||820,H=210,m={l:104,r:34,t:22,b:32};
 const iw=W-m.l-m.r,ih=H-m.t-m.b;
 const vals=keys.map(k=>R[k].avg_delta_norm); const lo=Math.min(-13,...vals), hi=Math.max(1,...vals);
 const X=v=>m.l+((v-lo)/(hi-lo))*iw;
 const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%",height:H});
 svg.appendChild(el("text",{class:"ttl",x:m.l-2,y:12},"Despace · avg MC accuracy change (acc_norm, pp) over 6 benchmarks"));
 svg.appendChild(el("line",{x1:X(0),x2:X(0),y1:m.t,y2:m.t+ih,stroke:cssv("--border")}));
 svg.appendChild(el("text",{x:X(0),y:H-6,"text-anchor":"middle"},"0 = robust"));
 keys.forEach((k,i)=>{const v=R[k].avg_delta_norm,cy=m.t+ih*(i+0.5)/keys.length,col=cssv(fam[k][1]),bw=Math.min(30,ih/keys.length*0.55);
  svg.appendChild(el("rect",{x:Math.min(X(0),X(v)),y:cy-bw/2,width:Math.max(1.5,Math.abs(X(v)-X(0))),height:bw,fill:col,rx:3}));
  svg.appendChild(el("text",{x:m.l-10,y:cy+4,"text-anchor":"end",fill:cssv("--ink")},fam[k][0]));
  svg.appendChild(el("text",{x:X(v)+(v<0?-6:6),y:cy+4,"text-anchor":v<0?"end":"start"},v.toFixed(2)));
 });
 host.appendChild(svg);
 document.getElementById("dlegend").innerHTML="<span>acc_norm — AU-Net (word) worst (avg −12.6): its pooling boundaries ARE the whitespace; Llama · BPEByte-rg · Hybrid degrade ~2× less (−6.8 to −7.0).</span>";
 // per-benchmark heatmaps
 const BEN=[["arc_easy","ARC-Easy"],["arc_challenge","ARC-Chal."],["piqa","PIQA"],["boolq","BoolQ"],["hellaswag","HellaSwag"],["winogrande","WinoGr."]];
 function dTable(cf,df,lf,af){
  let hh="<table><thead><tr><th>Model</th>"+BEN.map(b=>`<th>${b[1]}</th>`).join("")+"<th>Avg</th></tr></thead><tbody>";
  keys.forEach(k=>{const r=R[k];const by=r.by_bench||{};
   hh+=`<tr><td class="fam"><span class="dot" style="background:${cssv(fam[k][1])}"></span>${fam[k][0]}</td>`;
   BEN.forEach(([bk])=>{const o=by[bk]||{},dl=o[lf],dg=o[df],cn=o[cf];
    const a=dl==null?0:Math.min(0.9,Math.abs(dl)/36*0.85+(dl<=-0.3?0.06:0));
    const txt=dg==null?'—':`${cn!=null?cn.toFixed(1)+'→':''}${dg.toFixed(1)} <span style="opacity:.7">(${dl>0?'+':''}${dl.toFixed(2)})</span>`;
    hh+=`<td style="${dl==null?'':'background:rgba(213,94,0,'+a.toFixed(3)+');color:'+(a>0.5?'#fff':'var(--ink2)')}">${txt}</td>`;});
   const avg=r[af];
   hh+=`<td class="avg">${avg==null?'—':(avg>0?'+':'')+avg.toFixed(2)}</td></tr>`;});
  return hh+"</tbody></table>";
 }
 document.getElementById("dpbp").innerHTML=dTable("clean_norm","degraded_norm","delta_norm","avg_delta_norm");
 document.getElementById("dpbpnorm").innerHTML=dTable("clean","degraded","delta","avg_delta");
}

// ---- 100M parsing ablation ----
function drawAblation(){
 const A=DATA.ablation.rows, gray="#8a97a8";
 const col={"Llama (subword)":"--s-llama","AU-Net (word)":"--s-aunet","BPEByte root_greedy":"--s-rg","Hybrid leaf_mid":"--s-leaf","Hybrid bt":"--s-greedy"};
 const host=document.getElementById("achart");const W=host.clientWidth||820,H=380,m={l:52,r:24,t:16,b:46};
 const iw=W-m.l-m.r,ih=H-m.t-m.b, xmin=1.03,xmax=1.115,ymin=36,ymax=49;
 const X=v=>m.l+(v-xmin)/(xmax-xmin)*iw, Y=v=>m.t+ih-(v-ymin)/(ymax-ymin)*ih;
 const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%",height:H});
 for(let i=0;i<=5;i++){const v=ymin+(ymax-ymin)*i/5;svg.appendChild(el("line",{class:"grid",x1:m.l,x2:m.l+iw,y1:Y(v),y2:Y(v)}));svg.appendChild(el("text",{x:m.l-8,y:Y(v)+3,"text-anchor":"end"},v.toFixed(0)));}
 [1.04,1.06,1.08,1.10].forEach(v=>{svg.appendChild(el("line",{class:"grid",x1:X(v),x2:X(v),y1:m.t,y2:m.t+ih}));svg.appendChild(el("text",{x:X(v),y:H-26,"text-anchor":"middle"},v.toFixed(2)));});
 svg.appendChild(el("text",{x:m.l+iw/2,y:H-6,"text-anchor":"middle"},"BPB  (← better)"));
 svg.appendChild(el("text",{x:14,y:m.t+ih/2,"text-anchor":"middle",transform:`rotate(-90 14 ${m.t+ih/2})`},"Avg3 acc (%)  (↑ better)"));
 A.forEach(r=>{const c=col[r.model]?cssv(col[r.model]):gray;
  svg.appendChild(el("circle",{cx:X(r.bpb),cy:Y(r.avg3),r:6,fill:c,stroke:cssv("--surface"),"stroke-width":1.5}));
  const lab=r.model.replace(" (subword)","").replace(" (word)","");
  svg.appendChild(el("text",{x:X(r.bpb)+9,y:Y(r.avg3)+3,fill:cssv("--ink")},lab));});
 host.appendChild(svg);
 document.getElementById("alegend").innerHTML="<span>Upper-left is best (low BPB + high accuracy). Llama sits top (best downstream) but not lowest BPB; the hybrids reach the lowest BPB at mid accuracy.</span>";
 const cols=["hs","arce","arcc","boolq","piqa","wino"];
 const head=["Model","Boundary rule","BPB","HS","ARC-E","ARC-C","BoolQ","PIQA","WinoG","Avg3","all-6"];
 let h="<table><thead><tr>"+head.map(x=>`<th>${x}</th>`).join("")+"</tr></thead><tbody>";
 A.slice().sort((a,b)=>a.bpb-b.bpb).forEach(r=>{const c=col[r.model]?cssv(col[r.model]):gray;
  h+=`<tr><td class="fam"><span class="dot" style="background:${c}"></span>${r.model}</td><td style="text-align:left;color:var(--muted)">${r.rule}</td><td class="avg">${r.bpb.toFixed(3)}</td>`+cols.map(k=>`<td>${r[k].toFixed(1)}</td>`).join("")+`<td class="avg">${r.avg3.toFixed(1)}</td><td>${r.all6.toFixed(1)}</td></tr>`;});
 h+="</tbody></table>";document.getElementById("atable").innerHTML=h;
}

// nav scrollspy
const secs=[...document.querySelectorAll("section")];
const io=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){const id=e.target.id;document.querySelectorAll("nav.sticky a").forEach(a=>a.classList.toggle("on",a.getAttribute("href")==="#"+id));}});},{rootMargin:"-40% 0px -55% 0px"});
secs.forEach(s=>io.observe(s));

drawDown();drawTrain();drawRobust();drawDespace();drawAblation();
addEventListener("resize",()=>{drawDown();drawTrain();drawRobust();drawDespace();drawAblation();});
</script>'''
HTML=HTML.replace("__JSON__",J).replace("__TILES__",json.dumps(tiles))
open(f"{ROOT}/reports/scaling_dashboard.html","w").write(HTML)
print("wrote reports/scaling_dashboard.html", len(HTML)//1024,"KB")
