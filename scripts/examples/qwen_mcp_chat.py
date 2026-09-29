#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""A chat page for the local vision model that labels through the MCP bridge.

    python scripts/examples/qwen_mcp_chat.py [--host 127.0.0.1] [--port 8765]
        [--model qwen3.8:27b] [--backend ollama] [--base-url URL] [--api-key-env NAME]
        [--api http://127.0.0.1:8002] [--lang ja|en]

Type an instruction -- "Project <project-id>. Mark the dark specks on the
contact pads in img001 and img002." -- and watch the model call the bridge's
tools, one line per call, until it answers. One conversation per page load,
one job at a time.

The page has no login, and the bridge behind it runs under --policy write.
Whoever can send this server a request can write masks into the trainer's
projects, can read the conversation with its pictures, and decides which
model server gets the project's images -- together with the key named by
--api-key-env, which goes out on every request to that server. So:

- it listens on this machine only: --host must be a loopback address, and
  another machine reaches it through an SSH tunnel, not over the network;
- a request whose Host is not that address is refused, so a web page whose
  name was pointed at 127.0.0.1 gets nothing;
- a POST must be JSON and must come from this page: a POST from any other
  page's origin is refused before it reaches a route;
- where the model server is (--base-url) and which environment variable
  holds its key (--api-key-env) come from the command line and nowhere else.
  The page chooses between the known servers and names a model; it cannot
  send the key, or the images, anywhere the command line did not say.
"""
from __future__ import annotations

import argparse
import asyncio
import html
import json
import os
import re
import sys
import time
import urllib.error
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_mcp_agent import run  # noqa: E402
from vlm_backends import ALIASES, describe, make_backend  # noqa: E402

PAGE = r"""<!doctype html><html lang="{{lang}}"><meta charset="utf-8"><title>{{title}}</title>
<style>
 :root{color-scheme:light}
 :lang(ja){font-family:system-ui,"Hiragino Sans","Yu Gothic UI","Noto Sans JP",sans-serif}
 *{box-sizing:border-box}
 body{font:14px system-ui,sans-serif;margin:0;background:#f4f4f5;color:#1c1c1e;height:100vh;overflow:hidden}
 header{display:flex;align-items:baseline;gap:12px;padding:10px 16px;background:#fff;border-bottom:1px solid #e3e3e6}
 header h1{font-size:15px;margin:0}
 header small{color:#6b6b70}
 .cols{display:grid;grid-template-columns:var(--left,42%) 6px 1fr;height:calc(100vh - 45px)}
 .grip{cursor:col-resize;background:#e3e3e6;position:relative}
 .grip:hover,.grip.dragging{background:#D55E00}
 .grip::after{content:"";position:absolute;inset:0 -5px;} /* a wider grab area than the line */
 .col{display:flex;flex-direction:column;min-width:0;min-height:0}
 .col.right{border-left:1px solid #e3e3e6;background:#fbfbfc}
 .colhead{padding:6px 14px;font-size:11px;font-weight:700;letter-spacing:.06em;color:#6b6b70;background:#f7f7f8;border-bottom:1px solid #ececef}
 #chat,#side{flex:1;overflow:auto;padding:14px;min-height:0}
 /* left: the conversation */
 .msg{margin:0 0 12px;max-width:92%;padding:9px 12px;border-radius:12px;line-height:1.55;white-space:pre-wrap;word-break:break-word}
 .msg.me{margin-left:auto;background:#D55E00;color:#fff;border-bottom-right-radius:3px}
 .msg.vlm{background:#fff;border:1px solid #e3e3e6;border-bottom-left-radius:3px}
 .msg.q{background:#fff6f0;border:1px solid #D55E00;border-bottom-left-radius:3px}
 .msg.q b{color:#D55E00}
 /* a question with settled answers: buttons, in one column, same width */
 .choices{display:flex;flex-direction:column;gap:8px;margin-top:10px}
 .choices button{height:34px;padding:0 12px;font-size:13px;text-align:left;background:#fff;color:#1c1c1e;border:1px solid #D55E00}
 .choices button:hover:not(:disabled){background:#D55E00;color:#fff}
 .choices.done button{border-color:#d6d6da;color:#8a8a8f}
 .choices.done button.picked{border-color:#D55E00;color:#fff;background:#D55E00}
 .msg.sys{background:transparent;color:#8a8a8f;font-size:12px;padding:2px 0;text-align:center;max-width:100%}
 .msg.err{background:#fff0f0;border:1px solid #c33;color:#a00}
 .working{color:#6b6b70;font-size:12px;padding:2px 4px}
 .working b{color:#D55E00}
 /* right: what it looked at and what it called */
 .shot{margin:0 0 14px}
 .shot img{width:100%;border:1px solid #ddd;border-radius:8px;display:block}
 .shot small{display:block;color:#6b6b70;margin-top:4px}
 .call{font-family:ui-monospace,Consolas,monospace;font-size:11.5px;line-height:1.5;padding:5px 8px;margin-bottom:5px;background:#fff;border:1px solid #ececef;border-radius:6px;overflow-wrap:anywhere}
 .call b{color:#1c4e80}
 .call .res{color:#5a5a60}
 .call .t{color:#9a9aa0;float:right}
 .call.bad{border-color:#e0b4b4;background:#fffafa}
 .call.bad .res{color:#a00}
 /* composer */
 form{display:flex;gap:8px;padding:10px 14px;border-top:1px solid #e3e3e6;background:#fff}
 textarea{flex:1;height:60px;font:inherit;padding:8px;border:1px solid #d6d6da;border-radius:8px;resize:none}
 .buttons{display:flex;flex-direction:column;gap:6px}
 button{padding:7px 14px;font:inherit;font-weight:600;background:#D55E00;color:#fff;border:0;border-radius:8px;cursor:pointer}
 button:disabled{opacity:.45;cursor:default}
 #stop{background:#8a8a8f}
 .toolbar{display:flex;align-items:center;gap:8px;padding:6px 14px;border-top:1px solid #ececef;background:#fafafb;font-size:12px;flex-wrap:wrap}
 .toolbar button{padding:4px 10px;font-size:12px;font-weight:600;background:#fff;color:#1c1c1e;border:1px solid #d6d6da;border-radius:6px}
 .toolbar button:not(:disabled):hover{border-color:#D55E00;color:#D55E00}
 .toolbar button.on{background:#D55E00;color:#fff;border-color:#D55E00}
 .toolbar label{display:inline-flex;align-items:center;gap:5px;color:#4a4a50;cursor:pointer}
 .toolbar .spacer{flex:1}
 #status{color:#6b6b70}
 #ctx{color:#6b6b70;font-variant-numeric:tabular-nums}
 #ctx .bar{display:inline-block;width:70px;height:6px;background:#e3e3e6;border-radius:3px;overflow:hidden;vertical-align:middle;margin:0 5px}
 #ctx .bar i{display:block;height:100%;background:#8a8a8f}
 #ctx.warn .bar i{background:#D55E00}
 #ctx.warn{color:#D55E00;font-weight:600}
 .msg.paused{background:#fffaf3;border:1px dashed #D55E00;color:#8a4a10}
 #stop:not(:disabled){background:#4a4a50;box-shadow:0 0 0 2px rgba(213,94,0,.4)}
 .empty{color:#9a9aa0;font-size:12px;padding:8px}
 .where{color:#9a9aa0}
 /* the connection dialog: which model answers, and where it runs */
 dialog#conndlg{border:0;border-radius:12px;padding:0;width:min(460px,92vw);box-shadow:0 10px 40px rgba(0,0,0,.22)}
 dialog#conndlg::backdrop{background:rgba(0,0,0,.28)}
 .dlg-head{padding:12px 16px;border-bottom:1px solid #ececef;font-weight:700}
 .dlg-body{padding:14px 16px;display:flex;flex-direction:column;gap:10px}
 .dlg-body label{display:flex;flex-direction:column;gap:4px;font-size:12px;color:#4a4a50}
 .dlg-body input,.dlg-body select{font:inherit;padding:7px 8px;border:1px solid #d6d6da;border-radius:8px;background:#fff;color:#1c1c1e}
 .dlg-body input[readonly]{background:#f4f4f5;color:#6b6b70}
 .dlg-body .hint{color:#8a8a8f;font-size:11.5px}
 .dlg-foot{display:flex;gap:8px;padding:0 16px 14px}
 .dlg-foot button{flex:1;height:34px;padding:0}
 .dlg-foot button.plain{background:#fff;color:#1c1c1e;border:1px solid #d6d6da}
 #conn-said{padding:0 16px 14px;font-size:12px;min-height:18px;color:#6b6b70;white-space:pre-wrap}
 #conn-said.bad{color:#D55E00;font-weight:600}
</style>
<header>
  <h1>{{title}}</h1>
  <small>{{h_model}} <span id="hmodel">__MODEL__</span> <span class="where" id="hwhere">__WHERE__</span> {{h_tail}}</small>
</header>
<dialog id="conndlg">
  <div class="dlg-head">{{d_head}}</div>
  <div class="dlg-body">
    <label>{{d_server}}
      <select id="c-backend"></select>
      <span class="hint" id="c-kind"></span>
    </label>
    <label>{{d_url}}
      <input id="c-url" type="text" spellcheck="false" readonly>
      <span class="hint">{{d_url_hint}}</span>
    </label>
    <label>{{d_model}}
      <input id="c-model" type="text" list="c-models" spellcheck="false" placeholder="qwen3.8:27b">
      <datalist id="c-models"></datalist>
      <span class="hint">{{d_model_hint}}</span>
    </label>
    <label>{{d_key}}
      <input id="c-key" type="text" spellcheck="false" readonly>
      <span class="hint">{{d_key_hint}}</span>
    </label>
  </div>
  <div class="dlg-foot">
    <button id="c-test" type="button" class="plain">{{d_test}}</button>
    <button id="c-save" type="button">{{d_save}}</button>
    <button id="c-close" type="button" class="plain">{{d_close}}</button>
  </div>
  <div id="conn-said"></div>
</dialog>
<div class="cols">
  <div class="col">
    <div class="colhead">{{col_chat}}</div>
    <div id="chat"></div>
    <div class="toolbar">
      <button id="pause" type="button" disabled title="{{t_pause}}">{{b_pause}}</button>
      <button id="newsession" type="button" title="{{t_new}}">{{b_new}}</button>
      <button id="reset" type="button" title="{{t_reset}}">{{b_reset}}</button>
      <button id="connbtn" type="button" title="{{t_conn}}">{{b_conn}}</button>
      <label title="{{t_confirm}}"><input type="checkbox" id="confirm" checked>{{b_confirm}}</label>
      <span class="spacer"></span>
      <span id="status">{{s_idle}}</span>
      <span id="ctx" title="{{t_ctx}}"></span>
    </div>
    <form id="f">
      <textarea id="q" placeholder="{{p_instruction}}"></textarea>
      <div class="buttons"><button id="b">{{b_send}}</button><button id="stop" type="button" disabled>{{b_stop}}</button></div>
    </form>
  </div>
  <div class="grip" id="grip" role="separator" aria-orientation="vertical" title="{{t_grip}}"></div>
  <div class="col right">
    <div class="colhead">{{col_side}}</div>
    <div id="side"><div class="empty">{{side_empty}}</div></div>
  </div>
</div>
<script>
// Every word the script shows, in the language the server was started with.
const T={{js}};
const JSONH={'content-type':'application/json'};
function fmt(s,v){return s.replace(/\{(\w+)\}/g,(m,k)=>(k in v)?String(v[k]):m);}
const chat=document.getElementById('chat'),side=document.getElementById('side'),q=document.getElementById('q'),
      b=document.getElementById('b'),f=document.getElementById('f'),stopBtn=document.getElementById('stop'),
      pauseBtn=document.getElementById('pause'),resetBtn=document.getElementById('reset'),
      newBtn=document.getElementById('newsession'),
      confirmBox=document.getElementById('confirm'),statusEl=document.getElementById('status'),
      ctxEl=document.getElementById('ctx');
function setCtx(ev){const pct=Math.min(100,ev.pct);
  ctxEl.className=pct>=80?'warn':'';
  ctxEl.innerHTML=T.ctx+' <span class="bar"><i style="width:'+pct+'%"></i></span>'+pct+'%';}
function emptySide(){side.innerHTML='';const d=document.createElement('div');d.className='empty';d.textContent=T.sideEmpty;side.appendChild(d);}
let paused=false,toolCount=0;
// One conversation per tab. sessionStorage, not localStorage: a second tab is
// a second conversation, and closing the tab lets its history go.
function newId(){return (crypto.randomUUID?crypto.randomUUID():String(Math.random()).slice(2))}
let session=(()=>{try{let v=sessionStorage.getItem('seg.vlm.session');if(!v){v=newId();sessionStorage.setItem('seg.vlm.session',v);}return v;}catch(e){return newId();}})();
function setStatus(s){statusEl.textContent=s;}
// The toggles are remembered: a person who turns confirmation off for a batch
// does not want it back on the next page load.
try{confirmBox.checked=localStorage.getItem('seg.vlm.confirm')!=='0';}catch(e){}
confirmBox.addEventListener('change',()=>{try{localStorage.setItem('seg.vlm.confirm',confirmBox.checked?'1':'0');}catch(e){}});
pauseBtn.addEventListener('click',async()=>{
  const r=await fetch('/pause',{method:'POST',headers:JSONH,body:JSON.stringify({paused:!paused})});
  const d=await r.json();paused=d.paused;pauseBtn.textContent=paused?T.resume:T.pause;pauseBtn.classList.toggle('on',paused);
  setStatus(paused?T.paused:T.running);});
newBtn.addEventListener('click',()=>{
  if(running){say('sys',T.busyNew);return;}
  session=newId();try{sessionStorage.setItem('seg.vlm.session',session);}catch(e){}
  chat.innerHTML='';emptySide();
  toolCount=0;setStatus(T.idle);say('sys',T.newStarted);});
resetBtn.addEventListener('click',async()=>{
  const r=await fetch('/reset',{method:'POST',headers:JSONH,body:JSON.stringify({session:session})});const d=await r.json();
  if(d.status==='busy'){say('sys',T.busyReset);return;}
  chat.innerHTML='';emptySide();
  toolCount=0;setStatus(T.idle);say('sys',T.resetDone);});
let pending=false,working=null,running=false,queued=null;
// The divider: drag it, and the width is remembered per browser.
const cols=document.querySelector('.cols'),grip=document.getElementById('grip');
const savedW=localStorage.getItem('seg.vlm.leftWidth');
if(savedW)cols.style.setProperty('--left',savedW);
grip.addEventListener('pointerdown',e=>{e.preventDefault();grip.setPointerCapture(e.pointerId);grip.classList.add('dragging');
 const move=ev=>{const r=cols.getBoundingClientRect();const px=Math.min(Math.max(ev.clientX-r.left,280),r.width-320);
   cols.style.setProperty('--left',px+'px');};
 const up=()=>{grip.classList.remove('dragging');grip.removeEventListener('pointermove',move);grip.removeEventListener('pointerup',up);
   localStorage.setItem('seg.vlm.leftWidth',cols.style.getPropertyValue('--left'));};
 grip.addEventListener('pointermove',move);grip.addEventListener('pointerup',up);});
grip.addEventListener('dblclick',()=>{cols.style.setProperty('--left','42%');localStorage.removeItem('seg.vlm.leftWidth');});
function atBottom(el){return el.scrollHeight-el.scrollTop-el.clientHeight<80;}
let sideStick=true;
side.addEventListener('scroll',()=>{sideStick=atBottom(side);});
chat.addEventListener('scroll',()=>{chatStick=atBottom(chat);});
let chatStick=true;
function stickOf(el){return el===side?sideStick:chatStick;}
function toBottom(el){el.scrollTop=el.scrollHeight;}
// Follow the newest line unless the reader has scrolled up to read.
function put(el,node){const stick=stickOf(el);el.appendChild(node);if(stick)requestAnimationFrame(()=>toBottom(el));}
function say(cls,text){const d=document.createElement('div');d.className='msg '+cls;d.textContent=text;put(chat,d);return d;}
function ask(text,choices,live){const d=document.createElement('div');d.className='msg q';
  const t=document.createElement('b');t.textContent=T.check+' ';d.appendChild(t);d.appendChild(document.createTextNode(text));
  if(choices&&choices.length){const box=document.createElement('div');box.className='choices';
    choices.forEach(c=>{const b=document.createElement('button');b.type='button';b.textContent=c;
      if(!live)b.disabled=true;
      b.addEventListener('click',()=>{box.classList.add('done');b.classList.add('picked');
        [...box.children].forEach(x=>x.disabled=true);q.value=c;send();});
      box.appendChild(b);});
    d.appendChild(box);}
  put(chat,d);}
function shot(ev){side.querySelector('.empty')?.remove();const d=document.createElement('div');d.className='shot';
  const s=document.createElement('small');s.textContent=ev.caption;
  if(ev.jpeg_b64){const i=document.createElement('img');i.src='data:image/jpeg;base64,'+ev.jpeg_b64;
    i.addEventListener('load',()=>{if(sideStick)toBottom(side);});d.appendChild(i);}
  d.appendChild(s);put(side,d);}
function call(ev){side.querySelector('.empty')?.remove();const d=document.createElement('div');
  const bad=/"error"|"accepted": false|"written": false/.test(ev.result);d.className='call'+(bad?' bad':'');
  const t=document.createElement('span');t.className='t';t.textContent=ev.think_s+'s';
  const n=document.createElement('b');n.textContent=ev.name;
  const a=document.createElement('span');a.textContent=' '+JSON.stringify(ev.args).slice(0,110);
  const r=document.createElement('div');r.className='res';r.textContent=ev.result.slice(0,300);
  d.appendChild(t);d.appendChild(n);d.appendChild(a);d.appendChild(r);put(side,d);}
function busy(on){if(on){if(!working){working=document.createElement('div');working.className='working';working.innerHTML='<b>●</b> ';working.appendChild(document.createTextNode(T.thinking));put(chat,working);}}
  else if(working){working.remove();working=null;}}
function render(ev,live){
 if(ev.type==='me')say('me',ev.text);
 else if(ev.type==='tool'){call(ev);toolCount++;if(live)setStatus(fmt(T.toolStatus,{state:paused?T.paused:T.running,n:toolCount,name:ev.name}));}
 else if(ev.type==='context')setCtx(ev);
 else if(ev.type==='auto'){say('sys',ev.text);if(live)confirmBox.checked=false;}
 else if(ev.type==='trimmed')say('sys',ev.text);
 else if(ev.type==='paused'){say('paused',ev.text);if(live){paused=true;pauseBtn.textContent=T.resume;pauseBtn.classList.add('on');setStatus(T.paused);}}
 else if(ev.type==='resumed'){say('sys',ev.text);if(live){paused=false;pauseBtn.textContent=T.pause;pauseBtn.classList.remove('on');setStatus(T.running);}}
 else if(ev.type==='image')shot(ev);
 else if(ev.type==='question'){ask(ev.text,ev.choices,live);if(live){busy(false);pending=true;b.disabled=false;q.placeholder=T.pickOrReply;q.focus();}}
 else if(ev.type==='final'){if(live)busy(false);say('vlm',ev.text);}
 else if(ev.type==='stopped'){if(live)busy(false);say('sys',ev.text);}
 else {if(live)busy(false);say('err',ev.text||JSON.stringify(ev));}}
async function halt(reason){stopBtn.disabled=true;say('sys',reason);await fetch('/stop',{method:'POST',headers:JSONH,body:'{}'});}
stopBtn.addEventListener('click',()=>halt(T.stopping));
async function send(){const t=q.value.trim();if(!t)return;q.value='';say('me',t);
 if(pending){pending=false;q.placeholder=T.placeholder;busy(true);
   await fetch('/reply',{method:'POST',headers:JSONH,body:JSON.stringify({text:t,session:session})});return;}
 // Typing while it works is how a person says stop: the model reads nothing
 // mid-turn, so the run ends and this message becomes the next instruction.
 if(running)say('sys',T.interrupting);
 running=true;toolCount=0;b.disabled=true;stopBtn.disabled=false;pauseBtn.disabled=false;busy(true);setStatus(T.running);
 const r=await fetch('/chat',{method:'POST',headers:JSONH,body:JSON.stringify({text:t,confirm:confirmBox.checked,session:session})});
 const rd=r.body.getReader();const dec=new TextDecoder();let buf='';
 while(true){const {value,done}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});
  let i;while((i=buf.indexOf('\n'))>=0){const line=buf.slice(0,i);buf=buf.slice(i+1);if(!line)continue;const ev=JSON.parse(line);
   render(ev,true);}}
 busy(false);running=false;paused=false;b.disabled=false;stopBtn.disabled=true;
 pauseBtn.disabled=true;pauseBtn.textContent=T.pause;pauseBtn.classList.remove('on');
 setStatus(fmt(T.idleTools,{n:toolCount}));q.focus();
 if(queued){const nx=queued;queued=null;q.value=nx;send();}}
// A reload used to take the whole conversation with it: the page was the only
// place it existed. It is drawn back from the server, then the page asks what
// is still running before it draws its buttons.
(async()=>{
 try{const t=await (await fetch('/transcript?session='+encodeURIComponent(session))).json();
  const es=t.entries||[];
  if(es.length){es.forEach(ev=>render(ev,false));
   const d=document.createElement('div');d.className='msg sys';
   d.textContent=fmt(T.earlier,{n:es.length});put(chat,d);
   toBottom(chat);toBottom(side);}
 }catch(e){}
 try{const d=await (await fetch('/state?session='+encodeURIComponent(session))).json();
 running=!!d.running;paused=!!d.paused;pending=!!d.waiting_for_reply;
 stopBtn.disabled=!running;pauseBtn.disabled=!running;
 if(paused){pauseBtn.textContent=T.resume;pauseBtn.classList.add('on');}
 if(pending)q.placeholder=T.replyPlaceholder;
 if(running){setStatus(paused?T.paused:T.running);say('sys',T.stillRunning);}
 }catch(e){}})();
// Which model answers. Where each server is, and which environment variable
// holds its key, were fixed on the command line: the dialog shows them and
// sends only the server's name and the model's.
const dlg=document.getElementById('conndlg'),cBackend=document.getElementById('c-backend'),
 cUrl=document.getElementById('c-url'),cModel=document.getElementById('c-model'),
 cKey=document.getElementById('c-key'),cKind=document.getElementById('c-kind'),
 cList=document.getElementById('c-models'),cSaid=document.getElementById('conn-said'),
 hModel=document.getElementById('hmodel'),hWhere=document.getElementById('hwhere');
let servers=[];
function saidNow(text,bad){cSaid.textContent=text;cSaid.className=bad?'bad':'';}
function kindOf(name){const s=servers.find(x=>x.name===name);return s?s:null;}
function fillKind(){const s=kindOf(cBackend.value);
 cKind.textContent=s?(s.kind==='ollama'?T.kindOllama:T.kindOpenAI):'';}
function options(el,values){el.innerHTML='';values.forEach(v=>{const o=document.createElement('option');
 o.value=v;if(el.tagName==='SELECT')o.textContent=v;el.appendChild(o);});}
cBackend.addEventListener('change',()=>{const s=kindOf(cBackend.value);
 if(s){cUrl.value=s.url||s.default_url;cKey.value=s.api_key_env||'';}fillKind();saidNow('');});
document.getElementById('connbtn').addEventListener('click',async()=>{
 try{const d=await (await fetch('/connection')).json();
  servers=d.servers||[];
  options(cBackend,servers.map(s=>s.name));
  cBackend.value=d.current.backend;cUrl.value=d.current.base_url||'';
  cModel.value=d.current.model||'';cKey.value=d.current.api_key_env||'';
  fillKind();saidNow('');dlg.showModal();
 }catch(e){say('err',T.connUnread);}});
document.getElementById('c-close').addEventListener('click',()=>dlg.close());
function connBody(){return JSON.stringify({backend:cBackend.value,model:cModel.value.trim()});}
document.getElementById('c-test').addEventListener('click',async()=>{
 saidNow(T.checking);
 try{const r=await fetch('/connection/test',{method:'POST',headers:JSONH,body:connBody()});
  const d=await r.json();
  options(cList,(d.models||[]).map(String));
  saidNow((d.ok?'✓ ':'✕ ')+(d.detail||''),!d.ok);
 }catch(e){saidNow('✕ '+T.checkFailed+' '+e,true);}});
document.getElementById('c-save').addEventListener('click',async()=>{
 const r=await fetch('/connection',{method:'POST',headers:JSONH,body:connBody()});
 const d=await r.json();
 if(d.status!=='ok'){saidNow('✕ '+(d.detail||T.saveFailed),true);return;}
 hModel.textContent=d.current.model;hWhere.textContent=d.current.backend+' · '+d.current.base_url;
 dlg.close();say('sys',fmt(T.switched,{model:d.current.model,backend:d.current.backend,url:d.current.base_url}));});
f.addEventListener('submit',e=>{e.preventDefault();send();});
q.addEventListener('keydown',e=>{if(e.key==='Enter'&&(e.ctrlKey||e.metaKey)){e.preventDefault();send();}});
</script>"""

#: The page's words, by --lang. "html" fills the {{name}} slots of PAGE
#: (escaped); "js" becomes the T object the script reads.
TEXT: dict[str, dict[str, dict[str, str]]] = {
    "ja": {
        "html": {
            "title": "Seg-Studio × ローカル VLM",
            "h_model": "モデル:",
            "h_tail": "・ 書き込み可 ・ 例: チュートリアルのプロジェクトに追加した画像の黒い点をマーキングして。",
            "d_head": "接続するモデル",
            "d_server": "サーバー",
            "d_url": "場所 (URL)",
            "d_url_hint": "起動時の --base-url で決まります。画面からは変えられません",
            "d_model": "モデル",
            "d_model_hint": "接続テストを押すと、そのサーバーにあるモデルが候補に出ます",
            "d_key": "APIキーの環境変数名（要る場合だけ）",
            "d_key_hint": "起動時の --api-key-env で決まります。鍵そのものは画面に出ません",
            "d_test": "接続テスト",
            "d_save": "保存して使う",
            "d_close": "閉じる",
            "col_chat": "会話",
            "t_pause": "実行を止めずに一時的に休ませます。再開すると続きから進みます",
            "b_pause": "一時停止",
            "t_new": "まっさらな会話を始めます。今の会話はこのタブから離れます",
            "b_new": "新しい会話",
            "t_reset": "この会話の記憶を消します。画像やマスクには触れません",
            "b_reset": "リセット",
            "t_conn": "どのモデルに答えてもらうかを選びます",
            "b_conn": "接続",
            "t_confirm": "外すと、書き込む前の確認を省いて最後まで走ります",
            "b_confirm": "毎回確認",
            "s_idle": "待機中",
            "t_ctx": "この会話が使っている文脈の量。上限に達すると古いやり取りを整理します",
            "p_instruction": "指示を書いて送信（Ctrl+Enter）",
            "b_send": "送信",
            "b_stop": "中止",
            "t_grip": "ドラッグで幅を変えられます",
            "col_side": "画像とツール呼び出し",
            "side_empty": "指示を送ると、モデルが見た画像と呼んだツールがここに出ます。",
        },
        "js": {
            "ctx": "文脈",
            "sideEmpty": "指示を送ると、モデルが見た画像と呼んだツールがここに出ます。",
            "pause": "一時停止",
            "resume": "再開",
            "paused": "一時停止中",
            "running": "実行中",
            "idle": "待機中",
            "busyNew": "実行中は始められません。中止してからどうぞ。",
            "newStarted": "新しい会話を始めました。前の会話は保存してあります。",
            "busyReset": "実行中はリセットできません。先に中止してください。",
            "resetDone": "会話をリセットしました（画像とマスクはそのままです）",
            "check": "確認",
            "thinking": "考えています…",
            "toolStatus": "{state} ・ ツール {n} 回 ・ 直近 {name}",
            "pickOrReply": "選ぶか、返事を書いて送信",
            "stopping": "中止しています…",
            "placeholder": "指示を書いて送信（Ctrl+Enter）",
            "interrupting": "実行中の指示を中止して、これを実行します。",
            "idleTools": "待機中 ・ ツール {n} 回",
            "earlier": "——— ここまでが前回までの記録（{n} 件）———",
            "replyPlaceholder": "返事を書いて送信（はい / 直してほしいこと）",
            "stillRunning": "実行中の指示があります。何か送ると中止して差し替えます。",
            "kindOllama": "Ollama の /api/chat",
            "kindOpenAI": "/v1/chat/completions (MLX・vLLM・LM Studio・llama.cpp)",
            "connUnread": "接続設定を読めませんでした",
            "checking": "確かめています…",
            "checkFailed": "確認できませんでした:",
            "saveFailed": "保存できませんでした",
            "switched": "接続先を変えました：{model}（{backend} · {url}）",
        },
    },
    "en": {
        "html": {
            "title": "Seg-Studio × local VLM",
            "h_model": "Model:",
            "h_tail": "· can write masks · e.g. In the tutorial project, mark the dark specks on the image I added.",
            "d_head": "Model to talk to",
            "d_server": "Server",
            "d_url": "Location (URL)",
            "d_url_hint": "Set by --base-url when this server starts; the page cannot change it",
            "d_model": "Model",
            "d_model_hint": "Test the connection to list the models that server holds",
            "d_key": "Environment variable holding the API key (only if one is needed)",
            "d_key_hint": "Set by --api-key-env when this server starts; the key itself never reaches the page",
            "d_test": "Test connection",
            "d_save": "Save and use",
            "d_close": "Close",
            "col_chat": "Conversation",
            "t_pause": "Hold the run without ending it; resuming carries on where it left off",
            "b_pause": "Pause",
            "t_new": "Start a fresh conversation; the current one leaves this tab",
            "b_new": "New conversation",
            "t_reset": "Forget this conversation; images and masks are not touched",
            "b_reset": "Reset",
            "t_conn": "Choose which model answers",
            "b_conn": "Connection",
            "t_confirm": "Untick to run to the end without asking before each write",
            "b_confirm": "Ask each time",
            "s_idle": "Idle",
            "t_ctx": "How much context this conversation uses; at the limit, older exchanges are trimmed",
            "p_instruction": "Type an instruction and send (Ctrl+Enter)",
            "b_send": "Send",
            "b_stop": "Stop",
            "t_grip": "Drag to change the width",
            "col_side": "Images and tool calls",
            "side_empty": "Send an instruction: the images the model looks at and the tools it calls appear here.",
        },
        "js": {
            "ctx": "Context",
            "sideEmpty": "Send an instruction: the images the model looks at and the tools it calls appear here.",
            "pause": "Pause",
            "resume": "Resume",
            "paused": "Paused",
            "running": "Running",
            "idle": "Idle",
            "busyNew": "A run is in progress. Stop it first.",
            "newStarted": "Started a new conversation. The previous one is kept.",
            "busyReset": "A run is in progress, so it cannot be reset. Stop it first.",
            "resetDone": "Conversation reset (images and masks are unchanged)",
            "check": "Check",
            "thinking": "Thinking…",
            "toolStatus": "{state} · {n} tool calls · last {name}",
            "pickOrReply": "Pick one, or type a reply and send",
            "stopping": "Stopping…",
            "placeholder": "Type an instruction and send (Ctrl+Enter)",
            "interrupting": "Stopping the running instruction to run this one.",
            "idleTools": "Idle · {n} tool calls",
            "earlier": "——— the conversation so far ({n} entries) ———",
            "replyPlaceholder": "Type a reply and send (yes / what to fix)",
            "stillRunning": "An instruction is running. Sending anything stops it and runs yours instead.",
            "kindOllama": "Ollama's /api/chat",
            "kindOpenAI": "/v1/chat/completions (MLX, vLLM, LM Studio, llama.cpp)",
            "connUnread": "Could not read the connection settings",
            "checking": "Checking…",
            "checkFailed": "Could not check:",
            "saveFailed": "Could not save",
            "switched": "Now talking to {model} ({backend} · {url})",
        },
    },
}

#: What the server itself says back, by --lang.
SAY: dict[str, dict[str, str]] = {
    "ja": {
        "interrupt": "前の指示を中止して、新しい指示を実行します",
        "busy_switch": "実行中は切り替えられません。中止してからどうぞ。",
        "busy_reset": "実行中はリセットできません。先に中止してください。",
        "unknown_server": "知らない接続先です: {name}",
        "need_model": "モデル名を入れてください",
        "key_empty": "環境変数 {name} が空です。",
        "cli_only": ("接続先の URL と API キーの環境変数名は、起動時の --base-url と --api-key-env "
                     "で決めます。画面からは変えられません。"),
        "unreachable": "つながりません: {reason}",
        "models_seen": "{n} 個のモデルが見えています（{ms} ms）",
        "model_missing": "つながりましたが {model} が見当たりません（{n} 個）",
        "wrong_host": "このページはこのマシンの {host} でだけ開けます。",
        "wrong_origin": "このページ以外からの書き込み要求は受け付けません。",
        "not_json": "要求は application/json で送ってください。",
        "bad_body": "要求の本文が JSON ではありません。",
    },
    "en": {
        "interrupt": "Stopped the previous instruction to run the new one",
        "busy_switch": "A run is in progress, so the model cannot be switched. Stop it first.",
        "busy_reset": "A run is in progress, so it cannot be reset. Stop it first.",
        "unknown_server": "Unknown model server: {name}",
        "need_model": "Enter a model name",
        "key_empty": "The environment variable {name} is empty. ",
        "cli_only": ("The server's address and the environment variable holding its key are set on the "
                     "command line (--base-url, --api-key-env), not from the page."),
        "unreachable": "Cannot connect: {reason}",
        "models_seen": "{n} models visible ({ms} ms)",
        "model_missing": "Connected, but {model} is not among its {n} models",
        "wrong_host": "This page is served on this machine only, at {host}.",
        "wrong_origin": "Requests that change anything are taken from this page only.",
        "not_json": "Send the request as application/json.",
        "bad_body": "The request body is not JSON.",
    },
}


def _words(lang: str, key: str, **values) -> str:
    return SAY.get(lang, SAY["en"])[key].format(**values)


def _page(lang: str) -> str:
    """PAGE with its words filled in, for one language."""
    text = TEXT[lang]
    script = json.dumps(text["js"], ensure_ascii=False).replace("</", "<\\/")

    def fill(m: re.Match) -> str:
        name = m.group(1)
        if name == "lang":
            return lang
        if name == "js":
            return script
        return html.escape(text["html"][name])

    return re.sub(r"\{\{(\w+)\}\}", fill, PAGE)


#: Addresses this example may listen on, and the names a browser uses for
#: each. The page has no login, so nothing else is offered.
LOOPBACK: dict[str, frozenset[str]] = {
    "127.0.0.1": frozenset({"127.0.0.1", "localhost"}),
    "localhost": frozenset({"127.0.0.1", "localhost", "::1"}),
    "::1": frozenset({"::1", "localhost"}),
}


def _hostname(host: str) -> str:
    """The name in a Host header: '127.0.0.1:8765' -> '127.0.0.1', '[::1]:8765' -> '::1'."""
    host = host.strip().lower()
    if host.startswith("["):
        end = host.find("]")
        return host[1:end] if end > 0 else ""
    return host.split(":", 1)[0]


def refusal(method: str, headers, allowed: frozenset[str], lang: str = "en") -> tuple[int, str] | None:
    """Why a request is turned away, or None to let it through.

    Host first, for every request: a page elsewhere whose name has been
    pointed at 127.0.0.1 arrives carrying its own name, and would otherwise
    read the conversation and its pictures. Then, for anything that is not a
    read, the origin and the content type. A browser sends Origin with every
    POST, so a POST from another page's origin is refused outright; one with
    no Origin at all comes from a program on this machine, not from a page.
    JSON is required because a page elsewhere can send text/plain without
    asking first, and cannot send JSON without asking -- a question this
    server never answers.
    """
    host = str(headers.get("host") or "")
    if _hostname(host) not in allowed:
        return 403, _words(lang, "wrong_host", host=" / ".join(sorted(allowed)))
    if method.upper() in ("GET", "HEAD"):
        return None
    origin = headers.get("origin")
    if origin is not None and origin.strip().lower().rstrip("/") != f"http://{host.strip().lower()}":
        return 403, _words(lang, "wrong_origin")
    ctype = str(headers.get("content-type") or "").split(";", 1)[0].strip().lower()
    if ctype != "application/json":
        return 415, _words(lang, "not_json")
    return None


class SameMachineOnly:
    """Apply refusal() to every request before any route sees it."""

    def __init__(self, app, allowed: frozenset[str], lang: str) -> None:
        self.app, self.allowed, self.lang = app, allowed, lang

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
            why = refusal(scope["method"], headers, self.allowed, self.lang)
            if why is not None:
                response = JSONResponse({"status": "refused", "detail": why[1]}, status_code=why[0])
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


async def _body(request: Request, lang: str) -> dict:
    """The request's JSON object; an empty body is an empty one."""
    raw = await request.body()
    if not raw:
        return {}
    try:
        body = json.loads(raw)
    except ValueError:
        raise HTTPException(status_code=400, detail=_words(lang, "bad_body")) from None
    return body if isinstance(body, dict) else {}


def _names_an_address(body: dict) -> bool:
    """A request that tries to say where the model server is, or which key to send."""
    return any(body.get(k) not in (None, "") for k in ("base_url", "api_key_env"))


def _probe(bk, lang: str) -> dict:
    """What bk.probe() reports, in the page's language: a listing, never a generation."""
    t0 = time.time()
    try:
        models = bk.list_models()
    except urllib.error.HTTPError as exc:
        return {"ok": False, "detail": f"HTTP {exc.code} {exc.reason}", "models": []}
    except urllib.error.URLError as exc:
        return {"ok": False, "detail": _words(lang, "unreachable", reason=exc.reason), "models": []}
    except (OSError, ValueError) as exc:
        return {"ok": False, "detail": f"{type(exc).__name__}: {exc}", "models": []}
    ms = int((time.time() - t0) * 1000)
    known = bk.model in models
    detail = (_words(lang, "models_seen", n=len(models), ms=ms) if known or not models
              else _words(lang, "model_missing", model=bk.model, n=len(models)))
    return {"ok": True, "models": models, "ms": ms, "has_model": known, "detail": detail}


MAX_ENTRIES = 600         # what a page replays after a reload
MAX_IMAGES = 30            # of those, how many keep their picture
MAX_SESSIONS = 20


def state_dir() -> Path:
    """Where conversations are kept between runs of this server."""
    env = os.environ.get("SEG_VLM_CHAT_STATE")
    path = Path(env) if env else Path.home() / ".seg-studio" / "vlm-chat"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _trim_entries(entries: list) -> list:
    """Keep the conversation, drop the weight: old pictures go, their captions stay."""
    entries = entries[-MAX_ENTRIES:]
    seen = 0
    for ev in reversed(entries):
        if ev.get("type") == "image" and ev.get("jpeg_b64"):
            seen += 1
            if seen > MAX_IMAGES:
                ev.pop("jpeg_b64", None)
    return entries


class Conversations:
    """The chat as the page draws it, and as the model remembers it.

    Both are kept on disk, one file per session. A reload used to take the
    whole conversation with it -- the page was the only place it existed --
    and restarting the server lost what the model knew as well.
    """

    def __init__(self) -> None:
        self._mem: dict[str, dict] = {}

    def _path(self, sid: str) -> Path:
        safe = "".join(c for c in sid if c.isalnum() or c in "-_")[:64] or "default"
        return state_dir() / f"{safe}.json"

    def get(self, sid: str) -> dict:
        if sid not in self._mem:
            data = {"history": [], "entries": []}
            try:
                raw = json.loads(self._path(sid).read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    data = {"history": raw.get("history") or [], "entries": raw.get("entries") or []}
            except (OSError, ValueError):
                pass
            if len(self._mem) >= MAX_SESSIONS:
                self._mem.pop(next(iter(self._mem)))
            self._mem[sid] = data
        return self._mem[sid]

    def history(self, sid: str) -> list:
        return self.get(sid)["history"]

    def entries(self, sid: str) -> list:
        return self.get(sid)["entries"]

    def add(self, sid: str, ev: dict) -> None:
        if ev.get("type") in ("context",):        # a gauge, not a line of the conversation
            return
        self.entries(sid).append(ev)

    def save(self, sid: str) -> None:
        data = self.get(sid)
        data["entries"] = _trim_entries(data["entries"])
        tmp = self._path(sid).with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._path(sid))
        except OSError:
            pass

    def reset(self, sid: str) -> None:
        self._mem[sid] = {"history": [], "entries": []}
        try:
            self._path(sid).unlink(missing_ok=True)
        except OSError:
            pass


#: What the page may choose, and so all that is remembered of its choice.
REMEMBERED = ("backend", "model")


def load_connection(default: dict, asked: dict | None = None) -> dict:
    """Which server and which model: what was asked for, else what was chosen last time.

    The choice is kept beside the conversations, so a person who picked a model
    on the page does not have to pick it again at every restart. An argument
    actually passed still wins -- a remembered setting is a default, not an
    override. Only the server's name and the model are read back: where the
    server is and which key it gets come from the command line alone, so an
    address an earlier version of this page saved is not used.
    """
    out = dict(default)
    try:
        saved = json.loads((state_dir() / "connection.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {}
    if isinstance(saved, dict):
        for key in REMEMBERED:
            if isinstance(saved.get(key), str) and saved[key]:
                out[key] = saved[key]
    for key, value in (asked or {}).items():
        if value is not None:
            out[key] = value
    if out.get("backend") not in ALIASES:
        out["backend"] = default["backend"]
    return out


def save_connection(conn: dict) -> None:
    """Remember the page's choice: the server's name and the model, nothing else."""
    path = state_dir() / "connection.json"
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps({k: conn.get(k) for k in REMEMBERED}, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass


def build(model: str | None = None, ollama: str | None = None, api: str = "http://127.0.0.1:8002",
          lang: str = "ja", backend: str | None = None, base_url: str | None = None,
          api_key_env: str | None = None, host: str = "127.0.0.1") -> FastAPI:
    """The chat app. base_url and api_key_env are the command line's; no request changes them.

    They belong to one server -- --backend, or the one in use at start when
    that is not given -- and go with it: the page may switch to another known
    server, which is then reached at its default address with no key, and
    switching back brings them back.
    """
    if host not in LOOPBACK:
        raise ValueError(f"--host {host!r}: this page has no login, so it listens on this machine only "
                         f"({', '.join(LOOPBACK)}); reach it from elsewhere through an SSH tunnel")
    if backend is not None and backend not in ALIASES:
        raise ValueError(f"unknown model server {backend!r}; try one of: {', '.join(ALIASES)}")
    lang = lang if lang in TEXT else "en"
    page = _page(lang)
    app = FastAPI(title="seg-studio local VLM chat")
    app.add_middleware(SameMachineOnly, allowed=LOOPBACK[host], lang=lang)
    # One model answers one request at a time, so the run lock is global; the
    # conversation is not. A tab keeps its own, which is what "a new session"
    # means here: a fresh history against the same project and the same model.
    lock = asyncio.Lock()
    talk = Conversations()
    if ollama and base_url is None and backend in (None, "ollama"):
        backend, base_url = "ollama", ollama       # the old argument, still honoured
    conn = load_connection({"backend": "ollama", "model": "qwen3.8:27b"},
                           {"backend": backend, "model": model})
    home = backend or conn["backend"]              # the server the command line describes

    def where(name: str) -> tuple[str, str | None]:
        """The address and key-variable name for a server, from the command line only."""
        if name == home:
            return (base_url or ALIASES[name][1]).strip().rstrip("/"), api_key_env or None
        return ALIASES[name][1], None

    conn["base_url"], conn["api_key_env"] = where(conn["backend"])
    waiting: dict = {"future": None}
    # A running turn cannot hear anything: the model reads messages only
    # between turns. Stopping is a flag the loop checks, not a request.
    stopping: dict = {"flag": False}
    # Held between steps, not inside one: a model turn cannot be interrupted,
    # so a pause takes effect within one turn and loses nothing.
    paused: dict = {"flag": False}

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        slots = {"MODEL": html.escape(conn["model"]),
                 "WHERE": html.escape(f'{conn["backend"]} · {conn["base_url"]}')}
        return re.sub(r"__(MODEL|WHERE)__", lambda m: slots[m.group(1)], page)

    @app.post("/chat")
    async def chat(request: Request):
        body = await _body(request, lang)
        text = str(body.get("text") or "").strip()
        sid = str(body.get("session") or "default")[:64]
        history = talk.history(sid)
        talk.add(sid, {"type": "me", "text": text})
        talk.save(sid)
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        stopping["flag"] = False
        paused["flag"] = False
        confirm = bool(body.get("confirm", True))

        def on_event(ev):
            loop.call_soon_threadsafe(queue.put_nowait, ev)

        async def ask(question: str) -> str:
            fut = loop.create_future()
            waiting["future"] = fut
            try:
                return await fut
            finally:
                waiting["future"] = None

        async def job():
            # An instruction arriving while a run is in flight is a person
            # interrupting, and the page cannot always know one is: a reloaded
            # tab believes nothing is running, and was then told "a job is
            # already running" -- true, useless, and with no way out. The run
            # is ended here and the new instruction takes its place.
            if lock.locked():
                queue.put_nowait({"type": "stopped", "text": _words(lang, "interrupt")})
                stopping["flag"] = True
                fut = waiting.get("future")
                if fut is not None and not fut.done():
                    fut.set_result("stop")
                for _ in range(600):          # one turn in flight, at most
                    if not lock.locked():
                        break
                    await asyncio.sleep(0.1)
                stopping["flag"] = False
            if True:
                async with lock:
                    try:
                        await run(text, model=conn["model"], api=api, on_event=on_event,
                                  history=history, ask=ask if confirm else None, lang=lang,
                                  should_stop=lambda: stopping["flag"],
                                  should_pause=lambda: paused["flag"],
                                  backend=conn["backend"], base_url=conn["base_url"],
                                  api_key_env=conn["api_key_env"])
                    except Exception as exc:  # the page must say so, not hang
                        queue.put_nowait({"type": "error", "text": f"{type(exc).__name__}: {exc}"})
            queue.put_nowait(None)

        asyncio.create_task(job())

        async def stream():
            # Everything the page draws is written down as it goes, so a reload
            # -- or a restart of this server -- picks the conversation back up.
            while True:
                ev = await queue.get()
                if ev is None:
                    talk.save(sid)
                    break
                talk.add(sid, ev)
                if ev.get("type") in ("final", "question", "stopped", "error"):
                    talk.save(sid)
                yield json.dumps(ev, ensure_ascii=False) + "\n"

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    @app.post("/stop")
    async def stop():
        """End the running job at the next turn or tool call."""
        stopping["flag"] = True
        fut = waiting.get("future")
        if fut is not None and not fut.done():
            fut.set_result("stop")   # a pending question must not hold the loop
        return {"status": "ok"}

    @app.post("/pause")
    async def pause(request: Request):
        """Hold the run between steps, or let it go again."""
        body = await _body(request, lang)
        paused["flag"] = bool(body.get("paused", not paused["flag"]))
        return {"status": "ok", "paused": paused["flag"]}

    @app.get("/state")
    async def state(session: str = "default"):
        """What the page needs to draw its buttons after a reload."""
        return {"running": lock.locked(), "paused": paused["flag"],
                "waiting_for_reply": waiting.get("future") is not None,
                "turns": len(talk.history(session[:64])),
                "sessions": len(list(state_dir().glob("*.json")))}

    @app.get("/connection")
    async def connection():
        """What we are talking to, and what else could be chosen -- and where each one is."""
        servers = []
        for n, (k, u) in ALIASES.items():
            url, key_env = where(n)
            servers.append({"name": n, "kind": k, "default_url": u, "url": url, "api_key_env": key_env})
        return {"current": conn, "servers": servers}

    @app.post("/connection")
    async def set_connection(request: Request):
        """Switch to another known server or model. Where it is, and its key, stay the command line's."""
        body = await _body(request, lang)
        if _names_an_address(body):
            return JSONResponse({"status": "error", "detail": _words(lang, "cli_only")}, status_code=400)
        if lock.locked():
            return {"status": "busy", "detail": _words(lang, "busy_switch")}
        name = str(body.get("backend") or conn["backend"])
        if name not in ALIASES:
            return {"status": "error", "detail": _words(lang, "unknown_server", name=name)}
        model_name = str(body.get("model") or "").strip()
        if not model_name:
            return {"status": "error", "detail": _words(lang, "need_model")}
        url, key_env = where(name)
        conn.update({"backend": name, "base_url": url, "model": model_name, "api_key_env": key_env})
        save_connection(conn)
        return {"status": "ok", "current": conn}

    @app.post("/connection/test")
    async def test_connection(request: Request):
        """Reach the server and list what it holds. Nothing is generated."""
        body = await _body(request, lang)
        if _names_an_address(body):
            return JSONResponse({"ok": False, "detail": _words(lang, "cli_only"), "models": []}, status_code=400)
        name = str(body.get("backend") or conn["backend"])
        if name not in ALIASES:
            return {"ok": False, "detail": _words(lang, "unknown_server", name=name), "models": []}
        url, key_env = where(name)
        try:
            bk = make_backend(name, model=str(body.get("model") or conn["model"]),
                              base_url=url, api_key_env=key_env)
        except ValueError as exc:
            return {"ok": False, "detail": str(exc), "models": []}
        result = await asyncio.to_thread(_probe, bk, lang)
        if key_env and not bk.api_key:
            result["detail"] = _words(lang, "key_empty", name=key_env) + result.get("detail", "")
        return result

    @app.get("/transcript")
    async def transcript(session: str = "default"):
        """The conversation as it was drawn, for a page that has just loaded."""
        return {"entries": talk.entries(session[:64])}

    @app.post("/reply")
    async def reply(request: Request):
        body = await _body(request, lang)
        fut = waiting.get("future")
        if fut is None or fut.done():
            return {"status": "no question pending"}  # the page only posts here after a question
        answer = str(body.get("text") or "")
        sid = str(body.get("session") or "default")[:64]
        talk.add(sid, {"type": "me", "text": answer})
        talk.save(sid)
        fut.set_result(answer)
        return {"status": "ok"}

    @app.post("/reset")
    async def reset(request: Request):
        """Forget this session's conversation. The project's masks are untouched."""
        if lock.locked():
            return {"status": "busy", "detail": _words(lang, "busy_reset")}
        body = await _body(request, lang)
        sid = str(body.get("session") or "default")[:64]
        talk.reset(sid)
        return {"status": "ok", "session": sid}

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", choices=list(LOOPBACK),
                    help="a loopback address: the page has no login; use an SSH tunnel to reach it from elsewhere")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", default=None, help="overrides what was last chosen on the page")
    ap.add_argument("--backend", default=None, choices=sorted(ALIASES),
                    help="which model server: " + describe())
    ap.add_argument("--base-url", default=None,
                    help="where it is, if not that server's default; the page cannot change it")
    ap.add_argument("--api-key-env", default=None,
                    help="name of the environment variable holding the key, for a server that wants one; "
                         "sent only to the --backend server (without --backend, the one in use at start)")
    ap.add_argument("--ollama", default=None, help="shorthand for --backend ollama --base-url")
    ap.add_argument("--api", default="http://127.0.0.1:8002")
    ap.add_argument("--lang", default="ja", choices=("ja", "en"),
                    help="the language of the page and of the model's replies")
    args = ap.parse_args()
    import uvicorn
    app = build(args.model, args.ollama, args.api, args.lang,
                backend=args.backend, base_url=args.base_url, api_key_env=args.api_key_env,
                host=args.host)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
