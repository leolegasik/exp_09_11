import json
from experiment import ROOT, LABELS, SEED, corpus, group_docs, load_json, predictions, load_annotations, protocol_hash

STYLE = '''
:root{color-scheme:light;font:17px/1.55 system-ui,sans-serif;color:#152d39;background:#edf3f5}
body{max-width:1040px;margin:28px auto;padding:0 22px 60px}h1{font-size:30px;line-height:1.2}h2{font-size:22px}
.card,header{background:white;padding:24px;margin:20px 0;border:1px solid #ccdadd;border-radius:12px}
.text{font-size:19px;border-left:4px solid #21857f;padding:12px 18px;background:#f1f8f7;white-space:pre-wrap}
label{display:block;margin:14px 0;cursor:pointer}.hint{font-size:14px;color:#526871}
input[type=checkbox]{width:19px;height:19px;vertical-align:middle;margin-right:8px}
input[type=text],textarea,select{box-sizing:border-box;width:100%;padding:10px;border:1px solid #9fb5bd;border-radius:6px;font:inherit;background:white}
.triple{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin:12px 0}.triple input{font-size:15px}
button{background:#116c68;color:white;padding:13px 19px;font:inherit;border:0;border-radius:7px;cursor:pointer;margin:6px}
button.secondary{background:#516a76}.toolbar{position:sticky;top:0;background:#e4eeeeee;padding:10px;border-radius:10px;z-index:2}
.warning{background:#fff5d8;padding:12px;border-radius:6px}.status{font-weight:600}.edge{padding:12px;background:#f1f8f7;margin:9px 0;border-radius:7px}
.pill{display:inline-block;background:#deeeec;color:#125b56;border-radius:20px;padding:4px 12px;margin:3px}
svg{width:100%;height:auto;display:block}.quote{font-size:14px;color:#536970}details{margin:12px 0}a{color:#126d79}
@media(max-width:650px){.triple{grid-template-columns:1fr}.card{padding:15px}body{padding:0 10px}}
@media print{.toolbar,button{display:none}.card{break-inside:avoid}body{background:white}}
'''

def js_json(obj):
    return json.dumps(obj, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')

def shell(title, body, script):
    return f'<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>{STYLE}</style><body>{body}<script>{script}</script></body></html>'

BLIND_JS = r'''
const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const key='para0911_'+meta.corpus_sha256+'_'+meta.group;
let saved={};try{saved=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
function message(text){document.querySelector('#notice').textContent=text;}
function render(){
 document.querySelector('#cards').innerHTML=docs.map((d,i)=>`<section class="card" data-id="${d.id}">
 <h2>Текст ${i+1} из ${docs.length} <span class="hint">${d.id}</span></h2><p class="text">${esc(d.text)}</p>
 <p>Какие смыслы вы видите? Укажите 1–3 главные связи своими словами. Каждое поле связи должно быть заполнено.</p>
 ${[0,1,2].map(j=>`<div class="triple">${['Кто / что','Что делает / как связано','С кем / с чем'].map((p,k)=>`<input type="text" data-triple="${j}" data-col="${k}" aria-label="${p}, связь ${j+1}" placeholder="${j+1}. ${p}" value="${esc(saved[d.id]?.triples?.[j]?.[k]||'')}">`).join('')}</div>`).join('')}
 <p><b>Отметьте все признаки, которые прямо выражены в тексте:</b></p>
 ${Object.entries(labels).map(([k,v])=>`<label><input type="checkbox" data-label="${k}" ${saved[d.id]?.labels?.includes(k)?'checked':''}><b>${esc(v[0])}</b><br><span class="hint">${esc(v[1])}</span></label>`).join('')}
 <label>Комментарий / сомнение (необязательно)<textarea data-comment rows="2">${esc(saved[d.id]?.comment||'')}</textarea></label>
 <label><input type="checkbox" data-complete ${saved[d.id]?.complete?'checked':''}>Я закончил этот текст; неотмеченные признаки считаю отсутствующими.</label></section>`).join('');
 document.querySelector('#cards').addEventListener('input',save); update();
}
function collect(){return [...document.querySelectorAll('[data-id]')].map(s=>{
 const triples=[0,1,2].map(j=>[0,1,2].map(k=>s.querySelector(`[data-triple="${j}"][data-col="${k}"]`).value.trim())).filter(t=>t.some(Boolean));
 return {id:s.dataset.id,triples,labels:[...s.querySelectorAll('[data-label]:checked')].map(x=>x.dataset.label),comment:s.querySelector('[data-comment]').value.trim(),complete:s.querySelector('[data-complete]').checked};});}
function update(){const answers=collect();document.querySelector('#status').textContent=`Завершено ${answers.filter(a=>a.complete).length}/${docs.length}`;}
function save(){saved=Object.fromEntries(collect().map(a=>[a.id,a]));try{localStorage.setItem(key,JSON.stringify(saved));document.querySelector('#storage').textContent='Черновик сохраняется в этом браузере.'}catch(e){document.querySelector('#storage').textContent='Браузер не разрешил автосохранение. Не закрывайте страницу до экспорта JSON.'}update();}
function download(obj,name){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(obj,null,2)],{type:'application/json'}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
document.querySelector('#export').onclick=()=>{
 const answers=collect();for(let i=0;i<answers.length;i++){
 const a=answers[i];if(!a.complete||a.triples.length<1||a.triples.some(t=>t.some(x=>!x))){message(`Проверьте текст ${i+1}: завершите ответ и заполните 1–3 связи целиком.`);return;}}
 download({...meta,annotator:'participant_1',exported_at:new Date().toISOString(),answers},`human_${meta.group}.json`);
};
document.querySelector('#draft').onclick=()=>download({...meta,draft:true,answers:collect()},`draft_${meta.group}.json`);
document.querySelector('#restore').onchange=async e=>{try{const x=JSON.parse(await e.target.files[0].text());if(x.group!==meta.group||x.corpus_sha256!==meta.corpus_sha256)throw Error('Другой корпус или группа');saved=Object.fromEntries(x.answers.map(a=>[a.id,a]));render();save();}catch(err){message(err.message)}};
render();
'''

def blind_form(group):
    docs = group_docs(group)
    # Порядок стабилен и не зависит от ответов модели.
    import random
    random.Random(SEED+10+(group == 'audit')).shuffle(docs)
    docs = [{k:d[k] for k in ['id', 'text']} for d in docs]  # Никаких предсказаний в исходном HTML.
    title = 'Слепая разметка: 10 текстов' if group == 'pilot' else 'Независимая проверка: 20 новых текстов'
    body = f'''<header><p class="hint">Эксперимент · один участник · {group}</p><h1>{title}</h1>
    <p>Читайте только приведённый текст. Не ищите статью и не спрашивайте ИИ. Собственных знаний для добавления фактов использовать не нужно.</p>
    <p>Сначала запишите 1–3 главные связи: например, «Исследователи → обнаружили → новый вид». Затем отметьте смысловые признаки. Можно выбрать несколько или ни одного.</p>
    <p>Сохраняйте отрицание, сомнения и авторство мнения: «министр считает» не равнозначно установленному факту. Здесь нет единственно правильной формулировки.</p>
    <p class="warning">До экспорта ответов не открывайте результаты модели, каталог results и демонстрацию. Это нужно для независимости ваших суждений.</p>
    <p>После завершения нажмите «Экспорт ответов» и сохраните JSON в <b>annotation/inbox</b>. Если браузер сохранил файл в «Загрузки», перенесите его в эту папку.</p></header>
    <div class="toolbar"><span id="status" class="status"></span><button id="export">Экспорт ответов</button><button id="draft" class="secondary">Скачать черновик</button><p id="storage" class="hint">Черновик сохраняется в этом браузере.</p><p id="notice" role="alert" aria-live="polite"></p></div>
    <details><summary>Восстановить черновик из JSON</summary><input type="file" id="restore" accept=".json"></details><main id="cards"></main>'''
    script = f'const docs={js_json(docs)},labels={js_json(LABELS)},meta={js_json({"group":group,"corpus_sha256":load_json(ROOT/"data/manifest.json")["corpus_file_sha256"],"protocol_version":"blind_v1"})};\n'+BLIND_JS
    dest = ROOT / ('annotation/01_pilot_blind.html' if group == 'pilot' else 'annotation/02_audit_blind.html')
    dest.write_text(shell(title, body, script), encoding='utf-8')

DEMO_JS = r'''
const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let allowed=false;
document.querySelector('#unlock').onchange=()=>{allowed=document.querySelector('#unlock').checked;document.querySelector('#viewer').hidden=!allowed;if(allowed)show()};
document.querySelector('#pick').innerHTML=docs.map((d,i)=>`<option value="${i}">Текст ${i+1} · ${d.id}</option>`).join('');
document.querySelector('#pick').onchange=show;
function show(){if(!allowed)return;const d=docs[+document.querySelector('#pick').value];const p=pred[d.id]?.prediction;
 document.querySelector('#text').textContent=d.text;
 document.querySelector('#source').innerHTML=`Источник: <a href="${esc(d.url)}" target="_blank" rel="noopener">Газета.ru</a>, ${esc(d.date)}. Анализируется поле summary.`;
 if(!p){document.querySelector('#graph').textContent='Модель пока не обработала этот текст.';return;}
 const triples=p.triples;
 document.querySelector('#graph').innerHTML=triples.length?`<svg role="img" aria-label="Смысловые связи" viewBox="0 0 1000 ${triples.length*150}"><defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#197c77"/></marker></defs>${triples.map((t,i)=>{
 const y=i*150+10;const node=(txt,x)=>`<rect x="${x}" y="${y+5}" width="290" height="105" rx="12" fill="#e4f3f0" stroke="#8dbfb6"/><foreignObject x="${x+12}" y="${y+12}" width="266" height="90"><div xmlns="http://www.w3.org/1999/xhtml" style="font:17px system-ui;overflow-wrap:anywhere;line-height:1.2">${esc(txt)}</div></foreignObject>`;
 return node(t[0],5)+node(t[2],705)+`<line x1="305" y1="${y+75}" x2="695" y2="${y+75}" stroke="#197c77" stroke-width="3" marker-end="url(#arrow)"/><foreignObject x="320" y="${y+5}" width="365" height="60"><div xmlns="http://www.w3.org/1999/xhtml" style="font:17px system-ui;text-align:center;line-height:1.2;overflow-wrap:anywhere">${esc(t[1])}</div></foreignObject>`;}).join('')}</svg>`:'Нет выделенных связей';
 document.querySelector('#edges').innerHTML=triples.map((t,i)=>`<div class="edge"><b>${i+1}. ${esc(t[0])} → ${esc(t[1])} → ${esc(t[2])}</b><p class="quote">Цитата: «${esc(t[3])}»</p></div>`).join('');
 document.querySelector('#labels').innerHTML=p.labels.map(k=>`<span class="pill">${esc(labels[k][0])}</span>`).join('')||'Ни один из шести признаков не выделен.';
}
'''

def demo():
    docs = group_docs('pilot')
    pred = predictions()
    body = '''<header><p class="hint">Смысл текста · Qwen3.5-9B · локальный эксперимент</p><h1>OpenIE: смысловые связи в тексте</h1><p>Граф читается слева направо: кто или что → отношение → с кем или с чем. Точная цитата позволяет проверить основание связи. Цитата сама по себе не доказывает корректность интерпретации.</p>
    <label class="warning"><input type="checkbox" id="unlock">Показать смысловые связи Qwen для пилотных текстов</label></header>
    <main id="viewer" hidden><section class="card"><label>Выберите текст<select id="pick"></select></label><p id="text" class="text"></p><p id="source" class="hint"></p><h2>Связи OpenIE</h2><div id="graph"></div><div id="edges"></div><h2>Смысловые признаки</h2><p id="labels"></p></section></main>'''
    script = f'const docs={js_json(docs)},pred={js_json({d["id"]:pred[d["id"]] for d in docs if d["id"] in pred})},labels={js_json(LABELS)};\n'+DEMO_JS
    (ROOT/'results/03_demo_after_annotation.html').write_text(shell('Смысловые графы Qwen', body, script), encoding='utf-8')

REVIEW_JS = r'''
const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const storageKey='para0911_review_'+meta.protocol_sha256+'_'+meta.group;
let draft={};try{draft=JSON.parse(localStorage.getItem(storageKey)||'{}')}catch(e){}
function message(text){document.querySelector('#notice').textContent=text;}
document.querySelector('#cards').innerHTML=docs.map((d,i)=>`<section class="card" data-id="${d.id}"><h2>Текст ${i+1}</h2><p class="text">${esc(d.text)}</p>
<h3>Ваши независимые связи</h3>${human[d.id].triples.map((t,j)=>`<p>${j+1}. ${t.map(esc).join(' → ')}</p>`).join('')}
<h3>Связи модели</h3>${pred[d.id].prediction.triples.map((t,j)=>`<div class="edge" data-link="${j}"><b>${t.slice(0,3).map(esc).join(' → ')}</b><p class="quote">Цитата: «${esc(t[3])}»</p>
<label>Подтверждается ли вся связь текстом, включая отрицание и авторство?<select data-supported><option value="">Выберите</option><option value="yes">Да</option><option value="no">Нет</option></select></label>
<label>Совпадает ли по смыслу с одной из ваших связей?<select data-match><option value="">Выберите</option><option value="-1">Нет эквивалентной связи</option>${human[d.id].triples.map((_,k)=>`<option value="${k}">Моя связь ${k+1}</option>`).join('')}</select></label></div>`).join('')||'<p>Модель не выделила связей.</p>'}</section>`).join('');
for(const s of document.querySelectorAll('[data-id]'))for(const l of s.querySelectorAll('[data-link]')){
 const a=draft[s.dataset.id]?.[+l.dataset.link];if(a){l.querySelector('[data-supported]').value=a.supported;l.querySelector('[data-match]').value=a.match;}}
function collect(){return [...document.querySelectorAll('[data-id]')].map(s=>({id:s.dataset.id,links:[...s.querySelectorAll('[data-link]')].map(l=>({supported:l.querySelector('[data-supported]').value,match:l.querySelector('[data-match]').value}))}));}
document.querySelector('#cards').addEventListener('change',()=>{try{localStorage.setItem(storageKey,JSON.stringify(Object.fromEntries(collect().map(a=>[a.id,a.links]))))}catch(e){}});
document.querySelector('#export').onclick=()=>{
 const answers=collect();for(let i=0;i<answers.length;i++){
 const used=[];for(const l of answers[i].links){if(!l.supported||l.match===''){message(`Заполните все оценки в тексте ${i+1}`);return;}
 if(+l.match>=0){if(l.supported!=='yes'){message('Смысловое совпадение требует подтверждения связи текстом.');return;}if(used.includes(l.match)){message(`Текст ${i+1}: одну вашу связь нельзя сопоставлять дважды.`);return;}used.push(l.match);}}}
 const data={...meta,annotator:'participant_1',exported_at:new Date().toISOString(),answers:answers.map(a=>({id:a.id,links:a.links.map(l=>({supported:l.supported==='yes',human_index:+l.match}))}))};
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.download=meta.group+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
};
'''

def review_form(group):
    human = load_annotations(group)
    if human is None:
        return
    docs = [d for d in group_docs(group) if d['id'] in human]
    pred = predictions()
    if any(d['id'] not in pred for d in docs):
        return
    title = 'Сопоставление ваших смыслов и связей Qwen'
    body = '''<header><h1>Сопоставление смысловых связей</h1><p>Ваши первоначальные ответы уже зафиксированы. Теперь оцените каждую связь модели: подтверждается ли она текстом и есть ли смысловой эквивалент среди ваших связей.</p><p>Формулировки могут отличаться. Для совпадения должны совпадать участники, действие, направление, отрицание и степень уверенности. Сопоставление взаимно однозначное: одну вашу связь можно использовать не более одного раза. Дополнительная верная связь модели без вашего эквивалента — не галлюцинация, но не совпадение с вашей разметкой.</p><p>Экспортируйте JSON в <b>annotation/inbox</b>. Оценку даёт тот же участник; независимым экспертным эталоном она не является.</p></header><div class="toolbar"><button id="export">Экспорт оценок связей</button><p id="notice" role="alert" aria-live="polite"></p></div><main id="cards"></main>'''
    meta = {'group': 'review_'+group, 'protocol_sha256': protocol_hash()}
    script = f'const docs={js_json(docs)},human={js_json(human)},pred={js_json({d["id"]:pred[d["id"]] for d in docs})},meta={js_json(meta)};\n'+REVIEW_JS
    (ROOT/f'annotation/04_review_{group}.html').write_text(shell(title, body, script), encoding='utf-8')

def build_forms():
    blind_form('pilot'); blind_form('audit'); demo()
    review_form('pilot'); review_form('audit')
    print('Created blind pilot, blind audit and gated model demonstration.', flush=True)

if __name__ == '__main__':
    build_forms()
