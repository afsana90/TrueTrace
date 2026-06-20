'use strict';

const S = {
  uploadFile: null, videoFile: null,
  penColor: '#1a1a2e', penSize: 6, canvasBg: '#ffffff',
  drawing: false, strokes: [],
  webcamStream: null,
  totalExtractions: 0, totalWords: 0,
  uploadTranslateTo: '', videoTranslateTo: '', webcamTranslateTo: '',
  webcamLastResultKey: '',
};

document.addEventListener('DOMContentLoaded', () => {
  initHealth();
  initTabs();
  initLanguageSelectors();
  initTranslateButtons();
  initUpload();
  initCanvas();
  initVideo();
  initWebcam();
});

async function initHealth() {
  const dot = document.getElementById('statusDot');
  const txt = document.getElementById('statusText');
  try {
    const r = await fetch('/api/health');
    const d = await r.json();
    if (d.status === 'ok') {
      dot.className = 'status-dot online';
      txt.textContent = 'System Ready';
    } else throw new Error();
  } catch {
    dot.className = 'status-dot error';
    txt.textContent = 'Server offline';
  }
}

async function initLanguageSelectors() {
  try {
    const r = await fetch('/api/languages');
    const d = await r.json();
    const selects = ['uploadLang', 'videoLang', 'webcamLang'];
    selects.forEach(id => {
      const sel = document.getElementById(id);
      if (!sel) return;
      sel.innerHTML = '';
      d.languages.forEach(lang => {
        const opt = document.createElement('option');
        opt.value = lang.key;
        opt.textContent = `${lang.flag} ${lang.name}`;
        if (lang.key === 'auto') opt.selected = true;
        sel.appendChild(opt);
      });
    });
  } catch(e) {
    console.warn('Could not load languages:', e);
  }
}

function initTabs() {
  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      document.getElementById('panel-' + tab.dataset.mode).classList.add('active');
      if (tab.dataset.mode !== 'webcam' && S.webcamStream) stopWebcam();
    });
  });
}

function initTranslateButtons() {
  [
    { group: 'uploadTranslateBtns',  key: 'uploadTranslateTo'  },
    { group: 'videoTranslateBtns',   key: 'videoTranslateTo'   },
    { group: 'webcamTranslateBtns',  key: 'webcamTranslateTo'  },
  ].forEach(({ group, key }) => {
    const container = document.getElementById(group);
    if (!container) return;
    container.querySelectorAll('.trans-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        container.querySelectorAll('.trans-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        S[key] = btn.dataset.lang;
      });
    });
  });
}

function uiState(prefix, which, errMsg) {
  ['Idle','Working','Output','Error'].forEach(s => {
    const el = document.getElementById(prefix + s);
    if (el) el.style.display = 'none';
  });
  const t = document.getElementById(prefix + which);
  if (t) t.style.display = 'flex';
  if (which === 'Error' && errMsg) {
    const map = {upload:'uploadErrMsg',draw:'drawErrMsg',video:'videoErrMsg',webcam:'webcamErrMsg'};
    const el = document.getElementById(map[prefix]);
    if (el) el.textContent = errMsg;
  }
}

function renderNotebook(containerId, lines) {
  const el = document.getElementById(containerId);
  if (!el) return;
  el.innerHTML = '';
  if (!lines || !lines.length) {
    el.innerHTML = '<div class="notebook-line empty">No text detected</div>';
    return;
  }
  lines.forEach(line => {
    const div = document.createElement('div');
    div.className = 'notebook-line';
    const span = document.createElement('span');
    span.textContent = line.text || '';
    const conf = document.createElement('span');
    conf.className = 'nb-conf';
    if (line.confidence) conf.textContent = line.confidence + '%';
    div.appendChild(span); div.appendChild(conf);
    el.appendChild(div);
  });
}

function renderConf(id, conf) {
  const el = document.getElementById(id);
  if (!el) return;
  const cls = conf >= 70 ? 'high' : conf >= 40 ? 'mid' : 'low';
  el.innerHTML = `<div class="conf-label">Confidence</div>
    <div class="conf-track"><div class="conf-fill ${cls}" style="width:${Math.min(conf,100)}%"></div></div>`;
}

function renderMeta(id, data) {
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = `Lines: <b>${data.total_lines}</b> &nbsp;·&nbsp; Words: <b>${data.word_count}</b> &nbsp;·&nbsp; Confidence: <b>${data.avg_confidence}%</b>`;
}

function showOverlay(wrapId, imgId, b64) {
  const wrap = document.getElementById(wrapId);
  const img  = document.getElementById(imgId);
  if (wrap && img && b64) {
    img.src = b64;
    wrap.style.display = 'block';
  }
}

function showTranslation(prefix, data) {
  const panel = document.getElementById(prefix + 'TransPanel');
  const nb    = document.getElementById(prefix + 'TransNotebook');
  const raw   = document.getElementById(prefix + 'TransRaw');
  const title = document.getElementById(prefix + 'TransTitle');

  if (!panel || !data.translated_text) { if(panel) panel.style.display='none'; return; }

  const langName = data.translate_to === 'en' ? '🇬🇧 English' : '🇧🇩 Bangla';
  if (title) title.textContent = `Translation → ${langName}`;

  const lines = data.translated_text.split('\n').map(t => ({text: t, confidence: null}));
  renderNotebook(prefix + 'TransNotebook', lines);
  if (raw) raw.value = data.translated_text;
  panel.style.display = 'block';
}

function setRaw(id, text) {
  const el = document.getElementById(id);
  if (el) el.value = text;
}

function applyResult(prefix, data) {
  renderMeta(prefix + 'Meta', data);
  renderConf(prefix + 'Conf', data.avg_confidence);
  renderNotebook(prefix + 'Notebook', data.lines);
  setRaw(prefix + 'Raw', data.full_text);
  if (data.overlay_image) showOverlay(prefix + 'OverlayWrap', prefix + 'OverlayImg', data.overlay_image);
  showTranslation(prefix, data);
  uiState(prefix, 'Output');
  S.totalExtractions++;
  S.totalWords += data.word_count || 0;
  const fs = document.getElementById('footerStats');
  if (fs) fs.textContent = `${S.totalExtractions} extraction${S.totalExtractions!==1?'s':''} · ${S.totalWords} words`;
}

function copyText(rawId) {
  const el = document.getElementById(rawId);
  if (!el || !el.value) return;
  navigator.clipboard.writeText(el.value).then(() => {
    const btn = [...document.querySelectorAll('.btn-action')].find(b => b.getAttribute('onclick')?.includes(rawId));
    if (btn) { const o = btn.textContent; btn.textContent = '✓ Copied!'; setTimeout(()=>btn.textContent=o,1600); }
  });
}
function downloadTxt(rawId, name) {
  const el = document.getElementById(rawId);
  if (!el || !el.value) return;
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([el.value], {type:'text/plain'}));
  a.download = (name||'truetrace') + '.txt';
  a.click();
}

function initUpload() {
  const drop   = document.getElementById('uploadDrop');
  const input  = document.getElementById('uploadInput');
  document.getElementById('uploadBrowse').addEventListener('click', e => { e.stopPropagation(); input.click(); });
  drop.addEventListener('click', () => input.click());
  drop.addEventListener('dragover',  e => { e.preventDefault(); drop.classList.add('drag'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('drag'));
  drop.addEventListener('drop', e => {
    e.preventDefault(); drop.classList.remove('drag');
    if (e.dataTransfer.files[0]) pickUploadFile(e.dataTransfer.files[0]);
  });
  input.addEventListener('change', e => { if (e.target.files[0]) pickUploadFile(e.target.files[0]); });
  document.getElementById('uploadRemove').addEventListener('click', clearUpload);
  document.getElementById('uploadBtn').addEventListener('click', runUpload);
}

function pickUploadFile(file) {
  S.uploadFile = file;
  const thumb = document.getElementById('uploadThumb');
  thumb.innerHTML = '';
  if (file.type.startsWith('image/')) {
    const img = document.createElement('img'); img.src = URL.createObjectURL(file); thumb.appendChild(img);
  }
  document.getElementById('uploadName').textContent = `${file.name} · ${(file.size/1024/1024).toFixed(1)} MB`;
  document.getElementById('uploadDrop').style.display = 'none';
  document.getElementById('uploadChip').style.display = 'flex';
  document.getElementById('uploadBtn').disabled = false;
  uiState('upload','Idle');
}

function clearUpload() {
  S.uploadFile = null;
  document.getElementById('uploadInput').value = '';
  document.getElementById('uploadDrop').style.display = 'block';
  document.getElementById('uploadChip').style.display = 'none';
  document.getElementById('uploadBtn').disabled = true;
  document.getElementById('uploadOverlayWrap').style.display = 'none';
  uiState('upload','Idle');
}

const UPLOAD_MSGS = ['Detecting language…','Preprocessing image…','Recognising text…','Sorting lines…','Generating overlay…'];
async function runUpload() {
  if (!S.uploadFile) return;
  uiState('upload','Working');
  document.getElementById('uploadBtn').disabled = true;
  let mi = 0;
  const lbl = document.getElementById('uploadWorkingMsg');
  const iv = setInterval(() => { if(lbl) lbl.textContent = UPLOAD_MSGS[mi++ % UPLOAD_MSGS.length]; }, 1600);
  const fd = new FormData();
  fd.append('file', S.uploadFile);
  fd.append('preprocess', 'true');
  fd.append('lang_key', document.getElementById('uploadLang').value);
  fd.append('translate_to', S.uploadTranslateTo);
  try {
    const r = await fetch('/api/extract', {method:'POST', body:fd});
    const d = await r.json();
    clearInterval(iv);
    if (!r.ok) throw new Error(d.detail || 'Failed');
    applyResult('upload', d);
  } catch(e) {
    clearInterval(iv);
    uiState('upload','Error', e.message);
  }
  document.getElementById('uploadBtn').disabled = false;
}

function initCanvas() {
  const canvas = document.getElementById('drawCanvas');
  const ctx    = canvas.getContext('2d');
  let lx = 0, ly = 0, currentStroke = [];
  let currentBg = '#ffffff';

  function setupCanvas() {
    const wrap = document.getElementById('canvasWrap');
    const W = wrap.clientWidth;
    const H = Math.max(280, Math.round(window.innerHeight * 0.36));
    const dpr = window.devicePixelRatio || 1;
    canvas.width  = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    canvas.style.width  = W + 'px'; canvas.style.height = H + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    fillBg();
  }

  function fillBg() {
    const W = parseInt(canvas.style.width), H = parseInt(canvas.style.height);
    ctx.fillStyle = currentBg; ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = currentBg === '#1a1a2e' ? 'rgba(108,99,255,0.2)' : 'rgba(108,99,255,0.12)';
    ctx.lineWidth = 1;
    for (let y = 36; y < H; y += 36) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke(); }
  }

  setupCanvas(); window.addEventListener('resize', setupCanvas);

  function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    const src = e.touches ? e.touches[0] : e;
    return { x: src.clientX - rect.left, y: src.clientY - rect.top };
  }

  function startDraw(e) {
    e.preventDefault(); S.drawing = true;
    const { x, y } = getPos(e); lx = x; ly = y;
    currentStroke = [{x,y}];
    ctx.beginPath(); ctx.arc(x, y, S.penSize/2, 0, Math.PI*2);
    ctx.fillStyle = S.penColor; ctx.fill();
  }
  function doDraw(e) {
    e.preventDefault(); if (!S.drawing) return;
    const { x, y } = getPos(e);
    ctx.beginPath(); ctx.moveTo(lx,ly); ctx.lineTo(x,y);
    ctx.strokeStyle = S.penColor; ctx.lineWidth = S.penSize;
    ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.stroke();
    currentStroke.push({x,y}); lx=x; ly=y;
  }
  function endDraw() {
    if (!S.drawing) return; S.drawing = false;
    if (currentStroke.length > 1) S.strokes.push({color:S.penColor, size:S.penSize, pts:[...currentStroke]});
    currentStroke = [];
  }

  canvas.addEventListener('mousedown',  startDraw);
  canvas.addEventListener('mousemove',  doDraw);
  canvas.addEventListener('mouseup',    endDraw);
  canvas.addEventListener('mouseleave', endDraw);
  canvas.addEventListener('touchstart', startDraw, {passive:false});
  canvas.addEventListener('touchmove',  doDraw,    {passive:false});
  canvas.addEventListener('touchend',   endDraw);

  document.getElementById('penSize').addEventListener('input', e => {
    S.penSize = +e.target.value; document.getElementById('penSizeVal').textContent = S.penSize;
  });
  document.querySelectorAll('.swatch').forEach(s => {
    s.addEventListener('click', () => { document.querySelectorAll('.swatch').forEach(x=>x.classList.remove('active')); s.classList.add('active'); S.penColor = s.dataset.c; });
  });
  document.querySelectorAll('.bg-btn').forEach(b => {
    b.addEventListener('click', () => { document.querySelectorAll('.bg-btn').forEach(x=>x.classList.remove('active')); b.classList.add('active'); currentBg=b.dataset.bg; S.strokes=[]; setupCanvas(); });
  });
  document.getElementById('drawUndo').addEventListener('click', () => {
    if (!S.strokes.length) return; S.strokes.pop(); fillBg();
    S.strokes.forEach(stroke => {
      if (!stroke.pts.length) return;
      ctx.strokeStyle=stroke.color; ctx.lineWidth=stroke.size; ctx.lineCap='round'; ctx.lineJoin='round';
      ctx.beginPath(); ctx.moveTo(stroke.pts[0].x, stroke.pts[0].y);
      stroke.pts.slice(1).forEach(p=>ctx.lineTo(p.x,p.y)); ctx.stroke();
    });
  });
  document.getElementById('drawClear').addEventListener('click', () => { S.strokes=[]; fillBg(); uiState('draw','Idle'); });

  document.getElementById('drawRecognise').addEventListener('click', async () => {
    if (!S.strokes.length) { uiState('draw','Error','Please write something on the canvas first.'); return; }
    uiState('draw','Working');
    const b64 = canvas.toDataURL('image/png');
    try {
      const r = await fetch('/api/extract-drawing', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({image_base64: b64}),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || 'Failed');
      applyResult('draw', d);
      updateTip(d.avg_confidence);
    } catch(e) { uiState('draw','Error', e.message); }
  });
}
function updateTip(conf) {
  const b = document.getElementById('tipBody');
  if (!b) return;
  if (conf >= 80) b.textContent = 'Excellent! Your handwriting is very clear.';
  else if (conf >= 55) b.textContent = 'Good! Try larger letters with bigger gaps between words.';
  else b.textContent = 'Try writing larger with firm strokes and clear spacing.';
}

function initVideo() {
  const drop = document.getElementById('videoDrop');
  const input = document.getElementById('videoInput');
  document.getElementById('videoBrowse').addEventListener('click', e => { e.stopPropagation(); input.click(); });
  drop.addEventListener('click', () => input.click());
  drop.addEventListener('dragover',  e => { e.preventDefault(); drop.classList.add('drag'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('drag'));
  drop.addEventListener('drop', e => {
    e.preventDefault(); drop.classList.remove('drag');
    if (e.dataTransfer.files[0]) pickVideoFile(e.dataTransfer.files[0]);
  });
  input.addEventListener('change', e => { if (e.target.files[0]) pickVideoFile(e.target.files[0]); });
  document.getElementById('videoRemove').addEventListener('click', clearVideo);
  document.getElementById('videoBtn').addEventListener('click', runVideo);
}

function pickVideoFile(file) {
  S.videoFile = file;
  document.getElementById('videoName').textContent = `${file.name} · ${(file.size/1024/1024).toFixed(1)} MB`;
  document.getElementById('videoDrop').style.display = 'none';
  document.getElementById('videoChip').style.display = 'flex';
  document.getElementById('videoBtn').disabled = false;
  uiState('video','Idle');
}
function clearVideo() {
  S.videoFile = null;
  document.getElementById('videoInput').value = '';
  document.getElementById('videoDrop').style.display = 'block';
  document.getElementById('videoChip').style.display = 'none';
  document.getElementById('videoBtn').disabled = true;
  document.getElementById('videoProgress').style.display = 'none';
  document.getElementById('videoOverlayWrap').style.display = 'none';
  uiState('video','Idle');
}

const VIDEO_MSGS = ['Detecting scenes…','Running OCR per scene…','Removing duplicates…','Merging results…','Translating…'];
async function runVideo() {
  if (!S.videoFile) return;
  uiState('video','Working');
  document.getElementById('videoBtn').disabled = true;
  document.getElementById('videoProgress').style.display = 'block';
  let mi = 0;
  const lbl  = document.getElementById('videoWorkingMsg');
  const pmsg = document.getElementById('progressMsg');
  const fill = document.getElementById('progressFill');
  const iv = setInterval(() => {
    const msg = VIDEO_MSGS[mi % VIDEO_MSGS.length];
    if (lbl) lbl.textContent = msg;
    if (pmsg) pmsg.textContent = msg;
    if (fill) fill.style.width = Math.min(90, mi * 18) + '%';
    mi++;
  }, 2000);
  const fd = new FormData();
  fd.append('file', S.videoFile);
  fd.append('preprocess', 'true');
  fd.append('lang_key', document.getElementById('videoLang').value);
  fd.append('translate_to', S.videoTranslateTo);
  try {
    const r = await fetch('/api/extract', {method:'POST', body:fd});
    const d = await r.json();
    clearInterval(iv);
    if (fill) fill.style.width = '100%';
    document.getElementById('videoProgress').style.display = 'none';
    if (!r.ok) throw new Error(d.detail || 'Failed');
    applyResult('video', d);
  } catch(e) {
    clearInterval(iv);
    document.getElementById('videoProgress').style.display = 'none';
    uiState('video','Error', e.message);
  }
  document.getElementById('videoBtn').disabled = false;
}


function setDisplaySafe(id, display) {
  const el = document.getElementById(id);
  if (el) el.style.display = display;
}

function setTextSafe(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

// ── Webcam: smart live OCR every 1 second ───────────────────────────────────
let _webcamTimer   = null;   // setInterval handle
let _webcamRunning = false;  // guard: skip tick if previous request in-flight

function initWebcam() {
  document.getElementById('webcamToggle').addEventListener('click', toggleWebcam);
}

async function toggleWebcam() {
  if (S.webcamStream) { stopWebcam(); return; }

  let stream = null;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
    });
  } catch(e) {
    uiState('webcam', 'Error', 'Camera access denied or camera not available: ' + e.message);
    return;
  }

  try {
    const video = document.getElementById('webcamVideo');
    if (!video) throw new Error('Webcam video element not found.');

    S.webcamStream = stream;
    video.srcObject = S.webcamStream;

    // Wait for the video to have actual dimensions before we start scanning.
    // The timeout prevents the app from hanging if metadata is delayed.
    await new Promise(res => {
      if (video.readyState >= 1 && video.videoWidth) return res();
      video.onloadedmetadata = res;
      setTimeout(res, 1200);
    });
    try { await video.play(); } catch (_) {}

    setDisplaySafe('webcamIdle2', 'none');
    setDisplaySafe('scanLine', 'block');
    setTextSafe('webcamToggle', 'Stop Camera');

    // Show the result area immediately so text updates are visible
    uiState('webcam', 'Idle');

    // Kick off smart live OCR loop
    S.webcamLastResultKey = '';
    _webcamRunning = false;
    clearInterval(_webcamTimer);
    _webcamTimer = setInterval(_webcamTick, 1000);
    _webcamTick(); // fire once immediately so there's no 1-second blank wait
  } catch(e) {
    if (stream) stream.getTracks().forEach(t => t.stop());
    S.webcamStream = null;
    uiState('webcam', 'Error', 'Webcam setup error: ' + e.message);
  }
}

function stopWebcam() {
  clearInterval(_webcamTimer);
  _webcamTimer = null;
  _webcamRunning = false;
  if (S.webcamStream) { S.webcamStream.getTracks().forEach(t => t.stop()); S.webcamStream = null; }
  const video = document.getElementById('webcamVideo');
  if (video) video.srcObject = null;
  setDisplaySafe('webcamIdle2', 'flex');
  setDisplaySafe('scanLine', 'none');
  setTextSafe('webcamToggle', 'Start Camera');
}

async function _webcamTick() {
  // Skip if a previous request is still in-flight or stream is gone
  if (_webcamRunning || !S.webcamStream) return;
  _webcamRunning = true;
  try {
    const video = document.getElementById('webcamVideo');
    const snap  = document.getElementById('webcamSnap');
    if (!video || !snap) throw new Error('Webcam preview elements not found.');

    const srcW = video.videoWidth  || 640;
    const srcH = video.videoHeight || 480;

    // Keep the frame smaller for transport. Backend still uses Claude Vision
    // smartly in the background, so the UI can poll every second without freezing.
    const maxW = 1100;
    const scale = Math.min(1, maxW / srcW);
    const w = Math.round(srcW * scale);
    const h = Math.round(srcH * scale);

    snap.width  = w;
    snap.height = h;
    snap.getContext('2d').drawImage(video, 0, 0, w, h);

    // Convert current live frame to blob and POST to API.
    // This is not saved or shown as a captured image; it is only used for recognition.
    const blob = await new Promise(res => snap.toBlob(res, 'image/jpeg', 0.78));
    if (!blob) throw new Error('Could not read webcam frame.');

    const fd = new FormData();
    fd.append('file', new File([blob], 'webcam.jpg', { type: 'image/jpeg' }));
    fd.append('preprocess', 'true');
    const langEl = document.getElementById('webcamLang');
    fd.append('lang_key', langEl ? langEl.value : 'auto');
    fd.append('translate_to', S.webcamTranslateTo);

    const r = await fetch('/api/extract-webcam-fast', { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Failed');

    // Only update result display — do NOT call uiState('webcam','Working')
    // so the UI never flickers to a loading state between ticks.
    _webcamApplyLive(d);
  } catch(e) {
    // Silently swallow transient errors so the loop keeps running;
    // only show error if camera was deliberately stopped.
    if (S.webcamStream) console.warn('Webcam OCR tick error:', e.message);
  } finally {
    _webcamRunning = false;
  }
}

/** Like applyResult but updates the webcam panel in-place without hiding/showing states */
function _webcamApplyLive(data) {
  // Ensure output state is visible (no flicker — already shown after start)
  const out = document.getElementById('webcamOutput');
  const idle = document.getElementById('webcamIdle');
  if (idle) idle.style.display = 'none';
  if (out)  out.style.display  = 'flex';

  renderMeta('webcamMeta', data);
  renderConf('webcamConf', data.avg_confidence);
  renderNotebook('webcamNotebook', data.lines);
  setRaw('webcamRaw', data.full_text);
  if (data.overlay_image) {
    showOverlay('webcamOverlayWrap', 'webcamOverlayImg', data.overlay_image);
  } else {
    setDisplaySafe('webcamOverlayWrap', 'none');
  }
  showTranslation('webcam', data);

  // Count only when a new OCR result arrives, not every cached 1-second poll.
  const resultKey = data.live_result_id || `${data.full_text || ''}:${data.avg_confidence || 0}`;
  if (resultKey && resultKey !== S.webcamLastResultKey && (data.full_text || '').trim()) {
    S.webcamLastResultKey = resultKey;
    S.totalExtractions++;
    S.totalWords += data.word_count || 0;
    const fs = document.getElementById('footerStats');
    if (fs) fs.textContent = `${S.totalExtractions} extraction${S.totalExtractions !== 1 ? 's' : ''} · ${S.totalWords} words`;
  }
}
