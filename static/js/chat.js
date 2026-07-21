/* ── marked.js configuration ───────────────────────────────────── */
marked.setOptions({
  breaks: true,
  gfm: true,
  highlight(code, lang) {
    if (lang && hljs.getLanguage(lang)) {
      return hljs.highlight(code, { language: lang }).value;
    }
    return hljs.highlightAuto(code).value;
  },
});

/* ── DOM refs ───────────────────────────────────────────────────── */
const messagesEl  = document.getElementById('messages');
const welcomeEl   = document.getElementById('welcome');
const msgInput    = document.getElementById('msgInput');
const sendBtn     = document.getElementById('sendBtn');
const charCount   = document.getElementById('charCount');
const clearBtn    = document.getElementById('clearBtn');
const newChatBtn  = document.getElementById('newChatBtn');
const statusPill  = document.getElementById('statusPill');
const statusText  = document.getElementById('statusText');

/* ── State ──────────────────────────────────────────────────────── */
let chatHistory = [];
let streaming   = false;

/* ── Auto-resize textarea ───────────────────────────────────────── */
msgInput.addEventListener('input', () => {
  msgInput.style.height = 'auto';
  msgInput.style.height = Math.min(msgInput.scrollHeight, 180) + 'px';
  const len = msgInput.value.length;
  charCount.textContent = `${len} / 2000`;
  sendBtn.disabled = len === 0 || streaming;
});

/* Send on Enter (Shift+Enter = newline) */
msgInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); if (!sendBtn.disabled) sendMessage(); }
});

sendBtn.addEventListener('click', sendMessage);
clearBtn.addEventListener('click', resetChat);
newChatBtn.addEventListener('click', resetChat);

/* Quick-topic chips */
document.querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', () => {
    msgInput.value = chip.dataset.q;
    msgInput.dispatchEvent(new Event('input'));
    sendMessage();
  });
});

/* Welcome suggestion cards (global so onclick= in HTML can reach it) */
window.injectSuggestion = function (card) {
  const text = card.dataset.q;
  msgInput.value = text;
  msgInput.dispatchEvent(new Event('input'));
  sendMessage();
};

/* ── Reset ──────────────────────────────────────────────────────── */
function resetChat() {
  chatHistory = [];
  messagesEl.innerHTML = '';
  const w = document.createElement('div');
  w.id = 'welcome';
  w.className = 'welcome';
  w.innerHTML = welcomeTemplate();
  messagesEl.appendChild(w);
  // re-attach onclick listeners for new cards
  w.querySelectorAll('.suggest-card').forEach(c => c.setAttribute('onclick', 'injectSuggestion(this)'));
  setStatus('ready');
}

function welcomeTemplate() {
  return `
    <div class="welcome-icon">
      <svg width="52" height="52" viewBox="0 0 52 52" fill="none">
        <rect width="52" height="52" rx="14" fill="url(#wg2)"/>
        <polyline points="10 38 18 26 24 32 32 18 42 24" stroke="white" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
        <defs>
          <linearGradient id="wg2" x1="0" y1="0" x2="52" y2="52">
            <stop offset="0%" stop-color="#10b981"/><stop offset="100%" stop-color="#065f46"/>
          </linearGradient>
        </defs>
      </svg>
    </div>
    <h1>Welcome to <span class="grad">FinBot</span></h1>
    <p>Your AI-powered financial insights assistant. Ask me to analyze financial reports from BlackRock, the Federal Reserve, and Vanguard — or explore investment strategies and market insights.</p>
    <div class="suggest-grid">
      <div class="suggest-card" data-q="What are the key investment insights from BlackRock's 2026 mid-year outlook?" onclick="injectSuggestion(this)">
        <span class="suggest-icon">📋</span>
        <div><strong>BlackRock 2026 Outlook</strong><span>What are the key investment insights from BlackRock's mid-year report?</span></div>
      </div>
      <div class="suggest-card" data-q="Summarize the Federal Reserve Beige Book findings for June 2026" onclick="injectSuggestion(this)">
        <span class="suggest-icon">🏦</span>
        <div><strong>Fed Beige Book Analysis</strong><span>Summarize the Federal Reserve Beige Book findings for June 2026</span></div>
      </div>
      <div class="suggest-card" data-q="What does Vanguard say about retirement income principles and sustainable withdrawal strategies?" onclick="injectSuggestion(this)">
        <span class="suggest-icon">🎯</span>
        <div><strong>Vanguard Retirement Income</strong><span>What does Vanguard say about sustainable withdrawal strategies?</span></div>
      </div>
      <div class="suggest-card" data-q="What macro risks and opportunities does BlackRock identify for equity investors in 2026?" onclick="injectSuggestion(this)">
        <span class="suggest-icon">📊</span>
        <div><strong>2026 Macro Insights</strong><span>What macro risks and opportunities do the reports highlight for 2026?</span></div>
      </div>
    </div>`;
}

/* ── Status helpers ─────────────────────────────────────────────── */
function setStatus(state, text) {
  const dot = statusPill.querySelector('.status-dot');
  if (state === 'thinking') {
    dot.style.background = '#f59e0b';
    dot.style.boxShadow  = '0 0 6px #f59e0b';
    statusText.textContent = text || 'Thinking…';
  } else if (state === 'streaming') {
    dot.style.background = '#3b82f6';
    dot.style.boxShadow  = '0 0 6px #3b82f6';
    statusText.textContent = text || 'Responding…';
  } else {
    dot.style.background = '';
    dot.style.boxShadow  = '';
    statusText.textContent = 'FinBot Ready';
  }
}

/* ── Main send function ─────────────────────────────────────────── */
async function sendMessage() {
  const message = msgInput.value.trim();
  if (!message || streaming) return;

  // Hide welcome screen
  const w = document.getElementById('welcome');
  if (w) w.remove();

  // Append user bubble
  const userMsgEl = appendUserMsg(message);
  chatHistory.push({ role: 'user', content: message });

  // Reset input
  msgInput.value = '';
  msgInput.style.height = 'auto';
  charCount.textContent = '0 / 2000';
  sendBtn.disabled = true;
  streaming = true;
  setStatus('thinking');

  // Create bot bubble placeholder
  const botEl = createBotBubble();

  try {
    const resp = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, history: chatHistory.slice(0, -1) }),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      removeBotBubble(botEl);
      userMsgEl.remove();
      chatHistory.pop();
      showToast(err.error || `Server error ${resp.status}`);
      return;
    }

    const reader  = resp.body.getReader();
    const decoder = new TextDecoder();
    let fullContent   = '';
    let fullReasoning = '';
    let hasContent    = false;
    let ragSources    = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const raw   = decoder.decode(value, { stream: true });
      const lines = raw.split('\n');

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6).trim();
        if (payload === '[DONE]') continue;

        let parsed;
        try { parsed = JSON.parse(payload); } catch { continue; }

        if (parsed.blocked) {
          chatHistory.pop();              // remove from memory
          showBlockedResponse(botEl, parsed.error || 'Your message was blocked by our safety guardrails.');
          return;
        }

        if (parsed.error) { showToast(parsed.error); continue; }

        if (parsed.status) {
          setStatus('thinking', parsed.status);
          continue;
        }

        if (parsed.tool_call) {
          showToolCall(botEl, parsed.tool_call);
          setStatus('thinking', `Using ${parsed.tool_call}…`);
          continue;
        }

        if (parsed.tool_done) {
          // stop spinners on all tool badges
          botEl.querySelectorAll('.tool-spinner').forEach(s => s.remove());
          setStatus('thinking');
          continue;
        }

        if (parsed.sources !== undefined) {
          ragSources = parsed.sources;
          continue;
        }

        if (parsed.reasoning) {
          fullReasoning += parsed.reasoning;
          updateReasoning(botEl, parsed.reasoning); // pass only new chunk — appends
          if (!hasContent) setStatus('thinking');
        }

        if (parsed.content) {
          if (!hasContent) { hasContent = true; setStatus('streaming'); }
          fullContent += parsed.content;
          updateContent(botEl, fullContent, /* streaming= */ true);
        }
      }
    }

    // Finalise
    if (fullContent) {
      chatHistory.push({ role: 'assistant', content: fullContent });
      finalise(botEl, fullContent, fullReasoning, ragSources);
    } else if (!fullReasoning) {
      removeBotBubble(botEl);
      showToast('No response received. Please try again.');
    }

  } catch (err) {
    removeBotBubble(botEl);
    showToast('Connection error. Please check your network and try again.');
    console.error(err);
  } finally {
    streaming = false;
    setStatus('ready');
    sendBtn.disabled = msgInput.value.trim().length === 0;
    scrollBottom();
  }
}

/* ── DOM builders ───────────────────────────────────────────────── */
function appendUserMsg(text) {
  const el = document.createElement('div');
  el.className = 'msg msg-user';
  el.innerHTML = `
    <div class="avatar avatar-user">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
        <circle cx="12" cy="7" r="4"/>
      </svg>
    </div>
    <div class="bubble">${escHtml(text)}</div>`;
  messagesEl.appendChild(el);
  scrollBottom();
  return el;
}

function createBotBubble() {
  const el = document.createElement('div');
  el.className = 'msg msg-bot';
  el.innerHTML = `
    <div class="avatar avatar-bot">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/>
        <polyline points="16 7 22 7 22 13"/>
      </svg>
    </div>
    <div class="msg-body">
      <div class="think-block" style="display:none">
        <div class="think-header">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 14.5v-9l6 4.5-6 4.5z"/></svg>
          Chain-of-Thought Reasoning
          <button class="think-toggle">Show</button>
        </div>
        <div class="think-body" style="display:none"></div>
      </div>
      <div class="bubble-bot">
        <div class="typing"><span></span><span></span><span></span></div>
      </div>
    </div>`;

  const toggle = el.querySelector('.think-toggle');
  const body   = el.querySelector('.think-body');
  toggle.addEventListener('click', () => {
    const open = body.style.display !== 'none';
    body.style.display = open ? 'none' : 'block';
    toggle.textContent = open ? 'Show' : 'Hide';
  });

  messagesEl.appendChild(el);
  scrollBottom();
  return el;
}

function updateReasoning(el, chunk) {
  const block  = el.querySelector('.think-block');
  const body   = el.querySelector('.think-body');
  const toggle = el.querySelector('.think-toggle');
  if (block.style.display !== 'flex') {
    // First reasoning chunk — auto-open so streaming is visible like DeepSeek
    block.style.display = 'flex';
    body.style.display  = 'block';
    if (toggle) toggle.textContent = 'Hide';
  }
  body.textContent += chunk;       // append only the new chunk (not full reset)
  // Throttle scroll to once per frame — prevents layout thrashing
  if (!body._scrollRaf) {
    body._scrollRaf = requestAnimationFrame(() => {
      body._scrollRaf = null;
      body.scrollTop  = body.scrollHeight;
      scrollBottom();
    });
  }
}

function showToolCall(el, toolName) {
  const msgBody = el.querySelector('.msg-body');
  let toolsEl   = msgBody.querySelector('.tool-calls');
  if (!toolsEl) {
    toolsEl = document.createElement('div');
    toolsEl.className = 'tool-calls';
    msgBody.insertBefore(toolsEl, msgBody.querySelector('.bubble-bot'));
  }
  const badge = document.createElement('div');
  badge.className = 'tool-call-badge';
  badge.innerHTML = `
    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M4.93 4.93a10 10 0 0 0 0 14.14"/></svg>
    <span>Using <strong>${escHtml(toolName)}</strong></span>
    <span class="tool-spinner"></span>`;
  toolsEl.appendChild(badge);
  scrollBottom();
}

function updateContent(el, markdown, isStreaming) {
  const bubble = el.querySelector('.bubble-bot');
  // Store latest markdown on the element and batch DOM write to one rAF
  // per frame — avoids full re-parse + innerHTML replacement on every chunk.
  bubble._pendingMd = markdown;
  if (bubble._rafId) return;        // already scheduled this frame
  bubble._rafId = requestAnimationFrame(() => {
    bubble._rafId = null;
    bubble.innerHTML = marked.parse(bubble._pendingMd);
    if (isStreaming) {
      const cur = document.createElement('span');
      cur.className = 'cursor';
      bubble.appendChild(cur);
    }
    // Skip hljs during streaming — finalise() runs it once at the end.
    scrollBottom();
  });
}

function finalise(el, markdown, reasoning, sources) {
  const bubble = el.querySelector('.bubble-bot');
  // Cancel any pending streaming rAF — if it fires after finalise it re-inserts
  // the blinking cursor, causing the end-of-stream flicker.
  if (bubble._rafId) {
    cancelAnimationFrame(bubble._rafId);
    bubble._rafId = null;
  }
  bubble.innerHTML = marked.parse(markdown);
  bubble.querySelectorAll('pre code').forEach(b => hljs.highlightElement(b));

  // Add copy button to each code block
  bubble.querySelectorAll('pre').forEach(pre => {
    const btn = document.createElement('button');
    btn.className = 'copy-pre';
    btn.textContent = 'Copy';
    btn.onclick = () => {
      navigator.clipboard.writeText(pre.querySelector('code').textContent);
      btn.textContent = 'Copied!';
      setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
    };
    pre.appendChild(btn);
  });

  // RAG source attribution
  const msgBody = el.querySelector('.msg-body');
  if (sources && sources.length > 0) {
    const ragEl = document.createElement('div');
    ragEl.className = 'rag-sources';
    const tags = sources.map(s =>
      `<span class="rag-tag">${escHtml(s.folder)} &rsaquo; ${escHtml(s.file)}</span>`
    ).join('');
    ragEl.innerHTML = `
      <div class="rag-header">
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
        RAG &mdash; retrieved from knowledge base
      </div>
      <div class="rag-tags">${tags}</div>`;
    msgBody.appendChild(ragEl);
  } else if (sources !== null) {
    const ragEl = document.createElement('div');
    ragEl.className = 'rag-sources rag-none';
    ragEl.innerHTML = `
      <div class="rag-header">
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
        LLM only &mdash; no knowledge base match
      </div>`;
    msgBody.appendChild(ragEl);
  }

  // Message actions (copy full response)
  const actions = document.createElement('div');
  actions.className = 'msg-actions';
  actions.innerHTML = `
    <button class="act-btn copy-resp">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
      Copy response
    </button>`;
  actions.querySelector('.copy-resp').onclick = () => {
    navigator.clipboard.writeText(markdown);
    actions.querySelector('.copy-resp').textContent = '✓ Copied!';
    setTimeout(() => {
      actions.querySelector('.copy-resp').innerHTML = `
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
        Copy response`;
    }, 2000);
  };
  el.querySelector('.msg-body').appendChild(actions);

  scrollBottom();
}

function showBlockedResponse(el, message) {
  const bubble = el.querySelector('.bubble-bot');
  bubble.innerHTML = `
    <div class="blocked-response">
      <div class="blocked-header">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/></svg>
        Blocked by Safety Guardrails
      </div>
      <p>${escHtml(message)}</p>
    </div>`;
  scrollBottom();
}

function removeBotBubble(el) { if (el?.parentNode) el.parentNode.removeChild(el); }

function showToast(message) {
  const t = document.createElement('div');
  t.className = 'err-toast';
  t.textContent = message;
  messagesEl.appendChild(t);
  scrollBottom();
  setTimeout(() => t.remove(), 6000);
}

/* ── Utilities ──────────────────────────────────────────────────── */
function scrollBottom() { messagesEl.scrollTop = messagesEl.scrollHeight; }

function escHtml(str) {
  const d = document.createElement('div');
  d.appendChild(document.createTextNode(str));
  return d.innerHTML;
}

/* ── Live Market Data ───────────────────────────────────────────── */
async function loadMarketData() {
  try {
    const resp = await fetch('/api/market');
    if (!resp.ok) return;
    const { pulse, tape } = await resp.json();

    // ── Update sidebar Market Pulse rows ──────────────────────────
    pulse.forEach((item, i) => {
      const row = document.getElementById(`pulse-row-${i}`);
      if (!row) return;
      const arrow = item.dir === 'up' ? '▲' : item.dir === 'down' ? '▼' : '';
      row.querySelector('.ticker-val').textContent = item.price;
      row.querySelector('.ticker-val').className   = `ticker-val ${item.dir}`;
      row.querySelector('.ticker-chg').textContent = `${arrow} ${item.chg}`;
      row.querySelector('.ticker-chg').className   = `ticker-chg ${item.dir}`;
    });
    const ts = document.getElementById('pulse-ts');
    if (ts) ts.textContent = new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});

    // ── Rebuild ticker tape with live data (duplicated for seamless loop) ──
    const track = document.getElementById('ticker-track');
    if (track && tape.length) {
      const makeSpans = () => tape.flatMap(item => {
        const arrow = item.dir === 'up' ? '▲' : '▼';
        const cls   = item.dir || 'up';
        return [
          `<span class="${cls}">${escHtml(item.sym)} ${escHtml(item.price)} ${arrow} ${escHtml(item.chg)}</span>`,
          `<span class="sep">·</span>`,
        ];
      }).join('');
      track.innerHTML = makeSpans() + makeSpans(); // duplicate for seamless loop
    }
  } catch (e) { /* fail silently */ }
}

// Load on page start, then refresh every 60 seconds
loadMarketData();
setInterval(loadMarketData, 60_000);
