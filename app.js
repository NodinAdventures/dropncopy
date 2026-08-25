/* =========================================================
   File Retype Admin
   - Password gate (default: admin123)
   - Drag/drop or click to upload
   - Extracts text from PDF, images (OCR), DOCX, TXT
   - Saves to localStorage; renders newest cards at bottom
   - Copy / Email (mailto) / Print per entry
========================================================= */

const PASSWORD = "LunchTime";
// Deploy marker — bump when shipping a new build. Visible in the footer so
// you can verify the browser is running the latest code without opening devtools.
const BUILD_ID = "2026-08-25-portrait-safe-crop-v25.35";

// v24: capture EVERYTHING that happens during a build so we can see
// silent failures. Wraps console.log/warn/error and fetch, and keeps
// the last 500 events in memory. The "Show debug log" button on any
// error banner prints them all in a copyable overlay.
window.__jnjDebugLog = [];
function jnjLog(kind, ...args) {
  try {
    const stamp = new Date().toISOString().slice(11, 23);
    const msg = args.map(a => {
      if (a instanceof Error) return `${a.name}: ${a.message}${a.stack ? "\n" + a.stack : ""}`;
      if (typeof a === "object") {
        try { return JSON.stringify(a); } catch { return String(a); }
      }
      return String(a);
    }).join(" ");
    window.__jnjDebugLog.push(`[${stamp}] [${kind}] ${msg}`);
    if (window.__jnjDebugLog.length > 500) window.__jnjDebugLog.shift();
  } catch {}
}
// Mirror console.log/warn/error into the debug log without breaking the console.
["log", "warn", "error", "info"].forEach(k => {
  const orig = console[k].bind(console);
  console[k] = (...args) => { jnjLog(k.toUpperCase(), ...args); orig(...args); };
});
// Catch unhandled promise rejections and top-level errors.
window.addEventListener("error", e => jnjLog("WINDOW-ERROR", e.message, "at", e.filename + ":" + e.lineno));
window.addEventListener("unhandledrejection", e => jnjLog("UNHANDLED", e.reason && e.reason.message || e.reason));
// Wrap fetch to log every request/response so we can see backend calls.
const __origFetch = window.fetch.bind(window);
window.fetch = async (...args) => {
  const url = typeof args[0] === "string" ? args[0] : (args[0] && args[0].url) || "?";
  const method = (args[1] && args[1].method) || "GET";
  jnjLog("FETCH-REQ", method, url);
  try {
    const resp = await __origFetch(...args);
    jnjLog("FETCH-RESP", method, url, "->", resp.status, resp.statusText);
    return resp;
  } catch (err) {
    jnjLog("FETCH-FAIL", method, url, "->", err.message);
    throw err;
  }
};
jnjLog("BOOT", "v25.35 boot. BUILD_ID:", "2026-08-25-portrait-safe-crop-v25.35");
const STORAGE_KEY = "retype_entries_v1";
const AUTH_KEY = "retype_authed_v1";

// Backend endpoint. Uses the same-origin proxy path when deployed via pplx sites,
// otherwise a relative path for local dev.
// Backend endpoint. On Perplexity previews, `__PORT_5000__` gets rewritten to
// a proxy path at deploy time. On Render (or any real host), no rewrite happens,
// so we fall back to a same-origin relative path.
const EXTRACT_URL = "__PORT_5000__".startsWith("__")
  ? "/api/extract-stream"
  : "__PORT_5000__/api/extract-stream";
const CSV_EXPORT_URL = "__PORT_5000__".startsWith("__")
  ? "/api/export-jnj-csv"
  : "__PORT_5000__/api/export-jnj-csv";
const JNJ_BUILD_URL = "__PORT_5000__".startsWith("__")
  ? "/api/jnj-build"
  : "__PORT_5000__/api/jnj-build";
const JNJ_BUILD_SHEET_URL = "__PORT_5000__".startsWith("__")
  ? "/api/jnj-build-sheet"
  : "__PORT_5000__/api/jnj-build-sheet";
const JNJ_MATCH_PHOTOS_URL = "__PORT_5000__".startsWith("__")
  ? "/api/jnj-match-photos"
  : "__PORT_5000__/api/jnj-match-photos";
// v18: AI is back in the photo path — gpt-4o-mini yes/no per photo (~250ms).
// Photos in a batch run concurrently on the server, so what matters is how
// many photos we let hit the API at once. 10 per batch, 3 batches concurrent
// = 30 in flight, well under OpenAI's rate limit and Render's 512MB RAM.
const JNJ_PHOTO_BATCH_SIZE = 10;
const JNJ_PHOTO_CONCURRENCY = 3; // Number of batches to run in parallel.
const JNJ_REMATCH_URL = "__PORT_5000__".startsWith("__")
  ? "/api/jnj-rematch"
  : "__PORT_5000__/api/jnj-rematch";
const JNJ_ZIP_URL = "__PORT_5000__".startsWith("__")
  ? "/api/jnj-zip"
  : "__PORT_5000__/api/jnj-zip";

// Remember the last-used sale name and seller ID prefix so the user doesn't
// have to retype for every entry in the same session.
const CSV_PREFS_KEY = "retype_csv_prefs_v1";
function loadCsvPrefs() {
  try { return JSON.parse(localStorage.getItem(CSV_PREFS_KEY) || "{}"); }
  catch { return {}; }
}
function saveCsvPrefs(prefs) {
  try { localStorage.setItem(CSV_PREFS_KEY, JSON.stringify(prefs)); } catch {}
}

/* -------------------- DOM refs -------------------- */
const lockScreen = document.getElementById("lockScreen");
const lockForm = document.getElementById("lockForm");
const passwordInput = document.getElementById("passwordInput");
const lockError = document.getElementById("lockError");
const app = document.getElementById("app");

const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const processingTray = document.getElementById("processingTray");
const processingList = document.getElementById("processingList");

const entriesGrid = document.getElementById("entriesGrid");
const emptyState = document.getElementById("emptyState");
const entryCount = document.getElementById("entryCount");

const lockBtn = document.getElementById("lockBtn");
const clearAllBtn = document.getElementById("clearAllBtn");
const toastEl = document.getElementById("toast");

/* -------------------- Storage helpers -------------------- */
function loadEntries() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
  } catch { return []; }
}
function saveEntries(entries) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
  } catch (e) {
    toast("Storage full — can't save more entries", "error");
  }
}
function addEntry(entry) {
  const entries = loadEntries();
  entries.push(entry);
  saveEntries(entries);
  return entries;
}
function updateEntryText(id, newText) {
  const entries = loadEntries();
  const idx = entries.findIndex(e => e.id === id);
  if (idx >= 0) {
    entries[idx].text = newText;
    saveEntries(entries);
  }
}
function deleteEntry(id) {
  const entries = loadEntries().filter(e => e.id !== id);
  saveEntries(entries);
  render();
}

/* -------------------- Auth -------------------- */
// Always require password on fresh page loads. No persistent auth.
function tryAuth() {
  // no-op: user must enter password every time the page loads
}
function unlock() {
  lockScreen.classList.add("hidden");
  app.classList.remove("hidden");
  render();
}
function lock() {
  app.classList.add("hidden");
  lockScreen.classList.remove("hidden");
  passwordInput.value = "";
  passwordInput.focus();
}

lockForm.addEventListener("submit", (e) => {
  e.preventDefault();
  if (passwordInput.value === PASSWORD) {
    lockError.textContent = "";
    unlock();
  } else {
    lockError.textContent = "Incorrect password";
    passwordInput.select();
  }
});

lockBtn.addEventListener("click", lock);

let clearConfirming = false;
clearAllBtn.addEventListener("click", () => {
  if (loadEntries().length === 0) { toast("Nothing to clear"); return; }
  if (clearConfirming) {
    saveEntries([]);
    render();
    toast("All entries cleared");
    clearAllBtn.textContent = "Clear all";
    clearConfirming = false;
    return;
  }
  clearConfirming = true;
  clearAllBtn.textContent = "Click again to confirm";
  setTimeout(() => {
    clearConfirming = false;
    clearAllBtn.textContent = "Clear all";
  }, 2500);
});

/* -------------------- File handling -------------------- */
dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener("change", (e) => {
  handleFiles(Array.from(e.target.files));
  fileInput.value = "";
});

["dragenter", "dragover"].forEach(ev =>
  dropZone.addEventListener(ev, (e) => {
    e.preventDefault(); e.stopPropagation();
    dropZone.classList.add("drag-over");
  })
);
["dragleave", "drop"].forEach(ev =>
  dropZone.addEventListener(ev, (e) => {
    e.preventDefault(); e.stopPropagation();
    dropZone.classList.remove("drag-over");
  })
);
dropZone.addEventListener("drop", (e) => {
  const files = Array.from(e.dataTransfer.files || []);
  if (files.length) handleFiles(files);
});

async function handleFiles(files) {
  if (!files.length) return;
  processingTray.classList.remove("hidden");
  const items = files.map(file => {
    const li = document.createElement("li");
    li.innerHTML = `
      <span class="spinner"></span>
      <span class="name"></span>
      <span class="status">preparing…</span>`;
    li.querySelector(".name").textContent = file.name;
    processingList.appendChild(li);
    return { file, li };
  });

  // Process files sequentially at the file level, but each file's pages run in parallel on the server.
  for (const { file, li } of items) {
    const statusEl = li.querySelector(".status");
    try {
      const text = await extractTextStream(file, (msg) => { statusEl.textContent = msg; });
      const entry = {
        id: crypto.randomUUID ? crypto.randomUUID() : String(Date.now() + Math.random()),
        name: file.name,
        type: file.type || guessType(file.name),
        size: file.size,
        text: text.trim() || "(no text found in this file)",
        createdAt: new Date().toISOString(),
      };
      addEntry(entry);
      li.classList.add("done");
      li.querySelector(".spinner").outerHTML =
        `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="color: var(--success); flex-shrink:0;"><polyline points="20 6 9 17 4 12"/></svg>`;
      statusEl.textContent = "done";
    } catch (err) {
      console.error(err);
      li.classList.add("error");
      li.querySelector(".spinner").outerHTML =
        `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="color: var(--danger); flex-shrink:0;"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>`;
      statusEl.textContent = err.message || "failed";
    }
    render();
  }

  // Auto-hide tray after a moment (all items finished)
  setTimeout(() => {
    processingList.innerHTML = "";
    processingTray.classList.add("hidden");
  }, 4500);
}

function guessType(name) {
  const ext = name.split(".").pop().toLowerCase();
  const map = {
    pdf: "application/pdf",
    docx: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    doc: "application/msword",
    txt: "text/plain",
    png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg",
    gif: "image/gif", bmp: "image/bmp", webp: "image/webp",
  };
  return map[ext] || "";
}

/* -------------------- Text extractor (streaming backend OCR) -------------------- */
async function extractTextStream(file, onProgress) {
  const fd = new FormData();
  fd.append("file", file);

  let res;
  try {
    res = await fetch(EXTRACT_URL, { method: "POST", body: fd });
  } catch (e) {
    throw new Error("network error");
  }
  if (!res.ok || !res.body) {
    throw new Error(`server error (${res.status})`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalText = "";
  let totalPages = 0;
  let totalNonBlank = 0;
  let blankCount = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let idx;
    while ((idx = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, idx).trim();
      buffer = buffer.slice(idx + 1);
      if (!line) continue;
      let evt;
      try { evt = JSON.parse(line); } catch { continue; }

      if (evt.type === "start") {
        totalPages = evt.pages;
        onProgress(totalPages > 1 ? `checking ${totalPages} pages…` : "reading…");
      } else if (evt.type === "skip") {
        blankCount++;
      } else if (evt.type === "done_page") {
        totalNonBlank = evt.total_nonblank;
        const skipMsg = blankCount ? ` (${blankCount} blank skipped)` : "";
        onProgress(`page ${evt.completed} of ${totalNonBlank} done${skipMsg}`);
      } else if (evt.type === "final") {
        finalText = evt.text || "";
      } else if (evt.type === "error") {
        throw new Error(evt.message || "extraction failed");
      }
    }
  }
  return finalText;
}

/* -------------------- Rendering -------------------- */
function render() {
  const entries = loadEntries();
  entryCount.textContent = entries.length;
  entriesGrid.innerHTML = "";
  if (entries.length === 0) {
    emptyState.classList.remove("hidden");
    return;
  }
  emptyState.classList.add("hidden");
  // Newest at the bottom (as requested — "boxes appear at the bottom of recent ones")
  entries.forEach(entry => entriesGrid.appendChild(entryCard(entry)));
}

/* -------------------- Auction item parsing -------------------- */
// Backend now outputs ONLY item lines. Each line = one auction item that
// starts with the item number then the description.
function parseAuctionItems(text) {
  if (!text) return { header: "", items: [] };
  const items = [];
  const ITEM_RE = /^[A-Z0-9]*\d{3,}[A-Z0-9]*\s/;
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) continue;
    if (/^---\s*Page/i.test(line)) continue;
    if (ITEM_RE.test(line)) {
      items.push({ description: line });
    } else if (items.length) {
      // Continuation of the previous item's description
      items[items.length - 1].description += " " + line;
    }
    // If it doesn't match and there's no previous item yet, silently drop it
  }
  return { header: "", items };
}

function auctionItemHtml(item, idx) {
  const description = escapeHtml(item.description || "");
  const descAttr = (item.description || "").replace(/"/g, "&quot;");
  // Pull the leading item number for the pill on the left.
  const numMatch = (item.description || "").match(/^([A-Z0-9]+)\b/);
  const numLabel = numMatch ? numMatch[1] : `#${idx+1}`;
  return `
    <div class="auction-item">
      <div class="ai-num">${escapeHtml(numLabel)}</div>
      <div class="ai-fields">
        <div class="ai-field">
          <div class="ai-label">
            <span>DESCRIPTION</span>
            <button class="field-copy" data-copy="${descAttr}" title="Copy description" type="button">Copy</button>
          </div>
          <div class="ai-value ai-desc">${description || '<em>(empty)</em>'}</div>
        </div>
      </div>
    </div>
  `;
}

function entryCard(entry) {
  const card = document.createElement("article");
  card.className = "entry-card";
  card.dataset.id = entry.id;

  const iconSvg = fileIconSvg(entry.type, entry.name);
  const date = formatDate(entry.createdAt);

  const parsed = parseAuctionItems(entry.text);
  const itemsHtml = parsed.items.length
    ? parsed.items.map((it, idx) => auctionItemHtml(it, idx)).join("")
    : `<div class="auction-empty">No item rows detected. Raw text below.</div>`;
  const headerHtml = parsed.header ? `<div class="auction-header">${escapeHtml(parsed.header)}</div>` : "";

  card.innerHTML = `
    <div class="entry-head">
      <div class="entry-icon">${iconSvg}</div>
      <div class="entry-meta">
        <div class="entry-name" title="${escapeHtml(entry.name)}">${escapeHtml(entry.name)}</div>
        <div class="entry-date">${date} · ${parsed.items.length} item${parsed.items.length===1?"":"s"}</div>
      </div>
      <button class="entry-delete" title="Delete entry" aria-label="Delete entry">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="3 6 5 6 21 6"/>
          <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
          <path d="M10 11v6M14 11v6"/>
          <path d="M9 6V4a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2"/>
        </svg>
      </button>
    </div>
    <div class="entry-body">
      ${headerHtml}
      <div class="auction-items">${itemsHtml}</div>
      <details class="raw-toggle">
        <summary>Show raw text (one item per line)</summary>
        <textarea class="entry-text" spellcheck="false"></textarea>
      </details>
    </div>
    <div class="entry-actions">
      <button class="action-btn primary" data-act="copy">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
        Copy
      </button>
      <button class="action-btn" data-act="csv">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="8" y1="13" x2="16" y2="13"/><line x1="8" y1="17" x2="16" y2="17"/></svg>
        JnJ CSV
      </button>
      <button class="action-btn" data-act="email">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/><polyline points="22,6 12,13 2,6"/></svg>
        Email
      </button>
      <button class="action-btn" data-act="print">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 6 2 18 2 18 9"/><path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"/><rect x="6" y="14" width="12" height="8"/></svg>
        Print
      </button>
      <button class="action-btn danger" data-act="delete">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6M14 11v6"/><path d="M9 6V4a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2"/></svg>
        Delete
      </button>
    </div>
  `;

  // Wire up
  const textarea = card.querySelector(".entry-text");
  textarea.value = entry.text;
  textarea.addEventListener("input", () => {
    updateEntryText(entry.id, textarea.value);
  });

  // Per-field copy buttons on each auction item
  card.querySelectorAll(".field-copy").forEach(btn => {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      const val = btn.getAttribute("data-copy") || "";
      copyText(val);
    });
  });

  // Small trash icon in header — no confirm, immediate delete (works even in iframes)
  card.querySelector(".entry-delete").addEventListener("click", () => {
    deleteEntry(entry.id);
    toast("Entry deleted");
  });

  card.querySelectorAll(".action-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const act = btn.dataset.act;
      const text = textarea.value;
      if (act === "copy") copyText(text);
      else if (act === "csv") downloadJnjCsv(text, entry.name);
      else if (act === "email") emailText(entry.name, text);
      else if (act === "print") printCard(card);
      else if (act === "delete") {
        // Delete with a lightweight inline confirm using button state
        if (btn.dataset.confirming === "1") {
          deleteEntry(entry.id);
          toast("Entry deleted");
        } else {
          btn.dataset.confirming = "1";
          const original = btn.innerHTML;
          btn.innerHTML = "Click again to confirm";
          setTimeout(() => {
            btn.dataset.confirming = "";
            btn.innerHTML = original;
          }, 2500);
        }
      }
    });
  });

  return card;
}

function fileIconSvg(type, name) {
  const n = (name || "").toLowerCase();
  if ((type || "").startsWith("image/") || /\.(png|jpe?g|gif|bmp|webp)$/.test(n)) {
    return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>`;
  }
  if ((type || "").includes("pdf") || n.endsWith(".pdf")) {
    return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>`;
  }
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="9" y1="13" x2="15" y2="13"/><line x1="9" y1="17" x2="15" y2="17"/></svg>`;
}

/* -------------------- Actions -------------------- */
async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    toast("Copied to clipboard");
  } catch {
    // Fallback
    const ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    document.body.removeChild(ta);
    toast("Copied to clipboard");
  }
}

function emailText(name, text) {
  const subject = encodeURIComponent(`Retyped: ${name}`);
  // mailto has practical length limits — cap body at ~1800 chars
  const capped = text.length > 1800
    ? text.slice(0, 1800) + "\n\n[…truncated — copy full text from the workspace]"
    : text;
  const body = encodeURIComponent(capped);
  window.location.href = `mailto:?subject=${subject}&body=${body}`;
}

async function downloadJnjCsv(transcript, sourceName) {
  const prefs = loadCsvPrefs();
  const dialog = showCsvDialog(prefs);
  const result = await dialog.result;
  if (!result) return; // user cancelled

  const { saleName, sellerId, sellerStart } = result;
  saveCsvPrefs({ saleName, sellerId, sellerStart });

  const fd = new FormData();
  fd.append("transcript", transcript);
  fd.append("sale_name", saleName);
  fd.append("seller_id", sellerId);
  fd.append("seller_start", String(sellerStart));

  let res;
  try {
    res = await fetch(CSV_EXPORT_URL, { method: "POST", body: fd });
  } catch (e) {
    toast("Network error — could not build CSV");
    return;
  }
  if (!res.ok) {
    let msg = `CSV export failed (${res.status})`;
    try { const j = await res.json(); if (j.detail) msg = j.detail; } catch {}
    toast(msg);
    return;
  }

  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  // Try to extract filename from Content-Disposition header
  const disp = res.headers.get("content-disposition") || "";
  const m = disp.match(/filename="([^"]+)"/);
  a.download = m ? m[1] : `jnj-export.csv`;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, 100);
  toast("CSV downloaded");
}

function showCsvDialog(prefs) {
  // Build a modal dialog for sale name + seller ID input
  const backdrop = document.createElement("div");
  backdrop.className = "csv-dialog-backdrop";
  backdrop.innerHTML = `
    <div class="csv-dialog">
      <h3>Download JnJ CSV</h3>
      <p class="csv-dialog-sub">Fill these in to build the JnJ import file. Values are remembered for next time.</p>
      <label class="csv-field">
        <span>Sale name</span>
        <input type="text" id="csvSaleName" placeholder="e.g. APRIL 16 ~ Q SALE" value="${escapeAttr(prefs.saleName || '')}" />
        <small>Goes in the Category column of the CSV</small>
      </label>
      <label class="csv-field">
        <span>Seller ID prefix</span>
        <input type="text" id="csvSellerId" placeholder="e.g. AA" value="${escapeAttr(prefs.sellerId || 'AA')}" />
        <small>Combined with the starting number to make cf_SellerID (e.g. AA1961)</small>
      </label>
      <label class="csv-field">
        <span>Starting number</span>
        <input type="number" id="csvSellerStart" min="0" step="1" value="${escapeAttr(prefs.sellerStart || 1000)}" />
        <small>First row gets this number, then it counts up (1000, 1001, 1002...)</small>
      </label>
      <div class="csv-dialog-actions">
        <button type="button" class="action-btn" id="csvCancel">Cancel</button>
        <button type="button" class="action-btn primary" id="csvOk">Download CSV</button>
      </div>
    </div>
  `;
  document.body.appendChild(backdrop);

  const saleInput = backdrop.querySelector("#csvSaleName");
  const sellerInput = backdrop.querySelector("#csvSellerId");
  const startInput = backdrop.querySelector("#csvSellerStart");
  setTimeout(() => saleInput.focus(), 30);

  const result = new Promise((resolve) => {
    const cleanup = () => backdrop.remove();
    backdrop.querySelector("#csvCancel").addEventListener("click", () => {
      cleanup(); resolve(null);
    });
    backdrop.querySelector("#csvOk").addEventListener("click", () => {
      const saleName = saleInput.value.trim();
      const sellerId = sellerInput.value.trim().toUpperCase();
      const sellerStart = parseInt(startInput.value, 10) || 1000;
      cleanup();
      resolve({ saleName, sellerId, sellerStart });
    });
    backdrop.addEventListener("click", (e) => {
      if (e.target === backdrop) { cleanup(); resolve(null); }
    });
    backdrop.addEventListener("keydown", (e) => {
      if (e.key === "Escape") { cleanup(); resolve(null); }
      else if (e.key === "Enter" && e.target.tagName === "INPUT") {
        backdrop.querySelector("#csvOk").click();
      }
    });
  });

  return { result };
}

function escapeAttr(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}

function printCard(card) {
  card.classList.add("print-target");
  // Sync textarea value into DOM for print (textarea .value doesn't print reliably)
  const ta = card.querySelector(".entry-text");
  const origRows = ta.getAttribute("rows");
  ta.setAttribute("rows", Math.max(6, ta.value.split("\n").length + 2));
  window.print();
  setTimeout(() => {
    card.classList.remove("print-target");
    if (origRows) ta.setAttribute("rows", origRows);
    else ta.removeAttribute("rows");
  }, 500);
}

/* -------------------- Utilities -------------------- */
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}
function formatDate(iso) {
  const d = new Date(iso);
  const opts = { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" };
  return d.toLocaleString(undefined, opts);
}
let toastTimer;
function toast(msg) {
  toastEl.textContent = msg;
  toastEl.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove("show"), 2000);
}

/* =========================================================
   JnJ Sale Builder
   - Second dropzone that takes a sheet + photos together
   - Sends everything to /api/jnj-build
   - Renders a preview with per-item photo assignments
   - User can retry AI match, drag photos between items,
     or drop a photo back onto the "unmatched" pool
   - Download button POSTs to /api/jnj-zip → downloads zip
========================================================= */

const jnjDropZone = document.getElementById("jnjDropZone");
const jnjFileInput = document.getElementById("jnjFileInput");
const jnjPreview = document.getElementById("jnjPreview");
const jnjItemsList = document.getElementById("jnjItemsList");
const jnjUnmatched = document.getElementById("jnjUnmatched");
const jnjSaleName = document.getElementById("jnjSaleName");
const jnjSellerId = document.getElementById("jnjSellerId");
const jnjSellerStart = document.getElementById("jnjSellerStart");
const jnjPreviewSub = document.getElementById("jnjPreviewSub");
const jnjDownloadBtn = document.getElementById("jnjDownloadBtn");
const jnjCancelBtn = document.getElementById("jnjCancelBtn");

// State: current build session in memory
let jnjState = null;
// Shape:
//   items: [{item_num, lot_code, description}]
//   photos: Map<filename, {file: File, thumb: string, tag_read, description_read, item_num_match, match_kind}>
//   itemPhotos: { item_num: [filename, ...] }  // ordered
//   unmatched: [filename, ...]

function jnjResetState() {
  jnjState = null;
  jnjPreview.classList.add("hidden");
  jnjItemsList.innerHTML = "";
  jnjUnmatched.innerHTML = "<em>none</em>";
}

function jnjRestorePrefs() {
  const prefs = loadCsvPrefs();
  if (prefs.sale_name) jnjSaleName.value = prefs.sale_name;
  if (prefs.seller_id) jnjSellerId.value = prefs.seller_id;
  if (prefs.seller_start) jnjSellerStart.value = prefs.seller_start;
}

function jnjSavePrefs() {
  saveCsvPrefs({
    sale_name: jnjSaleName.value.trim(),
    seller_id: jnjSellerId.value.trim(),
    seller_start: jnjSellerStart.value.trim() || "1000",
  });
}

/* ---- STAGING WORKFLOW ----
   iOS Safari can only pick photos OR files in a single picker session, not
   both. So we let the user stage the sheet and the photos in TWO separate
   picks, watch them accumulate in a preview list, then hit "Build sale" when
   they're ready. Drag-and-drop still works for desktop — dropped files just
   get merged into the staged list instead of firing the API immediately.
*/

const jnjSheetInput = document.getElementById("jnjSheetInput");
const jnjPhotosInput = document.getElementById("jnjPhotosInput");
const jnjAddSheetBtn = document.getElementById("jnjAddSheetBtn");
const jnjAddPhotosBtn = document.getElementById("jnjAddPhotosBtn");
const jnjBuildBtn = document.getElementById("jnjBuildBtn");
const jnjClearStagedBtn = document.getElementById("jnjClearStagedBtn");
const jnjStagedList = document.getElementById("jnjStagedList");
const jnjStageActions = document.getElementById("jnjStageActions");

// Keep only real image files. Drops:
//   - zero-byte entries (Windows "folder" drops)
//   - dotfiles and macOS .DS_Store / Thumbs.db
//   - non-image mime types (PDFs are handled separately as the sheet)
function isUsablePhoto(f) {
  if (!f) return false;
  if (!f.size || f.size <= 0) return false;
  const name = (f.name || "").toLowerCase();
  if (name.startsWith(".")) return false;
  if (name === "thumbs.db" || name === "desktop.ini") return false;
  const type = (f.type || "").toLowerCase();
  if (type.startsWith("image/")) return true;
  // Fallback for files with no mime type — check extension
  return /\.(jpe?g|png|heic|heif|webp|gif|bmp|tiff?)$/i.test(name);
}

// Staged files live here until the user taps Build.
// v23: jnjStaged.sheets is now an ARRAY so Dave can stage multiple hand-drawn
// intake sheets in one batch. Each sheet gets its own boxed seller # applied
// to its own items downstream.
let jnjStaged = { sheets: [], photos: [] };

// v25.27: staged-list is now COLLAPSIBLE. When Ashley stages 200+ photos
// the individual rows push the Build button off-screen. We show only the
// summary chip + Build/Clear buttons by default, and hide the file list
// behind a "Show files" toggle. Persist the open/closed state on window
// so it survives re-renders.
if (typeof window.__jnjStagedOpen === "undefined") window.__jnjStagedOpen = false;

function jnjRenderStaged() {
  const hasAny = jnjStaged.sheets.length > 0 || jnjStaged.photos.length > 0;
  jnjStagedList.style.display = hasAny ? "block" : "none";
  jnjStageActions.style.display = hasAny ? "flex" : "none";

  const rows = [];
  jnjStaged.sheets.forEach((s, i) => {
    const label = jnjStaged.sheets.length > 1 ? `SHEET ${i+1}` : "SHEET";
    rows.push(`<div class="jnj-staged-row" style="display:flex;align-items:center;gap:10px;padding:8px 12px;background:rgba(147,112,255,0.08);border:1px solid rgba(147,112,255,0.25);border-radius:8px;margin-bottom:6px;">
      <span style="font-size:11px;font-weight:600;color:#9370ff;letter-spacing:0.5px;">${label}</span>
      <span style="flex:1;font-size:13px;color:var(--text);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escapeHtml(s.name || "(no name)")}</span>
      <span style="font-size:11px;color:var(--muted);">${(s.size/1024).toFixed(0)} KB</span>
      <button type="button" data-remove-sheet="${i}" class="ghost-btn" style="padding:2px 8px;font-size:12px;">Remove</button>
    </div>`);
  });
  jnjStaged.photos.forEach((p, i) => {
    rows.push(`<div class="jnj-staged-row" style="display:flex;align-items:center;gap:10px;padding:6px 12px;font-size:13px;">
      <span style="font-size:11px;color:var(--muted);width:32px;flex-shrink:0;">${i+1}.</span>
      <span style="flex:1;color:var(--text);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escapeHtml(p.name || "photo")}</span>
      <span style="font-size:11px;color:var(--muted);">${(p.size/1024).toFixed(0)} KB</span>
      <button type="button" data-remove-photo="${i}" class="ghost-btn" style="padding:2px 8px;font-size:12px;">Remove</button>
    </div>`);
  });

  if (hasAny) {
    const total = jnjStaged.sheets.length + jnjStaged.photos.length;
    const missing = [];
    if (jnjStaged.sheets.length === 0) missing.push("at least 1 sheet");
    if (jnjStaged.photos.length === 0) missing.push("at least 1 photo");

    const summary = missing.length
      ? `<div style="font-size:13px;color:#ff9a3d;font-weight:600;">Still need: ${missing.join(" and ")}</div>`
      : `<div style="font-size:13px;color:#3ecf8e;font-weight:600;">Ready — ${jnjStaged.sheets.length} sheet${jnjStaged.sheets.length === 1 ? "" : "s"} + ${jnjStaged.photos.length} photo${jnjStaged.photos.length === 1 ? "" : "s"} (${total} files)</div>`;

    const isOpen = window.__jnjStagedOpen;
    const toggleLabel = isOpen ? `▼ Hide files` : `▶ Show files (${total})`;

    const listStyle = isOpen
      ? "max-height:280px;overflow-y:auto;margin-top:10px;padding:6px;background:rgba(255,255,255,0.02);border-radius:8px;border:1px solid var(--border);"
      : "display:none;";

    jnjStagedList.innerHTML = `
      <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;">
        ${summary}
        <button type="button" id="jnjToggleFiles" class="ghost-btn" style="padding:4px 10px;font-size:12px;flex-shrink:0;">${toggleLabel}</button>
      </div>
      <div id="jnjStagedFilesList" style="${listStyle}">${rows.join("")}</div>
    `;

    const toggleBtn = jnjStagedList.querySelector("#jnjToggleFiles");
    if (toggleBtn) {
      toggleBtn.addEventListener("click", (e) => {
        e.preventDefault();
        window.__jnjStagedOpen = !window.__jnjStagedOpen;
        jnjRenderStaged();
      });
    }
  }
  jnjBuildBtn.disabled = !(jnjStaged.sheets.length > 0 && jnjStaged.photos.length > 0);
  jnjBuildBtn.style.opacity = jnjBuildBtn.disabled ? "0.5" : "1";
}

// Remove buttons on staged rows
jnjStagedList.addEventListener("click", (e) => {
  const btn = e.target.closest("button");
  if (!btn) return;
  if (btn.dataset.removeSheet !== undefined) {
    const idx = parseInt(btn.dataset.removeSheet, 10);
    if (!isNaN(idx)) {
      jnjStaged.sheets.splice(idx, 1);
      jnjRenderStaged();
    }
  } else if (btn.dataset.removePhoto !== undefined) {
    const idx = parseInt(btn.dataset.removePhoto, 10);
    if (!isNaN(idx)) {
      jnjStaged.photos.splice(idx, 1);
      jnjRenderStaged();
    }
  }
});

// Wire the two staged inputs
jnjAddSheetBtn.addEventListener("click", (e) => {
  e.preventDefault();
  jnjSheetInput.click();
});
jnjAddPhotosBtn.addEventListener("click", (e) => {
  e.preventDefault();
  jnjPhotosInput.click();
});

jnjSheetInput.addEventListener("change", (e) => {
  // v23: accept multiple sheets. Dedupe by name+size so a user re-selecting
  // the same file twice doesn't add it a second time.
  const files = Array.from(e.target.files || []);
  for (const f of files) {
    if (!jnjStaged.sheets.some(s => s.name === f.name && s.size === f.size)) {
      jnjStaged.sheets.push(f);
    }
  }
  jnjRenderStaged();
  jnjSheetInput.value = "";
});

function jnjIngestPhotos(fileList) {
  const raw = Array.from(fileList || []);
  const kept = raw.filter(isUsablePhoto);
  const dropped = raw.length - kept.length;
  for (const f of kept) {
    // Avoid duplicates by name+size
    if (!jnjStaged.photos.some(p => p.name === f.name && p.size === f.size)) {
      jnjStaged.photos.push(f);
    }
  }
  jnjRenderStaged();
  if (dropped > 0) {
    toast(`Added ${kept.length} photos (skipped ${dropped} non-image or empty file${dropped === 1 ? "" : "s"}).`);
  }
}

jnjPhotosInput.addEventListener("change", (e) => {
  jnjIngestPhotos(e.target.files);
  jnjPhotosInput.value = "";
});

jnjClearStagedBtn.addEventListener("click", () => {
  jnjStaged = { sheets: [], photos: [] };
  jnjRenderStaged();
});

jnjBuildBtn.addEventListener("click", () => {
  if (jnjStaged.sheets.length === 0 || jnjStaged.photos.length === 0) return;
  // v23: hand jnjHandleFiles a struct so we don't have to guess sheets-vs-photos
  // by filename or size. The staged UI already knows which are sheets.
  jnjHandleFiles({ sheets: [...jnjStaged.sheets], photos: [...jnjStaged.photos] });
});

// Drag-and-drop desktop path — dropped files merge into staged list.
// v23: .pdf files (any number) go into the sheets array; images go into photos.
// If Dave drops phone photos of handwritten sheets (no PDFs), he needs to use
// the "1. Add sheet(s)" button to tell us which JPEGs are sheets vs items.
["dragenter", "dragover"].forEach(ev =>
  jnjDropZone.addEventListener(ev, (e) => {
    e.preventDefault(); e.stopPropagation();
    if (!e.dataTransfer.types.includes("application/x-jnj-photo")) {
      jnjDropZone.classList.add("drag-over");
    }
  })
);
["dragleave", "drop"].forEach(ev =>
  jnjDropZone.addEventListener(ev, (e) => {
    e.preventDefault(); e.stopPropagation();
    jnjDropZone.classList.remove("drag-over");
  })
);
// Read ALL entries from a directory reader. Chrome returns ~100 per call,
// so we have to keep calling readEntries() until it returns an empty array.
// CRITICAL: the entry objects become invalid once the drop event handler
// returns control to the browser, so we must aggressively read everything
// before doing any other async work.
function readAllEntries(reader) {
  return new Promise((resolve, reject) => {
    const all = [];
    const readBatch = () => {
      reader.readEntries(
        (entries) => {
          if (!entries.length) {
            resolve(all);
          } else {
            all.push(...entries);
            // Recurse immediately — no await between calls, no delay.
            readBatch();
          }
        },
        (err) => reject(err)
      );
    };
    readBatch();
  });
}

// Recursively walk a dropped folder via FileSystemEntry API.
// Uses readAllEntries() to defeat the 100-per-batch limit.
async function walkEntry(entry, out) {
  if (!entry) return;
  if (entry.isFile) {
    await new Promise((res) => entry.file((f) => { out.push(f); res(); }, () => res()));
  } else if (entry.isDirectory) {
    const reader = entry.createReader();
    let entries;
    try {
      entries = await readAllEntries(reader);
    } catch {
      entries = [];
    }
    for (const child of entries) await walkEntry(child, out);
  }
}

jnjDropZone.addEventListener("drop", async (e) => {
  if (e.dataTransfer.types.includes("application/x-jnj-photo")) return;

  // Try FileSystemEntry path first — this expands dropped folders.
  // IMPORTANT: grab all entries synchronously before any await, otherwise
  // Chrome invalidates the DataTransferItem list.
  const items = e.dataTransfer.items ? Array.from(e.dataTransfer.items) : [];
  const entriesFirst = [];
  const fallbackFiles = Array.from(e.dataTransfer.files || []);
  for (const it of items) {
    if (it.kind !== "file") continue;
    const entry = it.webkitGetAsEntry && it.webkitGetAsEntry();
    if (entry) entriesFirst.push(entry);
  }

  const collected = [];
  const usedEntryApi = entriesFirst.length > 0;
  for (const entry of entriesFirst) {
    await walkEntry(entry, collected);
  }
  const files = usedEntryApi ? collected : fallbackFiles;
  if (!files.length) return;

  let droppedCount = 0;
  for (const f of files) {
    const name = (f.name || "").toLowerCase();
    const isPdf = name.endsWith(".pdf") || f.type === "application/pdf";
    if (isPdf) {
      // v23: allow multiple PDF sheets, dedupe by name+size.
      if (!jnjStaged.sheets.some(s => s.name === f.name && s.size === f.size)) {
        jnjStaged.sheets.push(f);
      }
    } else if (isUsablePhoto(f)) {
      if (!jnjStaged.photos.some(p => p.name === f.name && p.size === f.size)) {
        jnjStaged.photos.push(f);
      }
    } else {
      droppedCount++;
    }
  }
  jnjRenderStaged();
  // Always confirm how many photos landed — makes silent losses obvious.
  const total = files.length;
  const staged = jnjStaged.photos.length;
  const summary = `Received ${total} file${total === 1 ? "" : "s"}, staged ${staged} photo${staged === 1 ? "" : "s"}${droppedCount > 0 ? `, skipped ${droppedCount} non-image` : ""}.`;
  toast(summary);
});

// Fallback legacy input — keep for anything still referencing jnjFileInput.
jnjFileInput.addEventListener("change", (e) => {
  const files = Array.from(e.target.files || []);
  for (const f of files) {
    const name = (f.name || "").toLowerCase();
    const isPdf = name.endsWith(".pdf") || f.type === "application/pdf";
    if (isPdf) {
      if (!jnjStaged.sheets.some(s => s.name === f.name && s.size === f.size)) {
        jnjStaged.sheets.push(f);
      }
    } else if (!jnjStaged.photos.some(p => p.name === f.name && p.size === f.size)) {
      jnjStaged.photos.push(f);
    }
  }
  jnjRenderStaged();
  jnjFileInput.value = "";
});

// Initial render (disables Build until files staged)
jnjRenderStaged();

// Stamp the build ID as a floating badge so we can visually confirm the deploy
// landed — both in the footer (small) AND as a corner badge (impossible to miss).
try {
  const footer = document.querySelector(".app-footer");
  if (footer) {
    const stamp = document.createElement("div");
    stamp.style.cssText = "margin-top:6px;font-size:11px;opacity:0.5;";
    stamp.textContent = `Build: ${BUILD_ID}`;
    footer.appendChild(stamp);
  }
  // Also add a fixed-position badge in the bottom-right corner so it's
  // visible from any part of the page without scrolling.
  const badge = document.createElement("div");
  badge.id = "buildIdBadge";
  badge.style.cssText = "position:fixed;bottom:8px;right:8px;z-index:9998;background:rgba(0,0,0,0.75);color:#7fff9f;padding:6px 10px;border-radius:6px;font-size:11px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-weight:600;letter-spacing:0.02em;pointer-events:none;box-shadow:0 2px 8px rgba(0,0,0,0.3);";
  badge.textContent = `v25.35 · ${BUILD_ID}`;
  // v24: clicking the badge opens the debug log overlay — same as the error
  // banner button, but lets the user check the log even when things went
  // "fine" (e.g. build ran but nothing happened afterward).
  badge.style.pointerEvents = "auto";
  badge.style.cursor = "pointer";
  badge.title = "Click to view debug log";
  badge.addEventListener("click", () => jnjShowDebugOverlay());
  document.body.appendChild(badge);
} catch {}

// Health-check URL for warming up the free-tier server before doing real work.
const JNJ_HEALTH_URL = "__PORT_5000__".startsWith("__")
  ? "/api/health"
  : "__PORT_5000__/api/health";

// Wake up the Render Free-tier server before hitting a real endpoint. The
// server spins down after ~15 min of inactivity and takes up to 50s to come
// back — without this, the first Build after a long idle will 502.
async function jnjWarmupServer(statusEl) {
  const started = Date.now();
  let lastErr = null;
  for (let attempt = 1; attempt <= 6; attempt++) {
    try {
      const res = await fetch(JNJ_HEALTH_URL, { method: "GET" });
      if (res.ok) {
        const elapsed = Math.round((Date.now() - started) / 1000);
        if (elapsed > 3 && statusEl) statusEl.textContent = `server woke up in ${elapsed}s…`;
        return;
      }
    } catch (e) { lastErr = e; }
    if (statusEl) statusEl.textContent = `waking up server… (${attempt}/6, this can take up to a minute after idle)`;
    await new Promise(r => setTimeout(r, 8000));
  }
  // Non-fatal — the real call may still work.
  console.warn("warmup exceeded 6 tries:", lastErr);
}

// POST wrapper with a single retry on transient failure (502/504/network).
async function postForJson(url, formData, contextLabel, { retry = true } = {}) {
  const doFetch = async () => fetch(url, { method: "POST", body: formData });
  let res;
  try {
    res = await doFetch();
  } catch (netErr) {
    if (retry) {
      console.warn(`${contextLabel}: retrying after network error —`, netErr.message);
      await new Promise(r => setTimeout(r, 4000));
      try { res = await doFetch(); }
      catch (retryErr) {
        throw new Error(`${contextLabel}: network error (${retryErr.message || "connection lost"}). The server may be waking up — try clicking Build again.`);
      }
    } else {
      throw new Error(`${contextLabel}: network error (${netErr.message || "connection lost"}).`);
    }
  }
  // Auto-retry on 502/503/504 (server waking up or transient proxy issue).
  if (retry && (res.status === 502 || res.status === 503 || res.status === 504)) {
    console.warn(`${contextLabel}: got ${res.status}, retrying in 5s…`);
    await new Promise(r => setTimeout(r, 5000));
    try { res = await doFetch(); } catch (e) {
      throw new Error(`${contextLabel}: server error (${res.status}) then network error on retry (${e.message}).`);
    }
  }
  if (!res.ok) {
    let msg = `${contextLabel}: server error (${res.status})`;
    let bodyText = "";
    try {
      bodyText = await res.text();
      try { const j = JSON.parse(bodyText); msg = `${contextLabel}: ${j.detail || j.error || msg}`; }
      catch { if (bodyText) msg = `${contextLabel}: ${bodyText.slice(0, 400)}`; }
    } catch {}
    console.error(`${contextLabel} failed:`, res.status, bodyText);
    throw new Error(msg);
  }
  try {
    return await res.json();
  } catch (parseErr) {
    console.error(`${contextLabel}: non-JSON response`, parseErr);
    throw new Error(`${contextLabel}: server returned an invalid response.`);
  }
}

async function jnjHandleFiles(input) {
  // v23: accept either a flat file[] (legacy drop of a folder) or a
  // {sheets, photos} struct from the staged UI. The struct path is preferred
  // because it removes all sheet-vs-photo guessing.
  let sheets, photos;
  if (input && !Array.isArray(input) && input.sheets && input.photos) {
    sheets = [...input.sheets];
    photos = [...input.photos];
  } else {
    // Legacy path: split by PDF-ness, then fall back to filename/size heuristics.
    const files = Array.isArray(input) ? input : Array.from(input || []);
    if (!files.length) return;
    if (files.length < 2) {
      toast("Drop the sheet + at least one photo together.");
      return;
    }
    sheets = [];
    photos = [];
    for (const f of files) {
      const name = (f.name || "").toLowerCase();
      const isPdf = name.endsWith(".pdf") || f.type === "application/pdf";
      if (isPdf) sheets.push(f);
      else photos.push(f);
    }
    if (sheets.length === 0) {
      for (let i = photos.length - 1; i >= 0; i--) {
        const n = (photos[i].name || "").toLowerCase();
        if (/(sheet|intake)/.test(n) || n.startsWith("file")) {
          sheets.unshift(photos.splice(i, 1)[0]);
        }
      }
    }
    if (sheets.length === 0 && photos.length > 1) {
      photos.sort((a, b) => a.size - b.size);
      sheets.push(photos.shift());
    }
    if (sheets.length === 0) {
      jnjShowErrorBanner("Couldn't identify a sheet in the upload. Please add the intake sheet as a PDF or a clear photo.");
      return;
    }
  }

  if (sheets.length === 0 || photos.length === 0) {
    jnjShowErrorBanner("Need at least 1 sheet and 1 item photo to build.");
    return;
  }

  jnjRestorePrefs();

  processingTray.classList.remove("hidden");
  processingList.innerHTML = "";
  const li = document.createElement("li");
  li.innerHTML = `<span class="spinner"></span><span class="name">JnJ Sale Builder</span><span class="status">reading sheet…</span>`;
  processingList.appendChild(li);
  const statusEl = li.querySelector(".status");

  try {
    // ---------- Step 0: wake up the server if it's cold ----------
    // Render Free tier sleeps after 15 min idle. First real request can take
    // up to 50s to wake it. Ping /api/health first so the real call is fast.
    statusEl.textContent = "checking server…";
    await jnjWarmupServer(statusEl);

    // ---------- Step 1: transcribe ALL sheets (in parallel) ----------
    // v21: each sheet contributes its own items + its own boxed seller #.
    // Every item is tagged with sheet_seller_num so the CSV builder can put
    // the right ID on each row.
    statusEl.textContent = sheets.length > 1 ? `reading ${sheets.length} sheets…` : "reading sheet…";
    // v25.13: helper that turns ONE page of transcription output (with its
    // own seller_groups) into a stamped list of items. Used both for single-
    // image uploads and for each PAGE of a multi-page PDF.
    const stampPage = (pageObj, virtualSheetIdx, filename, totalSheets) => {
      const sellerNum = (pageObj.seller_number || "").trim();
      let sellerGroups = Array.isArray(pageObj.seller_groups) ? pageObj.seller_groups.slice() : [];
      if (!sellerGroups.length && sellerNum) {
        sellerGroups = [{ seller_num: sellerNum, first_item_num: "" }];
      }
      if (sellerGroups.length) {
        sellerGroups[0] = { ...sellerGroups[0], first_item_num: "" };
      }
      jnjLog("SHEET-RESULT", `sheet=${virtualSheetIdx + 1}/${totalSheets}`, `filename=${filename}`, `seller_number=${JSON.stringify(pageObj.seller_number)}`, `groups=${JSON.stringify(sellerGroups)}`, `items=${(pageObj.items || []).length}`);

      const rawItems = pageObj.items || [];
      const normItem = (v) => String(v || "").replace(/[^A-Za-z0-9]/g, "").toLowerCase();
      const asInt = (s) => {
        const m = /^([0-9]+)/.exec(String(s || ""));
        return m ? parseInt(m[1], 10) : NaN;
      };
      let groupIdx = sellerGroups.length > 0 ? 0 : -1;
      const itemsFromSheet = rawItems.map((it) => {
        const itemNum = normItem(it.item_num || it.LotNumber || it.lot_number || "");
        const itemInt = asInt(itemNum);
        while (groupIdx + 1 < sellerGroups.length) {
          const boundary = normItem(sellerGroups[groupIdx + 1].first_item_num);
          const boundaryInt = asInt(boundary);
          const exactMatch = boundary && itemNum === boundary;
          const numericPassed = !isNaN(itemInt) && !isNaN(boundaryInt) && itemInt >= boundaryInt;
          if (exactMatch || numericPassed) {
            groupIdx += 1;
          } else {
            break;
          }
        }
        const activeSeller = groupIdx >= 0 ? sellerGroups[groupIdx].seller_num : (sellerNum || "");
        return {
          ...it,
          sheet_seller_num: activeSeller,
          sheet_index: virtualSheetIdx,
        };
      });
      jnjLog("SHEET-STAMP", `sheet=${virtualSheetIdx + 1}`, `groups=${sellerGroups.length}`, `stamped=${itemsFromSheet.map(i => `${(i.item_num||"").toString().slice(0,6)}=${i.sheet_seller_num}`).join(",")}`);
      return { items: itemsFromSheet, seller_number: sellerNum, seller_groups: sellerGroups, filename };
    };

    // v25.13: first pass — POST each uploaded sheet-file to the backend.
    // Response may include a `pages` array (multi-page PDF). We do the flatten
    // in a second pass so numbering stays sequential across files+pages.
    const rawSheetResponses = await Promise.all(sheets.map(async (sheetFile, sIdx) => {
      const sheetFd = new FormData();
      sheetFd.append("sheet", sheetFile);
      const sd = await postForJson(JNJ_BUILD_SHEET_URL, sheetFd, `Sheet ${sIdx + 1}/${sheets.length}`);
      return { sd, sheetFile };
    }));

    // Second pass: flatten each file into 1+ virtual sheets and stamp them.
    // Count total virtual sheets first so the log shows accurate "sheet=N/M".
    let totalVirtualSheets = 0;
    for (const { sd } of rawSheetResponses) {
      if (Array.isArray(sd.pages) && sd.pages.length) {
        totalVirtualSheets += sd.pages.length;
      } else {
        totalVirtualSheets += 1;
      }
    }
    const sheetResults = [];
    let virtualIdx = 0;
    for (const { sd, sheetFile } of rawSheetResponses) {
      const pages = Array.isArray(sd.pages) && sd.pages.length
        ? sd.pages
        : [{ items: sd.items || [], seller_number: sd.seller_number || "", seller_groups: sd.seller_groups || [] }];
      pages.forEach((pageObj, pageIdx) => {
        const label = pages.length > 1 ? `${sheetFile.name} p${pageIdx + 1}` : sheetFile.name;
        sheetResults.push(stampPage(pageObj, virtualIdx, label, totalVirtualSheets));
        virtualIdx += 1;
      });
    }

    // Flatten items across all sheets, preserving order (sheet 1's items first, etc.).
    const items = [];
    const sellerNums = [];
    for (const r of sheetResults) {
      items.push(...r.items);
      for (const g of (r.seller_groups || [])) {
        if (g.seller_num) sellerNums.push(g.seller_num);
      }
    }
    if (!items.length) throw new Error("Sheets transcribed but no item rows were parsed.");

    // Auto-fill Seller ID with the FIRST detected number (display fallback
    // for items whose sheet produced no boxed # at all). Each item's own
    // sheet_seller_num is what actually goes on its CSV row.
    if (sellerNums.length > 0) {
      jnjSellerId.value = sellerNums[0];
      jnjSavePrefs();
      const uniq = [...new Set(sellerNums)];
      if (uniq.length === 1) {
        toast(`Detected seller # ${uniq[0]} across your sheet${sheets.length > 1 ? "s" : ""}.`);
      } else {
        toast(`Detected ${uniq.length} seller #s: ${uniq.join(", ")}`);
      }
    }

    statusEl.textContent = `sheets done — ${items.length} items across ${sheets.length} sheet${sheets.length > 1 ? "s" : ""}${sellerNums.length ? `, sellers ${[...new Set(sellerNums)].join(", ")}` : ""}. Matching ${photos.length} photos…`;

    // ---------- Step 2: process photos in batches, in parallel ----------
    // Batch size 8 with concurrency 3 means we're running 24 photos through
    // OpenAI simultaneously, which is well under any rate limit and dramatically
    // faster than the old sequential 3-at-a-time approach.
    const batches = [];
    for (let i = 0; i < photos.length; i += JNJ_PHOTO_BATCH_SIZE) {
      batches.push(photos.slice(i, i + JNJ_PHOTO_BATCH_SIZE));
    }
    const itemsJson = JSON.stringify(items);
    const results = new Array(batches.length); // Store by index to preserve order.
    let completedBatches = 0;

    const runBatch = async (batchIdx) => {
      const fd = new FormData();
      for (const p of batches[batchIdx]) fd.append("photos", p);
      fd.append("items_json", itemsJson);
      const batchData = await postForJson(JNJ_MATCH_PHOTOS_URL, fd, `Photo batch ${batchIdx + 1}/${batches.length}`);
      results[batchIdx] = batchData.photos || [];
      completedBatches++;
      statusEl.textContent = `matching photos… ${completedBatches} of ${batches.length} batches done`;
    };

    // Run batches with a concurrency limit — keeps at most JNJ_PHOTO_CONCURRENCY
    // requests in flight at any time. Simpler than a full pool implementation:
    // we start N workers that each pull the next available batch index.
    let nextBatch = 0;
    const worker = async () => {
      while (true) {
        const idx = nextBatch++;
        if (idx >= batches.length) return;
        await runBatch(idx);
      }
    };
    const workers = [];
    for (let w = 0; w < Math.min(JNJ_PHOTO_CONCURRENCY, batches.length); w++) {
      workers.push(worker());
    }
    await Promise.all(workers);

    // Flatten results in the original batch order and re-index photo ids.
    const allPhotoInfos = [];
    for (const batchResults of results) {
      if (!batchResults) continue;
      for (const p of batchResults) {
        p.id = `p${allPhotoInfos.length}`;
        allPhotoInfos.push(p);
      }
    }

    statusEl.textContent = `done — ${items.length} items, ${allPhotoInfos.length} photos`;
    li.querySelector(".spinner").outerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="color: var(--success); flex-shrink:0;"><polyline points="20 6 9 17 4 12"/></svg>`;
    setTimeout(() => { processingList.innerHTML = ""; processingTray.classList.add("hidden"); }, 2500);

    // ---------- Step 3: order-based fill (scene-change detection) ----------
    // Dave shoots in sheet order but photo counts per item are wildly uneven
    // (1 to 20 photos per item). Tags are rarely visible. So we detect where
    // one item ends and the next begins by comparing perceptual hashes
    // (dhashes) of consecutive photos: consecutive photos of the SAME item
    // have similar dhashes; a big Hamming-distance jump means Dave moved to a
    // new item, so we advance the cursor.
    //
    // Verified with 5 scenarios (see test_order_match.py):
    //   - 5/2/3/1/4 photos across 5 items      → all correct
    //   - 20/1/1 extreme uneven               → all correct
    //   - 1 photo per item                     → all correct
    //   - tag anchor skips an item             → tag wins, following photos follow
    //   - only one item photographed           → no false scene changes
    const itemNumByIndex = items.map(i => i.item_num);
    const indexByItemNum = new Map(items.map((i, idx) => [i.item_num, idx]));

    // Hamming distance between two 16-char hex (64-bit) dhashes.
    function hammingDist(h1, h2) {
      if (!h1 || !h2 || h1.length !== h2.length) return 32;
      // Split 16-char hex into two 8-char halves so we can use 32-bit XOR
      // (JavaScript bitwise ops are 32-bit).
      let d = 0;
      for (let k = 0; k < h1.length; k += 8) {
        const a = parseInt(h1.slice(k, k + 8), 16) >>> 0;
        const b = parseInt(h2.slice(k, k + 8), 16) >>> 0;
        let x = a ^ b;
        // popcount
        x = x - ((x >>> 1) & 0x55555555);
        x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
        x = (x + (x >>> 4)) & 0x0f0f0f0f;
        d += (x * 0x01010101) >>> 24;
      }
      return d;
    }

    // v19 pass 2: for every photo the AI marked "maybe" (ambiguous close-up),
    // pull the base64 thumbs of the closest 2 confirmed "yes" neighbors and
    // ask the AI to decide — is the maybe a close-up detail of the same item,
    // or a real divider photo?
    async function resolveMaybe(idx) {
      const p = allPhotoInfos[idx];
      if (!p || p.first_pass !== "maybe" || !p.ai_thumb_b64) return;
      // Look outward from idx to gather up to 3 nearest 'yes' neighbors.
      const neighborThumbs = [];
      let step = 1;
      while (neighborThumbs.length < 3 && step < allPhotoInfos.length) {
        for (const j of [idx - step, idx + step]) {
          if (j < 0 || j >= allPhotoInfos.length) continue;
          const q = allPhotoInfos[j];
          if (!q || q.first_pass !== "yes") continue;
          if (!q.thumb_data_url) continue;
          // Extract raw base64 from the data URL prefix.
          const b64 = q.thumb_data_url.replace(/^data:image\/jpeg;base64,/, "");
          neighborThumbs.push(b64);
          if (neighborThumbs.length >= 3) break;
        }
        step += 1;
      }
      // No confirmed neighbors? Keep the photo (safer than dropping it).
      if (neighborThumbs.length === 0) {
        p.is_blank = false;
        return;
      }
      try {
        const fd = new FormData();
        fd.append("subject_b64", p.ai_thumb_b64);
        fd.append("neighbor_b64s_json", JSON.stringify(neighborThumbs));
        const url = "__PORT_5000__".startsWith("__")
          ? "/api/jnj-resolve-maybe"
          : "__PORT_5000__/api/jnj-resolve-maybe";
        // v25.5: hard 12s timeout per photo so a slow OpenAI call can't
        // stall the whole build. Safe default on abort = keep the photo.
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 12000);
        const res = await fetch(url, { method: "POST", body: fd, signal: controller.signal });
        clearTimeout(timer);
        const j = await res.json();
        p.is_blank = !j.is_item;
        jnjLog("MAYBE-RESOLVED", `filename=${p.filename}`, `neighbors=${neighborThumbs.length}`, `verdict=${j.is_item ? "item" : "blank"}`);
      } catch (e) {
        console.warn("resolve-maybe failed for", p.filename, e && e.message);
        p.is_blank = false;  // safe default — keep the photo
        jnjLog("MAYBE-FAILED", `filename=${p.filename}`, `reason=${e && e.message}`);
      }
      // Free memory — don't need the b64 anymore.
      p.ai_thumb_b64 = "";
    }

    const maybeIndices = [];
    for (let i = 0; i < allPhotoInfos.length; i++) {
      if (allPhotoInfos[i].first_pass === "maybe") maybeIndices.push(i);
    }
    if (maybeIndices.length > 0) {
      statusEl.textContent = `resolving ${maybeIndices.length} close-up photo${maybeIndices.length === 1 ? "" : "s"}…`;
      // Run 3 at a time so we don't hammer OpenAI.
      const CONC = 3;
      for (let i = 0; i < maybeIndices.length; i += CONC) {
        await Promise.all(maybeIndices.slice(i, i + CONC).map(resolveMaybe));
      }
    }

    // v25.31: score-based top-N divider picking.
    // Old approach (v25.28-30): server flagged each photo as is_blank based
    // on hard thresholds. Photos like FILE 13 038 (a real photo of a dark
    // object) got misclassified because their pixel signature overlaps with
    // true dividers.
    //
    // New approach: server returns a 0-1000 divider_score for every photo.
    // We know exactly how many dividers should exist (item_count - 1), so
    // we pick the top N scoring photos as dividers. Self-correcting:
    // - true dividers score 850-1000 (near-black content area)
    // - borderline dark item photos score 400-600
    // - if we detect too many candidates, the borderline ones lose to true
    //   dividers and get treated as regular photos
    // - if we detect too few, the next best candidates get pulled in
    const expectedDividers = Math.max(0, items.length - 1);
    const scoredPhotos = allPhotoInfos
      .map((p, idx) => ({ p, idx, score: (typeof p.divider_score === "number" ? p.divider_score : (p.is_blank ? 700 : 0)) }))
      .sort((a, b) => b.score - a.score);
    // Pick top N as dividers. Use a minimum score floor of 300 so if a
    // sale has FEWER dividers than expected (Dave forgot some), we don't
    // start treating regular photos as dividers.
    const dividerSet = new Set();
    // v25.34: server scoring got sharper, so raise the floor. True JnJ
    // dividers now score 500-900; borderline dark item photos score <150.
    const MIN_DIVIDER_SCORE = 400;
    for (let i = 0; i < expectedDividers && i < scoredPhotos.length; i++) {
      const sp = scoredPhotos[i];
      if (sp.score < MIN_DIVIDER_SCORE) break;
      dividerSet.add(sp.idx);
    }
    // Log the score distribution so we can tune if needed.
    jnjLog("DIVIDER-PICK",
      `expected=${expectedDividers}`,
      `picked=${dividerSet.size}`,
      `top5_scores=[${scoredPhotos.slice(0, 5).map(s => s.score.toFixed(0)).join(",")}]`,
      `bottom_of_picked=${scoredPhotos[dividerSet.size - 1]?.score.toFixed(0) || "n/a"}`,
      `just_below=${scoredPhotos[dividerSet.size]?.score.toFixed(0) || "n/a"}`
    );

    // Now the cursor walk uses dividerSet instead of is_blank.
    // v25.6: still collapses consecutive dividers and only advances if the
    // current item has already received at least one photo, so a leading
    // divider doesn't skip item 1.
    let cursor = 0;
    let blanksSeen = 0;
    let currentItemGotPhoto = false;
    const walkTrace = [];
    for (let i = 0; i < allPhotoInfos.length; i++) {
      const p = allPhotoInfos[i];
      const isPickedDivider = dividerSet.has(i);
      if (isPickedDivider) {
        blanksSeen += 1;
        if (currentItemGotPhoto && cursor + 1 < items.length) {
          cursor += 1;
          currentItemGotPhoto = false;
          walkTrace.push(`${p.filename}=DIV→item${cursor+1}`);
        } else {
          walkTrace.push(`${p.filename}=DIV-skip`);
        }
        p.item_num_match = "";
        p.match_kind = "blank";
        continue;
      }
      if (p.match_kind === "none") {
        p.item_num_match = itemNumByIndex[cursor] || "";
        p.match_kind = p.item_num_match ? "order" : "none";
        currentItemGotPhoto = true;
        walkTrace.push(`${p.filename}→${p.item_num_match || "?"}`);
      }
    }
    jnjLog("CURSOR-WALK", `photos=${allPhotoInfos.length}`, `dividers_picked=${dividerSet.size}`, `final_cursor=${cursor}`, `total_items=${items.length}`);
    jnjLog("CURSOR-TRACE", walkTrace.join(" | "));

    // v25.30: reverted v25.29's AI verification pass. Ashley confirmed the
    // photos ARE in order and the black dividers ARE the source of truth
    // — so any "reassign to next item" move by the AI would ACTIVELY BREAK
    // correct assignments. The right fix is making divider detection
    // more accurate (v25.30 server.py: tightened is_divider_photo to
    // require all 3 signals — low mean + low stddev + low edge_mean —
    // to fire) so we detect exactly 50 dividers, not 57.

    // Wire everything back into local state so the preview UI can render it.
    // v25: was `[sheet, ...photos]` (singular sheet from pre-v23) — v23 renamed to
    // `sheets` (array) but this line was missed, causing a ReferenceError at the
    // very end of the build after all photos matched. Now uses the sheets array.
    // v25.19: server may return a filename with a folder prefix (e.g.
    // "2026-08-21 TEST/TEST 001.JPG") even though the original File object
    // only has the leaf name ("TEST 001.JPG"). Index by BOTH the full name
    // and the leaf so the lookup always finds the File. Case-insensitive
    // as a final safety net.
    const filesByName = new Map();
    for (const f of [...sheets, ...photos]) {
      if (!f || !f.name) continue;
      filesByName.set(f.name, f);
      filesByName.set(f.name.toLowerCase(), f);
      const leaf = f.name.split("/").pop();
      if (leaf && leaf !== f.name) {
        filesByName.set(leaf, f);
        filesByName.set(leaf.toLowerCase(), f);
      }
    }
    function lookupFile(name) {
      if (!name) return null;
      let hit = filesByName.get(name);
      if (hit) return hit;
      hit = filesByName.get(name.toLowerCase());
      if (hit) return hit;
      const leaf = name.split("/").pop();
      if (leaf && leaf !== name) {
        hit = filesByName.get(leaf) || filesByName.get(leaf.toLowerCase());
        if (hit) return hit;
      }
      return null;
    }
    const photoMap = new Map();
    let fileMissCount = 0;
    for (const p of allPhotoInfos) {
      const fileHit = lookupFile(p.filename);
      if (!fileHit) fileMissCount++;
      photoMap.set(p.filename, {
        id: p.id,
        file: fileHit || null,
        thumb: p.thumb_data_url,
        tag_read: p.tag_read,
        description_read: p.description_read,
        item_num_match: p.item_num_match,
        match_kind: p.match_kind,
        dhash: p.dhash || "",
        is_blank: p.is_blank || false,
      });
    }
    if (fileMissCount > 0) {
      jnjLog("FILE-LOOKUP-MISS", `${fileMissCount} of ${allPhotoInfos.length} photos could not resolve to a File object; first few names:`, allPhotoInfos.slice(0, 5).map(p => p.filename).join(" | "));
    }
    const itemPhotos = {};
    const unmatched = [];
    const blanks = [];
    for (const it of items) itemPhotos[it.item_num] = [];
    // Iterate in original upload order so the item-cards list photos in the
    // sequence Dave took them.
    for (const p of allPhotoInfos) {
      const info = photoMap.get(p.filename);
      if (!info) continue;
      if (info.match_kind === "blank") {
        // Blank divider — not assigned, not unmatched, just tracked so we
        // can show a count/toast to the user.
        blanks.push(p.filename);
      } else if (info.item_num_match && itemPhotos[info.item_num_match]) {
        itemPhotos[info.item_num_match].push(p.filename);
      } else {
        unmatched.push(p.filename);
      }
    }
    if (blanks.length > 0) {
      toast(`Detected ${blanks.length} blank divider${blanks.length===1?"":"s"} — split items accordingly.`);
    }

    jnjState = { items, photos: photoMap, itemPhotos, unmatched };
    jnjLog("BUILD-COMPLETE", `items=${items.length}`, `photos=${photoMap.size}`, `unmatched=${unmatched.length}`);
    jnjRenderPreview();
    // v24: verify preview actually became visible — if it didn't, something is
    // hiding it. Log and force it visible.
    setTimeout(() => {
      const isHidden = jnjPreview.classList.contains("hidden");
      const style = window.getComputedStyle(jnjPreview);
      jnjLog("PREVIEW-CHECK", `hidden=${isHidden}`, `display=${style.display}`, `visibility=${style.visibility}`, `offsetHeight=${jnjPreview.offsetHeight}`);
      if (isHidden || style.display === "none") {
        jnjLog("PREVIEW-FORCE", "preview was hidden after build — forcing visible");
        jnjPreview.classList.remove("hidden");
        jnjPreview.style.display = "";
      }
      // Scroll into view so the user can see it.
      try { jnjPreview.scrollIntoView({ behavior: "smooth", block: "start" }); } catch {}
    }, 100);
  } catch (err) {
    console.error("JnJ build error:", err);
    statusEl.textContent = err.message || "failed";
    li.classList.add("error");
    li.querySelector(".spinner").outerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="color: var(--danger); flex-shrink:0;"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>`;
    jnjShowErrorBanner(err.message || "Something went wrong. Please try again.");
  }
}

// v24: Persistent error banner — never auto-dismisses, includes a "Show debug log"
// button that opens a full-screen overlay with every logged event, plus a
// "Copy log" button so the user can paste the full log back to me.
function jnjShowErrorBanner(msg) {
  jnjLog("ERROR-BANNER", msg);
  let banner = document.getElementById("jnjErrorBanner");
  if (!banner) {
    banner = document.createElement("div");
    banner.id = "jnjErrorBanner";
    banner.style.cssText = "position:fixed;top:12px;left:12px;right:12px;z-index:9999;background:#3b1414;border:1px solid #ff5252;color:#fff;padding:14px 44px 14px 16px;border-radius:12px;font-size:14px;line-height:1.4;box-shadow:0 8px 24px rgba(0,0,0,0.4);";
    document.body.appendChild(banner);
  }
  banner.innerHTML = `
    <div style="font-weight:600;margin-bottom:4px;">JnJ Sale Builder — error</div>
    <div id="jnjErrorBannerMsg" style="font-size:13px;opacity:0.95;word-break:break-word;margin-bottom:10px;"></div>
    <div style="display:flex;gap:8px;flex-wrap:wrap;">
      <button type="button" id="jnjShowLogBtn" style="background:#ff5252;border:none;color:#fff;font-size:13px;font-weight:600;padding:8px 14px;border-radius:6px;cursor:pointer;">Show debug log</button>
      <button type="button" id="jnjCopyMsgBtn" style="background:transparent;border:1px solid #ff5252;color:#ff8a8a;font-size:13px;padding:8px 14px;border-radius:6px;cursor:pointer;">Copy error message</button>
    </div>
    <button type="button" aria-label="Dismiss" style="position:absolute;top:8px;right:8px;background:transparent;border:none;color:#fff;font-size:22px;line-height:1;cursor:pointer;padding:4px 10px;">×</button>
  `;
  banner.querySelector("#jnjErrorBannerMsg").textContent = msg;
  banner.querySelector("[aria-label='Dismiss']").onclick = () => banner.remove();
  banner.querySelector("#jnjShowLogBtn").onclick = () => jnjShowDebugOverlay();
  banner.querySelector("#jnjCopyMsgBtn").onclick = () => {
    copyText(msg);
  };
}

// v24: Full-screen debug log overlay. Shows the last 500 events as monospace
// text with a big "Copy all" button so the user can paste it back to me.
function jnjShowDebugOverlay() {
  let overlay = document.getElementById("jnjDebugOverlay");
  if (overlay) overlay.remove();
  overlay = document.createElement("div");
  overlay.id = "jnjDebugOverlay";
  overlay.style.cssText = "position:fixed;inset:0;z-index:10000;background:rgba(0,0,0,0.92);color:#eee;padding:20px;display:flex;flex-direction:column;gap:12px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;";
  const log = (window.__jnjDebugLog || []).join("\n") || "(no events logged yet)";
  const summary = `BUILD_ID: ${BUILD_ID}\nURL: ${window.location.href}\nUserAgent: ${navigator.userAgent}\nEvents: ${(window.__jnjDebugLog || []).length}\n`;
  overlay.innerHTML = `
    <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
      <div style="font-size:15px;font-weight:600;color:#7fff9f;flex:1;">Debug log — ${(window.__jnjDebugLog || []).length} events</div>
      <button type="button" id="jnjDbgCopy" style="background:#7fff9f;color:#000;border:none;padding:10px 16px;border-radius:6px;font-size:14px;font-weight:600;cursor:pointer;">Copy all</button>
      <button type="button" id="jnjDbgClose" style="background:transparent;color:#eee;border:1px solid #666;padding:10px 16px;border-radius:6px;font-size:14px;cursor:pointer;">Close</button>
    </div>
    <textarea id="jnjDbgText" readonly style="flex:1;background:#0a0a0a;color:#c8ffce;border:1px solid #333;border-radius:6px;padding:12px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;line-height:1.4;white-space:pre;overflow:auto;resize:none;"></textarea>
  `;
  document.body.appendChild(overlay);
  const ta = overlay.querySelector("#jnjDbgText");
  ta.value = summary + "\n" + log;
  overlay.querySelector("#jnjDbgClose").onclick = () => overlay.remove();
  overlay.querySelector("#jnjDbgCopy").onclick = () => {
    ta.select();
    try {
      navigator.clipboard.writeText(ta.value).then(
        () => toast("Debug log copied — paste it to Ashley"),
        () => { document.execCommand("copy"); toast("Debug log copied"); }
      );
    } catch {
      document.execCommand("copy");
      toast("Debug log copied");
    }
  };
}

function jnjRenderPreview() {
  if (!jnjState) return;
  jnjPreview.classList.remove("hidden");

  // Summary sub-text
  const total = jnjState.photos.size;
  const matched = total - jnjState.unmatched.length;
  jnjPreviewSub.textContent = `${jnjState.items.length} items · ${total} photos · ${matched} matched, ${jnjState.unmatched.length} unmatched. Drag photos between items to fix, or click Retry.`;

  // Restore preference values into fields (only if empty, so we don't clobber user edits)
  jnjRestorePrefs();

  // Items list
  jnjItemsList.innerHTML = "";
  for (const it of jnjState.items) {
    const card = document.createElement("div");
    card.className = "jnj-item-card";
    card.dataset.itemNum = it.item_num;

    // v25.8: lot codes and item numbers are INLINE-EDITABLE. Tap the pill
    // to correct any OCR misread (e.g. "116C" → "4B"). Fixes ride along
    // through to the CSV / ZIP export automatically because we mutate
    // jnjState.items[i] on blur.
    // v25.9: seller # from the sheet's boxed number is shown as its own
    // editable pill on each row — so Ashley can see the grouping visually
    // and tap-fix any item that got attached to the wrong seller box.
    const lotHtml = `<span class="lot-code lot-code-editable" contenteditable="true" spellcheck="false" data-field="lot_code" title="Tap to fix lot code">${escapeHtml(it.lot_code || "—")}</span>`;
    const sellerHtml = `<span class="seller-pill seller-pill-editable" contenteditable="true" spellcheck="false" data-field="sheet_seller_num" title="Seller # from the boxed number on the sheet — tap to fix">SELLER ${escapeHtml(it.sheet_seller_num || "?")}</span>`;
    const photos = jnjState.itemPhotos[it.item_num] || [];
    const photosHtml = photos.length
      ? photos.map(fname => jnjPhotoThumbHtml(fname)).join("")
      : `<div class="jnj-item-photos empty">no photo</div>`;

    // Determine badge from first photo (if any)
    let badgeHtml = "";
    if (photos.length) {
      const firstInfo = jnjState.photos.get(photos[0]);
      const kind = firstInfo ? firstInfo.match_kind : "none";
      const label = { tag: "TAG", desc: "DESC", order: "ORDER", manual: "MANUAL", none: "" }[kind] || "";
      if (label) badgeHtml = `<span class="jnj-match-badge ${kind}">${label}</span>`;
    }

    card.innerHTML = `
      <div class="jnj-item-num"><span contenteditable="true" spellcheck="false" data-field="item_num" title="Tap to fix item #" style="cursor:text;">${escapeHtml(it.item_num)}</span><small>#${jnjState.items.indexOf(it) + 1}</small></div>
      <div class="jnj-item-desc">${sellerHtml} ${lotHtml} <span contenteditable="true" spellcheck="false" data-field="description" title="Tap to fix description" style="cursor:text;">${escapeHtml(it.description)}</span></div>
      <div class="jnj-item-actions">
        ${badgeHtml}
        <div class="jnj-item-photos">${photosHtml}</div>
        <button class="jnj-retry-btn" data-item="${escapeAttr(it.item_num)}">↻ Retry match</button>
      </div>
    `;
    jnjItemsList.appendChild(card);

    // Wire up inline editors — on blur, write back to jnjState.items so
    // downstream CSV/ZIP export uses the corrected values.
    const currentItem = it;
    card.querySelectorAll("[contenteditable='true']").forEach(el => {
      el.addEventListener("blur", () => {
        const field = el.dataset.field;
        let val = (el.textContent || "").trim();
        if (field === "lot_code" && (val === "\u2014" || val === "-")) val = "";
        // Strip the "SELLER " prefix if the user left it in.
        if (field === "sheet_seller_num") val = val.replace(/^SELLER\s*/i, "").trim();
        if (field === "sheet_seller_num" && (val === "?" || val === "—" || val === "-")) val = "";
        // Uppercase item_num / lot_code to match sanitizer downstream
        if (field === "item_num" || field === "lot_code" || field === "sheet_seller_num") val = val.toUpperCase().replace(/\s+/g, "");
        if (currentItem[field] !== val) {
          jnjLog("USER-EDIT", `item#${jnjState.items.indexOf(currentItem)+1} ${field}: "${currentItem[field]}" → "${val}"`);
          currentItem[field] = val;
          // Re-render text but don't nuke the DOM (keeps focus friendly)
          if (field === "lot_code") el.textContent = val || "\u2014";
          if (field === "sheet_seller_num") el.textContent = `SELLER ${val || "?"}`;
        }
      });
      // Enter commits (blurs)
      el.addEventListener("keydown", (e) => {
        if (e.key === "Enter") { e.preventDefault(); el.blur(); }
      });
    });

    // Drag targets
    card.addEventListener("dragover", (e) => {
      if (e.dataTransfer.types.includes("application/x-jnj-photo")) {
        e.preventDefault();
        card.classList.add("drag-over");
      }
    });
    card.addEventListener("dragleave", () => card.classList.remove("drag-over"));
    card.addEventListener("drop", (e) => {
      if (!e.dataTransfer.types.includes("application/x-jnj-photo")) return;
      e.preventDefault();
      card.classList.remove("drag-over");
      const fname = e.dataTransfer.getData("application/x-jnj-photo");
      jnjMovePhotoTo(fname, it.item_num);
    });
    // Retry button
    card.querySelector(".jnj-retry-btn").addEventListener("click", () => jnjRetryMatch(it.item_num));
  }

  // Unmatched photos pool
  jnjUnmatched.innerHTML = jnjState.unmatched.length
    ? jnjState.unmatched.map(fname => jnjPhotoThumbHtml(fname, true)).join("")
    : "<em>none</em>";
  jnjUnmatched.addEventListener("dragover", (e) => {
    if (e.dataTransfer.types.includes("application/x-jnj-photo")) {
      e.preventDefault();
      jnjUnmatched.classList.add("drag-over");
    }
  });
  jnjUnmatched.addEventListener("dragleave", () => jnjUnmatched.classList.remove("drag-over"));
  jnjUnmatched.addEventListener("drop", (e) => {
    if (!e.dataTransfer.types.includes("application/x-jnj-photo")) return;
    e.preventDefault();
    jnjUnmatched.classList.remove("drag-over");
    const fname = e.dataTransfer.getData("application/x-jnj-photo");
    jnjMovePhotoTo(fname, null);
  });

  // Wire up all thumbs (drag handlers)
  document.querySelectorAll(".jnj-photo-thumb").forEach(el => {
    el.setAttribute("draggable", "true");
    el.addEventListener("dragstart", (e) => {
      const fname = el.dataset.filename;
      e.dataTransfer.setData("application/x-jnj-photo", fname);
      e.dataTransfer.effectAllowed = "move";
      el.classList.add("dragging");
    });
    el.addEventListener("dragend", () => el.classList.remove("dragging"));
    // Remove-photo button (X)
    const rm = el.querySelector(".jnj-photo-remove");
    if (rm) rm.addEventListener("click", (e) => {
      e.stopPropagation();
      jnjMovePhotoTo(el.dataset.filename, null);
    });
  });
}

function jnjPhotoThumbHtml(fname, isUnmatched = false) {
  const info = jnjState.photos.get(fname);
  if (!info) return "";
  const selected = jnjSelectedPhoto === fname ? " selected" : "";
  const cls = (isUnmatched ? "jnj-photo-thumb unmatched" : "jnj-photo-thumb") + selected;
  const tagBadge = info.tag_read ? ` title="Tag: ${escapeAttr(info.tag_read)}"` : (info.description_read ? ` title="${escapeAttr(info.description_read)}"` : "");
  return `<div class="${cls}" data-filename="${escapeAttr(fname)}"${tagBadge}>
    <img src="${info.thumb}" alt="" />
    <button class="jnj-photo-remove" title="Remove">×</button>
  </div>`;
}

// Mobile tap-to-assign: tap a photo to select it, tap an item card to assign.
let jnjSelectedPhoto = null;
document.addEventListener("click", (e) => {
  if (!jnjState) return;
  // Ignore remove-button clicks — they have their own handler
  if (e.target.closest(".jnj-photo-remove")) return;
  // Ignore any actual button/link/input clicks
  if (e.target.closest("button, a, input, textarea, select, label")) return;

  const thumb = e.target.closest(".jnj-photo-thumb");
  const itemCard = e.target.closest(".jnj-item-card");

  if (thumb) {
    // Tap a photo -> select it (or unselect if already selected)
    const fname = thumb.dataset.filename;
    jnjSelectedPhoto = (jnjSelectedPhoto === fname) ? null : fname;
    if (jnjSelectedPhoto) {
      toast(`Selected photo. Now tap an item to assign it (or the Unmatched area to unassign).`);
    }
    jnjRenderPreview();
  } else if (itemCard && jnjSelectedPhoto) {
    // Tap an item card with a selected photo -> assign
    const itemNum = itemCard.dataset.itemNum;
    if (itemNum) {
      jnjMovePhotoTo(jnjSelectedPhoto, itemNum);
      toast(`Assigned to ${itemNum}.`);
      jnjSelectedPhoto = null;
    }
  } else if (jnjSelectedPhoto && e.target.closest("#jnjUnmatched")) {
    // Tap unmatched area with a selected photo -> unassign
    jnjMovePhotoTo(jnjSelectedPhoto, null);
    toast("Moved back to unmatched.");
    jnjSelectedPhoto = null;
  }
});

function jnjMovePhotoTo(fname, targetItemNum) {
  if (!jnjState) return;
  // Remove from wherever it currently is
  for (const key of Object.keys(jnjState.itemPhotos)) {
    jnjState.itemPhotos[key] = jnjState.itemPhotos[key].filter(f => f !== fname);
  }
  jnjState.unmatched = jnjState.unmatched.filter(f => f !== fname);

  // Add to target
  if (targetItemNum === null) {
    jnjState.unmatched.push(fname);
    const info = jnjState.photos.get(fname);
    if (info) { info.item_num_match = ""; info.match_kind = "none"; }
  } else if (jnjState.itemPhotos[targetItemNum] !== undefined) {
    jnjState.itemPhotos[targetItemNum].push(fname);
    const info = jnjState.photos.get(fname);
    if (info) { info.item_num_match = targetItemNum; info.match_kind = "manual"; }
  }
  jnjRenderPreview();
}

async function jnjRetryMatch(itemNum) {
  if (!jnjState) return;
  const currentPhotos = jnjState.itemPhotos[itemNum] || [];
  if (currentPhotos.length === 0) {
    toast("No photo assigned to retry.");
    return;
  }
  // Pop the first photo, ask AI to re-match it
  const fname = currentPhotos[0];
  const info = jnjState.photos.get(fname);
  if (!info) return;

  // Move it to unmatched first
  jnjState.itemPhotos[itemNum] = jnjState.itemPhotos[itemNum].filter(f => f !== fname);
  jnjState.unmatched.push(fname);
  jnjRenderPreview();

  // Ask backend for a new match (excluding the item that was wrong)
  try {
    const fd = new FormData();
    fd.append("photo_id", info.id || fname);
    fd.append("description", info.description_read || "");
    // Filter out the wrong item from the list we send
    const filteredItems = jnjState.items.filter(i => i.item_num !== itemNum);
    fd.append("items_json", JSON.stringify(filteredItems));
    const res = await fetch(JNJ_REMATCH_URL, { method: "POST", body: fd });
    if (!res.ok) throw new Error(`retry failed (${res.status})`);
    const data = await res.json();
    if (data.item_num_match) {
      jnjMovePhotoTo(fname, data.item_num_match);
      toast(`Re-matched to ${data.item_num_match}.`);
    } else {
      toast("AI couldn't confidently re-match. Drag it manually.");
    }
  } catch (e) {
    toast(e.message || "Retry failed");
  }
}

async function jnjDownloadZip() {
  if (!jnjState) return;
  const saleName = jnjSaleName.value.trim();
  const sellerId = jnjSellerId.value.trim();
  const sellerStart = parseInt(jnjSellerStart.value.trim() || "1000", 10);
  if (!saleName) { toast("Please enter the Category / Sale name."); jnjSaleName.focus(); return; }
  if (!sellerId) { toast("Please enter the Seller ID prefix."); jnjSellerId.focus(); return; }

  jnjSavePrefs();

  jnjDownloadBtn.disabled = true;
  const originalText = jnjDownloadBtn.textContent;
  jnjDownloadBtn.textContent = "Building zip…";

  try {
    // Build photo_map (filename → item_num) from current state
    const photoMap = {};
    for (const [itemNum, fnames] of Object.entries(jnjState.itemPhotos)) {
      for (const fname of fnames) photoMap[fname] = itemNum;
    }

    const fd = new FormData();
    fd.append("sale_name", saleName);
    fd.append("seller_id", sellerId);
    fd.append("seller_start", String(sellerStart));
    fd.append("items_json", JSON.stringify(jnjState.items));
    fd.append("photo_map_json", JSON.stringify(photoMap));
    // Attach every matched photo file. v25.18: log EVERY attach so we can see
    // in the debug log whether photos actually made it into the multipart body.
    // Previously we saw a ZIP with 0 photos even though preview showed 83
    // matched — need to know if browser attached them or dropped them silently.
    let attachedCount = 0, missingFile = 0, missingInfo = 0;
    let totalBytes = 0;
    for (const fname of Object.keys(photoMap)) {
      const info = jnjState.photos.get(fname);
      if (!info) { missingInfo++; continue; }
      if (!info.file) { missingFile++; continue; }
      fd.append("photos", info.file, fname);
      attachedCount++;
      totalBytes += info.file.size || 0;
    }
    jnjLog("ZIP-UPLOAD", `photoMap has ${Object.keys(photoMap).length} entries; attached ${attachedCount} photos to FormData (${(totalBytes/1024/1024).toFixed(1)} MB total); missingInfo=${missingInfo}, missingFile=${missingFile}`);
    if (attachedCount === 0 && Object.keys(photoMap).length > 0) {
      toast(`WARNING: 0 of ${Object.keys(photoMap).length} matched photos attached to upload — tap the version badge to see the debug log.`);
    }

    const res = await fetch(JNJ_ZIP_URL, { method: "POST", body: fd });
    if (!res.ok) {
      let msg = `server error (${res.status})`;
      try { const j = await res.json(); msg = j.detail || msg; } catch {}
      throw new Error(msg);
    }
    const blob = await res.blob();
    const cd = res.headers.get("Content-Disposition") || "";
    const match = cd.match(/filename="([^"]+)"/);
    const filename = match ? match[1] : `jnj-sale-${jnjState.items.length}items.zip`;
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = filename; document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
    toast("Zip downloaded.");
  } catch (e) {
    console.error(e);
    toast(e.message || "Download failed");
  } finally {
    jnjDownloadBtn.disabled = false;
    jnjDownloadBtn.textContent = originalText;
  }
}

jnjDownloadBtn.addEventListener("click", jnjDownloadZip);
jnjCancelBtn.addEventListener("click", () => {
  if (confirm("Discard this sale preview?")) jnjResetState();
});

// v22: bottom-of-preview download + cancel buttons so Dave doesn't have to
// scroll back to the top to find the Download ZIP button after reviewing
// dozens of items across multiple sheets. Both bottom buttons just delegate
// to the top ones — that keeps the disable/enable + "Building zip…" state
// logic in exactly one place.
const jnjDownloadBtnBottom = document.getElementById("jnjDownloadBtnBottom");
const jnjCancelBtnBottom = document.getElementById("jnjCancelBtnBottom");
if (jnjDownloadBtnBottom) {
  jnjDownloadBtnBottom.addEventListener("click", () => jnjDownloadBtn.click());
}
if (jnjCancelBtnBottom) {
  jnjCancelBtnBottom.addEventListener("click", () => jnjCancelBtn.click());
}
// Keep the bottom download button's label + disabled state visually in sync
// with the top one while the zip is being built. We watch the top button and
// mirror its state to the bottom.
if (jnjDownloadBtnBottom && "MutationObserver" in window) {
  const syncBottom = () => {
    jnjDownloadBtnBottom.disabled = jnjDownloadBtn.disabled;
    // Only mirror text when the top button is in the "Building zip…" state or
    // returning to its default — don't override the bottom's own label otherwise.
    if (jnjDownloadBtn.disabled) {
      jnjDownloadBtnBottom.textContent = jnjDownloadBtn.textContent;
    } else {
      jnjDownloadBtnBottom.textContent = "Download ZIP";
    }
  };
  new MutationObserver(syncBottom).observe(jnjDownloadBtn, {
    attributes: true, childList: true, characterData: true, subtree: true,
  });
}

/* -------------------- Init -------------------- */
tryAuth();
if (!lockScreen.classList.contains("hidden")) {
  passwordInput.focus();
}
