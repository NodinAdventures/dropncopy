/* =========================================================
   File Retype Admin
   - Password gate (default: admin123)
   - Drag/drop or click to upload
   - Extracts text from PDF, images (OCR), DOCX, TXT
   - Saves to localStorage; renders newest cards at bottom
   - Copy / Email (mailto) / Print per entry
========================================================= */

const PASSWORD = "LunchTime";
const STORAGE_KEY = "retype_entries_v1";
const AUTH_KEY = "retype_authed_v1";

// Backend endpoint. Uses the same-origin proxy path when deployed via pplx sites,
// otherwise a relative path for local dev.
const EXTRACT_URL = "__PORT_5000__/api/extract-stream";

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

/* -------------------- Init -------------------- */
tryAuth();
if (!lockScreen.classList.contains("hidden")) {
  passwordInput.focus();
}
