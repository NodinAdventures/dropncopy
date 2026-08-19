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
const BUILD_ID = "2026-08-19-anchor-distribute-v14";
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
// v14: gpt-4o-mini reads tags in ~1s each (vs 3-4s for gpt-4o). Photos in a
// batch fire concurrently, so an 8-photo batch takes ~1.2s. 4 batches parallel
// = 32 photos in flight, well within OpenAI rate limits and Render's memory.
const JNJ_PHOTO_BATCH_SIZE = 8;
const JNJ_PHOTO_CONCURRENCY = 4; // Number of batches to run in parallel.
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
let jnjStaged = { sheet: null, photos: [] };

function jnjRenderStaged() {
  const hasAny = jnjStaged.sheet || jnjStaged.photos.length > 0;
  jnjStagedList.style.display = hasAny ? "block" : "none";
  jnjStageActions.style.display = hasAny ? "flex" : "none";

  const rows = [];
  if (jnjStaged.sheet) {
    const s = jnjStaged.sheet;
    rows.push(`<div class="jnj-staged-row" style="display:flex;align-items:center;gap:10px;padding:8px 12px;background:rgba(147,112,255,0.08);border:1px solid rgba(147,112,255,0.25);border-radius:8px;margin-bottom:6px;">
      <span style="font-size:11px;font-weight:600;color:#9370ff;letter-spacing:0.5px;">SHEET</span>
      <span style="flex:1;font-size:13px;color:var(--text);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escapeHtml(s.name || "(no name)")}</span>
      <span style="font-size:11px;color:var(--muted);">${(s.size/1024).toFixed(0)} KB</span>
      <button type="button" data-remove="sheet" class="ghost-btn" style="padding:2px 8px;font-size:12px;">Remove</button>
    </div>`);
  }
  jnjStaged.photos.forEach((p, i) => {
    rows.push(`<div class="jnj-staged-row" style="display:flex;align-items:center;gap:10px;padding:6px 12px;font-size:13px;">
      <span style="font-size:11px;color:var(--muted);width:24px;">${i+1}.</span>
      <span style="flex:1;color:var(--text);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escapeHtml(p.name || "photo")}</span>
      <span style="font-size:11px;color:var(--muted);">${(p.size/1024).toFixed(0)} KB</span>
      <button type="button" data-remove-photo="${i}" class="ghost-btn" style="padding:2px 8px;font-size:12px;">Remove</button>
    </div>`);
  });

  if (hasAny) {
    const total = (jnjStaged.sheet ? 1 : 0) + jnjStaged.photos.length;
    const missing = [];
    if (!jnjStaged.sheet) missing.push("sheet");
    if (jnjStaged.photos.length === 0) missing.push("at least 1 photo");
    const header = missing.length
      ? `<div style="font-size:12px;color:#ff9a3d;margin-bottom:8px;font-weight:600;">Still need: ${missing.join(" and ")}</div>`
      : `<div style="font-size:12px;color:#3ecf8e;margin-bottom:8px;font-weight:600;">Ready — ${total} files staged</div>`;
    jnjStagedList.innerHTML = header + rows.join("");
  }
  jnjBuildBtn.disabled = !(jnjStaged.sheet && jnjStaged.photos.length > 0);
  jnjBuildBtn.style.opacity = jnjBuildBtn.disabled ? "0.5" : "1";
}

// Remove buttons on staged rows
jnjStagedList.addEventListener("click", (e) => {
  const btn = e.target.closest("button");
  if (!btn) return;
  if (btn.dataset.remove === "sheet") {
    jnjStaged.sheet = null;
    jnjRenderStaged();
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
  const f = e.target.files && e.target.files[0];
  if (f) {
    jnjStaged.sheet = f;
    jnjRenderStaged();
  }
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
  jnjStaged = { sheet: null, photos: [] };
  jnjRenderStaged();
});

jnjBuildBtn.addEventListener("click", () => {
  if (!jnjStaged.sheet || jnjStaged.photos.length === 0) return;
  const allFiles = [jnjStaged.sheet, ...jnjStaged.photos];
  jnjHandleFiles(allFiles);
});

// Drag-and-drop desktop path — dropped files merge into staged list.
// A .pdf file goes into the sheet slot; images go into photos.
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
    if (isPdf && !jnjStaged.sheet) {
      jnjStaged.sheet = f;
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
    if (isPdf && !jnjStaged.sheet) jnjStaged.sheet = f;
    else if (!jnjStaged.photos.some(p => p.name === f.name && p.size === f.size)) {
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
  badge.textContent = `v14 · ${BUILD_ID}`;
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

async function jnjHandleFiles(files) {
  if (!files.length) return;
  if (files.length < 2) {
    toast("Drop the sheet + at least one photo together.");
    return;
  }

  // Separate sheet from photos on the client (server used to do this, but we
  // now call two separate endpoints so we split here).
  let sheet = null;
  const photos = [];
  for (const f of files) {
    const name = (f.name || "").toLowerCase();
    const isPdf = name.endsWith(".pdf") || f.type === "application/pdf";
    if (isPdf && !sheet) sheet = f;
    else photos.push(f);
  }
  // If no PDF, fall back to filename hints (any "sheet"/"intake"/"file*" image).
  if (!sheet) {
    for (let i = 0; i < photos.length; i++) {
      const n = (photos[i].name || "").toLowerCase();
      if (/(sheet|intake)/.test(n) || n.startsWith("file")) {
        sheet = photos.splice(i, 1)[0];
        break;
      }
    }
  }
  // Last-resort: use the smallest file as the sheet (single-page PDFs are ~200KB,
  // item photos from phones are 2–4MB).
  if (!sheet && photos.length > 1) {
    photos.sort((a, b) => a.size - b.size);
    sheet = photos.shift();
  }
  if (!sheet) {
    jnjShowErrorBanner("Couldn't identify a sheet in the upload. Please add the intake sheet as a PDF or a clear photo.");
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

    // ---------- Step 1: transcribe the sheet ----------
    statusEl.textContent = "reading sheet…";
    const sheetFd = new FormData();
    sheetFd.append("sheet", sheet);
    const sheetData = await postForJson(JNJ_BUILD_SHEET_URL, sheetFd, "Sheet transcription");
    const items = sheetData.items || [];
    if (!items.length) throw new Error("Sheet transcribed but no item rows were parsed.");
    statusEl.textContent = `sheet done — ${items.length} items. Matching ${photos.length} photos…`;

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

    // v14 matching: photos are in shoot order = sheet order (per Dave).
    // Instead of guessing scene changes, we anchor on tags read by AI and
    // distribute the photos between anchors evenly across the items in that
    // range. If NO tags are read, we distribute all photos evenly across all
    // items. This is deterministic and never "bounces" — the user drag-fixes
    // any misalignments in the preview UI, which is the whole point of it.

    // Step 1: build list of anchors [{photoIdx, itemIdx}] from tag reads.
    // Photos with unreadable/ambiguous tags stay unanchored.
    const anchors = [];
    allPhotoInfos.forEach((p, photoIdx) => {
      if (p.match_kind === "tag" && indexByItemNum.has(p.item_num_match)) {
        anchors.push({ photoIdx, itemIdx: indexByItemNum.get(p.item_num_match) });
      }
    });

    // Step 2: split photos into segments by anchors, then distribute each
    // segment evenly across the item range it spans.
    //   • segment start = previous anchor's photoIdx (or 0)
    //   • segment end   = this anchor's photoIdx (or allPhotoInfos.length)
    //   • item range    = previous anchor's itemIdx (or 0) → this anchor's itemIdx
    // Distribution is proportional: if segment has 12 photos over 4 items,
    // photos 0–2 → item A, 3–5 → item B, 6–8 → item C, 9–11 → item D.
    function assignSegment(startPhoto, endPhoto, startItem, endItem) {
      const nPhotos = endPhoto - startPhoto;
      const nItems = Math.max(1, endItem - startItem + 1);
      if (nPhotos <= 0) return;
      for (let i = 0; i < nPhotos; i++) {
        // Even split across the item range.
        const bucket = Math.min(nItems - 1, Math.floor((i * nItems) / nPhotos));
        const itemIdx = Math.min(items.length - 1, startItem + bucket);
        const p = allPhotoInfos[startPhoto + i];
        if (p.match_kind === "none") {
          p.item_num_match = itemNumByIndex[itemIdx] || "";
          p.match_kind = p.item_num_match ? "order" : "none";
        }
      }
    }

    if (anchors.length === 0) {
      // No tags read — just spread all photos across all items evenly.
      assignSegment(0, allPhotoInfos.length, 0, items.length - 1);
    } else {
      // Leading segment (photos before first anchor → items 0 → first anchor's item)
      const first = anchors[0];
      assignSegment(0, first.photoIdx, 0, first.itemIdx);
      // Between consecutive anchors
      for (let i = 0; i < anchors.length - 1; i++) {
        const a = anchors[i], b = anchors[i + 1];
        assignSegment(a.photoIdx + 1, b.photoIdx, a.itemIdx, b.itemIdx);
      }
      // Trailing segment (photos after last anchor → last anchor's item → last item)
      const last = anchors[anchors.length - 1];
      assignSegment(last.photoIdx + 1, allPhotoInfos.length, last.itemIdx, items.length - 1);
    }

    // Wire everything back into local state so the preview UI can render it.
    const filesByName = new Map();
    for (const f of [sheet, ...photos]) filesByName.set(f.name, f);
    const photoMap = new Map();
    for (const p of allPhotoInfos) {
      photoMap.set(p.filename, {
        id: p.id,
        file: filesByName.get(p.filename) || null,
        thumb: p.thumb_data_url,
        tag_read: p.tag_read,
        description_read: p.description_read,
        item_num_match: p.item_num_match,
        match_kind: p.match_kind,
        dhash: p.dhash || "",
      });
    }
    const itemPhotos = {};
    const unmatched = [];
    for (const it of items) itemPhotos[it.item_num] = [];
    // Iterate in original upload order so the item-cards list photos in the
    // sequence Dave took them.
    for (const p of allPhotoInfos) {
      const info = photoMap.get(p.filename);
      if (!info) continue;
      if (info.item_num_match && itemPhotos[info.item_num_match]) {
        itemPhotos[info.item_num_match].push(p.filename);
      } else {
        unmatched.push(p.filename);
      }
    }

    jnjState = { items, photos: photoMap, itemPhotos, unmatched };
    jnjRenderPreview();
  } catch (err) {
    console.error("JnJ build error:", err);
    statusEl.textContent = err.message || "failed";
    li.classList.add("error");
    li.querySelector(".spinner").outerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="color: var(--danger); flex-shrink:0;"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>`;
    jnjShowErrorBanner(err.message || "Something went wrong. Please try again.");
  }
}

// Persistent error banner — stays until user dismisses it.
function jnjShowErrorBanner(msg) {
  let banner = document.getElementById("jnjErrorBanner");
  if (!banner) {
    banner = document.createElement("div");
    banner.id = "jnjErrorBanner";
    banner.style.cssText = "position:fixed;top:12px;left:12px;right:12px;z-index:9999;background:#3b1414;border:1px solid #ff5252;color:#fff;padding:14px 44px 14px 16px;border-radius:12px;font-size:14px;line-height:1.4;box-shadow:0 8px 24px rgba(0,0,0,0.4);";
    document.body.appendChild(banner);
  }
  banner.innerHTML = `
    <div style="font-weight:600;margin-bottom:4px;">JnJ Sale Builder — error</div>
    <div id="jnjErrorBannerMsg" style="font-size:13px;opacity:0.95;word-break:break-word;"></div>
    <button type="button" aria-label="Dismiss" style="position:absolute;top:8px;right:8px;background:transparent;border:none;color:#fff;font-size:22px;line-height:1;cursor:pointer;padding:4px 10px;">×</button>
  `;
  banner.querySelector("#jnjErrorBannerMsg").textContent = msg;
  banner.querySelector("button").onclick = () => banner.remove();
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

    const lotHtml = it.lot_code ? `<span class="lot-code">${escapeHtml(it.lot_code)}</span>` : "";
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
      <div class="jnj-item-num">${escapeHtml(it.item_num)}<small>#${jnjState.items.indexOf(it) + 1}</small></div>
      <div class="jnj-item-desc">${lotHtml}${escapeHtml(it.description)}</div>
      <div class="jnj-item-actions">
        ${badgeHtml}
        <div class="jnj-item-photos">${photosHtml}</div>
        <button class="jnj-retry-btn" data-item="${escapeAttr(it.item_num)}">↻ Retry match</button>
      </div>
    `;
    jnjItemsList.appendChild(card);

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
    // Attach every matched photo file
    for (const fname of Object.keys(photoMap)) {
      const info = jnjState.photos.get(fname);
      if (info && info.file) fd.append("photos", info.file, fname);
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

/* -------------------- Init -------------------- */
tryAuth();
if (!lockScreen.classList.contains("hidden")) {
  passwordInput.focus();
}
