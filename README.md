# File Retype Admin

A small web app that takes a handwritten JnJ auction intake sheet (PDF or image), reads every row with a vision AI, and outputs each item on its own line formatted for the JnJ auction listing format (uppercase, only A-Z / 0-9 / space / `$`, lot codes stripped, abbreviations expanded).

## What's inside

| File | Purpose |
|---|---|
| `server.py` | FastAPI backend. Renders PDF pages, filters blank pages, sends real pages to Claude Sonnet vision in parallel, streams progress, and sanitizes the output. |
| `index.html` | Single-page UI. Password gate, file drop, live progress, item cards, entry history. |
| `app.js` | Frontend logic. Parses the transcript into item cards, handles Copy / Delete / Email buttons, persists entries in `localStorage`. |
| `style.css` | Styling. |

## How it works

1. User uploads a PDF or image of a handwritten intake sheet.
2. Backend renders every PDF page to an image at 180 DPI (PyMuPDF).
3. Blank pages are filtered out by a fast pixel check (PIL grayscale, <0.5% dark pixels = blank).
4. Real pages are sent to OpenAI's GPT-4o vision model in parallel (asyncio.Semaphore of 8) with a detailed prompt describing the JnJ format rules.
5. Progress streams back over NDJSON so the frontend can show "page X of Y".
6. The final transcript is sanitized (uppercase, disallowed chars removed, lot codes stripped) and returned.
7. Frontend parses each line into an item card with a Copy button, and stacks the entry into the history at the bottom.

## Running locally

You need:

- Python 3.10+
- An OpenAI API key exported as `OPENAI_API_KEY`

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
uvicorn server:app --host 0.0.0.0 --port 5000
```

Then open `http://localhost:5000` in a browser. The password is `LunchTime` (see `app.js`).

## Deploying elsewhere

The frontend expects the backend on the same origin, using the `__PORT_5000__` proxy pattern from the Perplexity deploy tool. If deploying somewhere else, replace `__PORT_5000__` in `app.js` with your backend URL.

## Format rules baked into the prompt

- Output is UPPERCASE, only A-Z, 0-9, space, and `$`
- One row on the sheet = one line in the output = one auction listing
- The item number is read exactly as written (letters and digits both preserved)
- Lot codes like `18A`, `Z`, `P 3` between the item number and the description are stripped
- Common abbreviations are expanded (Lg → LARGE, w/ → WITH, NIB → NEW IN BOX, etc.)
- Every visible item number gets a line; unreadable ones become `NUMBER ILLEGIBLE`
- Seller name, address, phone, page numbers, and form headers are NOT output
- Order is preserved page-by-page, top-to-bottom (no sorting)

## Notes

- `__pycache__/` is excluded from the zip.
- The frontend is a static bundle — the only backend endpoints are `/api/extract-stream` (POST, multipart) and the static file server.
- Item history persists in the browser's `localStorage` only.
