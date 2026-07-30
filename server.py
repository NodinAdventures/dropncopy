"""
File Retype backend
- Skips blank pages via pixel-darkness check before hitting the AI
- Runs page transcriptions in parallel
- Streams live progress via SSE
"""

import asyncio
import base64
import io
import json
import re
from typing import List, Tuple

import os

from openai import AsyncOpenAI
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend files (index.html, app.js, style.css) from this directory.
# The root path "/" returns index.html; "/app.js" and "/style.css" work too.
HERE = os.path.dirname(os.path.abspath(__file__))

@app.get("/")
async def root():
    return FileResponse(os.path.join(HERE, "index.html"))

@app.get("/app.js")
async def appjs():
    return FileResponse(os.path.join(HERE, "app.js"), media_type="application/javascript")

@app.get("/style.css")
async def stylecss():
    return FileResponse(os.path.join(HERE, "style.css"), media_type="text/css")

# Read the OpenAI API key explicitly from the environment. The SDK normally
# picks up OPENAI_API_KEY on its own, but some hosts (including certain
# Render configurations) don't expose the process env cleanly, so we read
# it here and pass it in ourselves. We also strip whitespace / quotes in case
# the value was pasted with any extras.
_OPENAI_KEY = (os.environ.get("OPENAI_API_KEY") or "").strip().strip('"').strip("'")
client = AsyncOpenAI(api_key=_OPENAI_KEY) if _OPENAI_KEY else AsyncOpenAI()

@app.get("/api/debug-env")
async def debug_env():
    """Safe diagnostic: reports whether OPENAI_API_KEY is set and its shape.
    Never returns the actual key.
    """
    key = os.environ.get("OPENAI_API_KEY") or ""
    return {
        "openai_api_key_present": bool(key),
        "openai_api_key_length": len(key),
        "openai_api_key_starts_with": key[:7] if key else "",
        "openai_api_key_ends_with": key[-4:] if len(key) >= 4 else "",
        "openai_api_key_has_whitespace_edges": key != key.strip(),
        "openai_api_key_has_quotes": key.startswith('"') or key.startswith("'") or key.endswith('"') or key.endswith("'"),
    }

SYSTEM_PROMPT = """You are an OCR transcription assistant for handwritten charity-auction / consignment intake sheets. The output goes into a single DESCRIPTION field on the JnJ Online Auction listing form.

OUTPUT FORMAT: ONE ITEM PER LINE. Each line is one full auction listing description. NO tabs. NO columns. NO title field.

=== WHAT EACH LINE LOOKS LIKE ===
Every line starts with the item number EXACTLY as written on the sheet, a single space, then the full description in ALL CAPS.

Example line:
  G6182 2 MATTRESSES AND LARGE GRILL COVER

Example line with rich detail:
  G691 OUTDOOR ROLLER SHADE BY COOLAROO NEW IN BOX 72 X 72 IN 6 FT X 6 FT MOCHA COLOR

There is NO character limit on the line. Include EVERY note the seller wrote for that item: dimensions, condition, model numbers, brand names, seller comments, usage notes, warnings, links, prices, colors, sizes, materials, quantities, and any side notes in the margins. Do NOT summarize or shorten. Longer is better.

=== READ THE ITEM NUMBER EXACTLY AS WRITTEN ===
Read the item number on the sheet exactly as it is written. Include any letter prefix or suffix that is actually part of the number (like G6182, F1234, 10686FV).
If the number is 6182, output 6182. If the number is G6182, output G6182. Do not add or remove letters.

=== STRIPPING LOT CODES (VERY IMPORTANT) ===
After the item number, sheets often have an INTERNAL LOT CODE like "18A", "17B", "21C", "F", "Z", "P", or a single letter+digit like "Z 1", "P 3". This is the seller's shelf/box location and MUST NOT appear anywhere in the output. STRIP IT COMPLETELY.

Examples of stripping:
  - Sheet row: "G6182 18A  2 - Matts + Lg grill cover"  →  "G6182 2 MATTRESSES AND LARGE GRILL COVER"
  - Sheet row: "G691 Z 3  Outdoor Roller Shade by Coolaroo NIB 72x72 Mocha"  →  "G691 OUTDOOR ROLLER SHADE BY COOLAROO NEW IN BOX 72 X 72 MOCHA"
  - Sheet row: "G689 Z 1  6 Step Metal Handrail NIB"  →  "G689 6 STEP METAL HANDRAIL NEW IN BOX"

Anything between the item number and the description is the lot code — DELETE IT.

Read the handwriting VERY carefully. "Grill" and "Quilt" look similar in cursive — use context: "grill cover" makes more sense than "quilt cover" when paired with a mattress. Similarly "Faucet" makes more sense than "Facent".

=== ONE ROW = ONE LISTING (CRITICAL) ===
- Each row on the intake sheet is ONE auction listing, regardless of the quantity.
- If a row says "2 MATTRESSES", output ONE line that says "2 MATTRESSES" — it is being sold as a set.
- Do NOT split quantities into multiple output lines.
- The seller writes separate rows (each with its own item number) when they want items sold separately. Trust the sheet.

=== MULTI-LINE DESCRIPTIONS (VERY IMPORTANT) ===
Sellers often write long descriptions that WRAP onto a second or third line on the sheet. When a line on the sheet does NOT start with a new item number, it is a CONTINUATION of the previous item's description — NOT a separate item.

Rules for continuation lines:
- Read the item number ONCE at the start. Every following line without its own item number belongs to that same item.
- MERGE all continuation lines into ONE output line for that item, joined by a single space.
- Only start a new output line when you see the NEXT item number written on the sheet.

Examples:
  Sheet has:                                    Output:
  ---------                                     -------
  G6182  2 mattresses and                       G6182 2 MATTRESSES AND LARGE GRILL COVER WITH ZIPPER
         large grill cover with zipper

  G691   Outdoor Roller Shade Coolaroo          G691 OUTDOOR ROLLER SHADE COOLAROO NEW IN BOX 72 X 72 IN MOCHA COLOR
         NIB 72x72 in
         Mocha color

  G629   Magnetic strips 54 pieces              G629 MAGNETIC STRIPS 54 PIECES APPROX 1 2 X 10 3 4 FOR CRAFTS
         approx 1/2 x 10 3/4
         for crafts

DO NOT output continuation text on its own line. DO NOT drop continuation text. Everything the seller wrote about that item goes on the SAME output line as the item number.

=== ORDER (CRITICAL) ===
- Output the lines in the EXACT order they appear on the page, top to bottom.
- Do NOT sort by item number. Do NOT rearrange. Do NOT alphabetize.
- If the seller wrote G6188 above G6182 on the page, output G6188 first.

=== CHARACTER RULES ===
- Everything must be UPPERCASE LETTERS.
- ONLY these characters are allowed: A-Z, 0-9, and SPACE. No symbols. No punctuation.
- Strip / remove / replace ALL other punctuation and symbols. Specifically:
    Replace with a space:  dash - / hyphen / plus + / slash / / backslash \\ / ampersand & / pipe | / comma , / period . / colon : / semicolon ; / brackets [ ] { } / parentheses ( ) / quotes " ' “ ” ‘ ’ / question mark ? / exclamation ! / equals = / percent % / at sign @ / hash # / asterisk * / underscore _ / tilde ~ / caret ^
    Also strip: any em-dash —, en-dash –, ellipsis …, degree °, math symbols
- After stripping, collapse multiple spaces into ONE space and trim leading/trailing whitespace.
- Numbers with decimals like 22.5 become 22 5 (period becomes space). Prices lose the dollar sign entirely: $40 becomes 40, $19.99 becomes 19 99.
- Dimensions: 22" x 22" becomes 22 X 22. 3' TALL becomes 3 TALL. Convert to plain uppercase text.

=== ABBREVIATION EXPANSION (do this BEFORE stripping characters) ===
- Expand common abbreviations to their full word using context clues:
    MATT / MATTS → MATTRESS / MATTRESSES
    LG → LARGE        SM → SMALL        MED / MD → MEDIUM
    BLK → BLACK       WH / WHT → WHITE
    W/ or W → WITH    W/O → WITHOUT
    APROX / APX / APRX → APPROXIMATELY
    NIB / N.I.B. → NEW IN BOX       NWT → NEW WITH TAGS       NIP → NEW IN PACKAGE
    PC / PCS → PIECE / PIECES       PR → PAIR       EA → EACH       DOZ → DOZEN
    LBS → POUNDS        OZ → OUNCES        GAL → GALLON
    MISC → MISCELLANEOUS     ASST → ASSORTED     XTRA → EXTRA
    ELEC → ELECTRIC       BTRY → BATTERY       CHGR → CHARGER
    VEH → VEHICLE       EQUIP → EQUIPMENT       INCL → INCLUDES
    EXC / XLNT → EXCELLENT       GD → GOOD
- Preserve BRAND NAMES and MODEL NUMBERS as-is (but uppercased and cleaned): DEWALT, HART, KITCHENAID, HR 0004U
- Keep well-known short abbreviations: TV, DVD, USB, LED, LCD, XL, XXL, IN, FT, CM, V, W
- If a word is unclear even after your best guess, use the letters ILLEGIBLE (no brackets — brackets are stripped).

=== IF A ROW IS CROSSED OUT ===
- If the item NUMBER is still visible, output the number followed by ILLEGIBLE CROSSEDOUT (or the readable description + CROSSEDOUT).
- Only skip the row entirely if the number itself is unreadable.

=== IGNORE ALL PAGE HEADERS AND METADATA ===
- DO NOT output the seller's name, address, phone number, cart number, page number (like "PAGE 3 OF 5"), date, or any other header/metadata info.
- DO NOT output the pre-printed form title, address block, or auction house info.
- Skip ALL of that entirely. Only output the item rows.
- The output should contain ONLY lines that start with an item number and describe an item for sale.

=== EVERY ITEM NUMBER MUST HAVE A DESCRIPTION ===
- If you can see an item number on the sheet, you MUST output a line for it.
- If the description handwriting is unclear or you cannot read it, still output the item number followed by the word ILLEGIBLE.
- If the item description is crossed out but you can still read the number, output: NUMBER ILLEGIBLE CROSSEDOUT
- Never output just a bare item number with nothing after it — always put SOMETHING descriptive (even if it's just ILLEGIBLE).

=== FINAL CHECK BEFORE OUTPUTTING EACH LINE ===
- Every character on the line must be: A-Z, 0-9, or SPACE. Nothing else.
- No tabs. No lowercase. No punctuation at all. No brackets. No dashes. No slashes. No dollar signs.

Do NOT add commentary, do NOT add a "Transcription:" header, do NOT add column labels. Output only the transcribed lines.
"""

# Parallelism cap so we don't overrun API rate limits on huge docs
MAX_CONCURRENT = 8


# --------------------- helpers ---------------------

async def transcribe_image(image_bytes: bytes, media_type: str) -> str:
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{media_type};base64,{b64}"
    resp = await client.chat.completions.create(
        model="gpt-4o",
        max_tokens=8000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url, "detail": "high"},
                    },
                    {
                        "type": "text",
                        "text": "Transcribe all text on this page. Output only the transcription.",
                    },
                ],
            },
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()
    return sanitize_transcript(raw)


# ---- Post-processing safety net ----
# The AI is instructed to output only [A-Z0-9 \n], but we enforce it in code
# so the auction website never sees a stray comma, dash, or apostrophe.

def sanitize_line(text: str) -> str:
    if not text:
        return ""
    s = text.upper()
    # Strip tabs (we're single-column now) - convert to spaces
    s = s.replace("\t", " ")
    # Keep only A-Z, 0-9, and space. Everything else (including $, commas,
    # dashes, apostrophes, periods, etc.) becomes a space so JnJ never sees
    # a symbol in the description.
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    # Collapse runs of spaces into a single space
    s = re.sub(r" +", " ", s)
    return s.strip()

# Item number can be pure digits (6182) or letters+digits+optional letters (G6182, 10686FV).
# We use this to detect valid item lines.
ITEM_NUMBER_RE = re.compile(r'^([A-Z]*\d{3,}[A-Z]*)\b')

# Lot codes are specifically: 1-2 digits + 1 uppercase letter (18A, 17B, 21C, 33C),
# or a single uppercase letter (F, Z, P, C). These come between the item number
# and the actual description.
LOT_CODE_STRICT_RE = re.compile(r'^(\d{1,2}[A-Z]|[A-Z])$')

# The AI often glues the item number and a small following code together with
# no space, like "G198 1969" or "G200 19611" or "G205 P16". If the item number
# is followed by another all-digit or letter+digit chunk of 2-5 chars, and then
# real English words follow, that chunk is the lot code.
def strip_lot_code(line: str) -> str:
    """
    Remove the shelf/lot code that sits between the item number and the description.
    Examples:
      G6182 18A 2 MATTRESSES...  ->  G6182 2 MATTRESSES...
      G691 Z 3 COOLAROO...       ->  G691 3 COOLAROO...  (Z is shelf, 3 stays)
      G198 1969 KITCHENAID...    ->  G198 KITCHENAID...
      G205 P16 HUNTER GOLF...    ->  G205 HUNTER GOLF...
      G6182 2 MATTRESSES         ->  unchanged (2 is a quantity, not a lot code)
    """
    parts = line.split(" ")
    if len(parts) < 2:
        return line
    m = ITEM_NUMBER_RE.match(parts[0])
    if not m:
        return line
    kept = [parts[0]]
    i = 1
    stripped = 0
    while i < len(parts) and stripped < 2:
        tok = parts[i]
        # 1) Strict lot codes: single letter (F, Z, P, C) or digits+letter (18A, 17B, 21C)
        if LOT_CODE_STRICT_RE.match(tok):
            i += 1
            stripped += 1
            continue
        # 2) All-digit shelf codes 2-4 chars, ONLY if the very next token starts
        #    a real English word (letters-only, 3+ chars). Otherwise the digits
        #    are probably a quantity, dimension, or model number.
        if re.fullmatch(r'\d{2,4}', tok) and i + 1 < len(parts):
            next_tok = parts[i + 1]
            if re.fullmatch(r'[A-Z]{3,}', next_tok):
                i += 1
                stripped += 1
                continue
        # 3) Letter+digits like P16, A22, etc. (max 4 chars total) followed by a word
        if re.fullmatch(r'[A-Z]\d{1,3}', tok) and i + 1 < len(parts):
            next_tok = parts[i + 1]
            if re.fullmatch(r'[A-Z]{3,}', next_tok):
                i += 1
                stripped += 1
                continue
        break
    kept.extend(parts[i:])
    return " ".join(kept)


def sanitize_transcript(text: str) -> str:
    out_lines = []
    for line in text.splitlines():
        if not line.strip():
            continue  # drop blank lines entirely
        # Section markers like '--- Page 3 ---' pass through untouched
        if line.lstrip().startswith("---") and line.rstrip().endswith("---"):
            out_lines.append(line.strip())
            continue
        cleaned = sanitize_line(line)
        if not cleaned:
            continue
        # If this line starts with an item number, it's a new item row.
        # Strip lot codes and dedupe, then append.
        if ITEM_NUMBER_RE.match(cleaned):
            cleaned = strip_lot_code(cleaned)
            tokens = cleaned.split(" ")
            # Dedupe repeated leading item-number tokens ("6182 6182 ...")
            if len(tokens) >= 3 and tokens[0] == tokens[1]:
                cleaned = " ".join([tokens[0]] + tokens[2:])
                tokens = cleaned.split(" ")
            # Guarantee every item has SOMETHING after the number
            if len(tokens) < 2 or not " ".join(tokens[1:]).strip():
                cleaned = f"{tokens[0]} ILLEGIBLE"
            out_lines.append(cleaned)
            continue

        # Otherwise this line does NOT start with an item number.
        # It's most likely a continuation of the previous item's description
        # (the seller wrapped the text onto a second line on the sheet).
        # Merge it onto the last real item line instead of dropping it.
        # Look backwards for the last real item line (skip section markers).
        merge_idx = None
        for i in range(len(out_lines) - 1, -1, -1):
            prev = out_lines[i]
            if prev.startswith("---") and prev.endswith("---"):
                continue  # skip page markers
            if ITEM_NUMBER_RE.match(prev):
                merge_idx = i
                break
        if merge_idx is not None:
            # Append the continuation text with a single space.
            out_lines[merge_idx] = f"{out_lines[merge_idx]} {cleaned}"
        # If there's no previous item to attach to (rare — page starts with
        # continuation text), just drop the orphan line silently.
    return "\n".join(out_lines).strip()


def render_pdf_pages(pdf_bytes: bytes, dpi: int = 180) -> List[bytes]:
    import fitz
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    images: List[bytes] = []
    try:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            images.append(pix.tobytes("png"))
    finally:
        doc.close()
    return images


def is_blank_image(png_bytes: bytes, dark_ratio_threshold: float = 0.005) -> bool:
    """
    Blank-page detector: convert to grayscale, count "dark" pixels.
    Real content pages have at least ~0.5% dark pixels from handwriting/print.
    Blank scans have only speckle noise, well under that.
    """
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(png_bytes)).convert("L")
        img.thumbnail((400, 520))
        px = img.load()
        w, h = img.size
        dark = 0
        total = w * h
        for y in range(h):
            for x in range(w):
                if px[x, y] < 128:
                    dark += 1
        ratio = dark / total if total else 0
        return ratio < dark_ratio_threshold
    except Exception:
        return False


# --------------------- streaming endpoint ---------------------

@app.post("/api/extract-stream")
async def extract_stream(file: UploadFile = File(...)):
    """
    Streams NDJSON progress events. Frontend reads line-by-line.
    Events:
      {"type":"start", "pages": N}
      {"type":"skip",  "page": i, "reason":"blank"}
      {"type":"done_page", "page": i, "completed": k, "total_nonblank": m}
      {"type":"final", "text": "..."}
      {"type":"error", "message": "..."}
    """
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")

    ctype = (file.content_type or "").lower()
    name = (file.filename or "").lower()

    async def stream():
        try:
            # -------- PDF path (multi-page, parallel) --------
            if ctype == "application/pdf" or name.endswith(".pdf"):
                images = render_pdf_pages(data)
                total = len(images)
                yield json.dumps({"type": "start", "pages": total}) + "\n"

                keep: List[Tuple[int, bytes]] = []
                for i, img in enumerate(images, 1):
                    if is_blank_image(img):
                        yield json.dumps({"type": "skip", "page": i, "reason": "blank"}) + "\n"
                    else:
                        keep.append((i, img))

                total_nonblank = len(keep)
                if total_nonblank == 0:
                    yield json.dumps({"type": "final", "text": "(no readable text found)"}) + "\n"
                    return

                sem = asyncio.Semaphore(MAX_CONCURRENT)
                results: dict[int, str] = {}
                completed = 0
                completed_lock = asyncio.Lock()
                progress_q: asyncio.Queue = asyncio.Queue()

                async def worker(page_num: int, img: bytes):
                    nonlocal completed
                    async with sem:
                        try:
                            text = await transcribe_image(img, "image/png")
                        except Exception as e:
                            text = f"[error transcribing page {page_num}: {e}]"
                        results[page_num] = text
                        async with completed_lock:
                            completed += 1
                            done_now = completed
                        await progress_q.put(
                            {"type": "done_page", "page": page_num,
                             "completed": done_now, "total_nonblank": total_nonblank}
                        )

                tasks = [asyncio.create_task(worker(p, img)) for p, img in keep]

                async def waiter():
                    await asyncio.gather(*tasks)
                    await progress_q.put(None)

                waiter_task = asyncio.create_task(waiter())

                while True:
                    evt = await progress_q.get()
                    if evt is None:
                        break
                    yield json.dumps(evt) + "\n"

                await waiter_task

                parts = []
                use_headers = total_nonblank > 1
                for page_num in sorted(results.keys()):
                    t = results[page_num].strip()
                    if not t:
                        continue
                    parts.append(f"--- Page {page_num} ---\n{t}" if use_headers else t)
                final_text = "\n\n".join(parts) if parts else "(no readable text found)"
                yield json.dumps({"type": "final", "text": final_text}) + "\n"
                return

            # -------- single-image path --------
            if ctype.startswith("image/") or name.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")):
                media_type = ctype if ctype.startswith("image/") else "image/png"
                if media_type not in ("image/png", "image/jpeg", "image/webp", "image/gif"):
                    media_type = "image/png"
                yield json.dumps({"type": "start", "pages": 1}) + "\n"
                text = await transcribe_image(data, media_type)
                yield json.dumps({"type": "done_page", "page": 1, "completed": 1, "total_nonblank": 1}) + "\n"
                yield json.dumps({"type": "final", "text": text or "(no readable text found)"}) + "\n"
                return

            # -------- text passthrough --------
            if ctype.startswith("text/") or name.endswith((".txt", ".md", ".csv")):
                txt = data.decode("utf-8", errors="replace")
                yield json.dumps({"type": "start", "pages": 1}) + "\n"
                yield json.dumps({"type": "final", "text": txt}) + "\n"
                return

            # -------- docx --------
            if name.endswith(".docx"):
                try:
                    from docx import Document
                    doc = Document(io.BytesIO(data))
                    txt = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
                    yield json.dumps({"type": "start", "pages": 1}) + "\n"
                    yield json.dumps({"type": "final", "text": txt or "(no text found)"}) + "\n"
                    return
                except Exception:
                    yield json.dumps({"type": "error", "message": "could not read .docx"}) + "\n"
                    return

            yield json.dumps({"type": "error", "message": f"unsupported file type: {ctype or 'unknown'}"}) + "\n"

        except Exception as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/api/health")
def health():
    return {"ok": True}
