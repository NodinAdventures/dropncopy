"""
File Retype backend
- Skips blank pages via pixel-darkness check before hitting the AI
- Runs page transcriptions in parallel
- Streams live progress via SSE
"""

import asyncio
import base64
import csv
import io
import json
import re
import uuid
import zipfile
from typing import List, Tuple, Dict, Any, Optional

import os

from openai import AsyncOpenAI
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse, FileResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

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

=== LOT NUMBERS (VERY IMPORTANT — KEEP THEM SEPARATE FROM THE ITEM NUMBER) ===
After the item number, sheets have a LOT NUMBER in a separate column labeled "OFFICE USE ONLY" or similar. This looks like "18A", "17B", "21C", "F", "Z", "P", "15C", "9B", "2046", "204B", or a letter+digit like "Z 1", "P 3".

CRITICAL RULE: The item number and the lot number are in TWO SEPARATE COLUMNS on the sheet. They must appear as TWO SEPARATE TOKENS in the output, with a SINGLE SPACE between them. NEVER glue them together into one token.

Wrong: "79428A KITCHENAID MIXER"   (glued — this breaks the CSV)
Right: "7942 18A KITCHENAID MIXER"  (separated by a space)

Even if the lot number is written tight against the item number on the sheet, or is circled and touches the item column, you must output them with a space between.

THESE ARE REQUIRED and must appear in the output IMMEDIATELY AFTER the item number, separated by a single space, and BEFORE the description.

Format: ITEM_NUMBER LOT_CODE DESCRIPTION

Examples:
  - Sheet row: "G6182  18A  2 - Matts + Lg grill cover"  →  "G6182 18A 2 MATTRESSES AND LARGE GRILL COVER"
  - Sheet row: "G6183  18A  3 unused faucet covers"      →  "G6183 18A 3 UNUSED FAUCET COVERS"
  - Sheet row: "G6184  F    apox 3' tall"                →  "G6184 F APPROXIMATELY 3 TALL"
  - Sheet row: "G6185  17B  Comforter"                   →  "G6185 17B COMFORTER"
  - Sheet row: "G6186  18C  2 aprons"                    →  "G6186 18C 2 APRONS"
  - Sheet row: "G6187  21C  Pink + Black"                →  "G6187 21C PINK AND BLACK"
  - Sheet row: "G6188  18A  Java Seat"                   →  "G6188 18A JAVA SEAT"

If you see two chunks between the item number and the description (like "Z 3" or "P 16"), merge them WITHOUT a space — they form ONE lot code:
  - Sheet row: "G691  Z 3  Outdoor Roller Shade"         →  "G691 Z3 OUTDOOR ROLLER SHADE"
  - Sheet row: "G205  P 16  Hunter Golf Cart"            →  "G205 P16 HUNTER GOLF CART"

The lot number is always in the second column of the sheet, right after the item number column, and BEFORE the description column. Read them from the columns exactly as laid out.

If a row has NO lot number written (the office-use column is empty for that row), just output the item number and description with no lot code in between.

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


async def is_intake_sheet(image_bytes: bytes, media_type: str) -> bool:
    """Ask the AI whether an image is a JnJ intake sheet (grid of item#/description
    rows) vs. a photo of a physical item. Returns True only for a clear yes.
    Used when no PDF was uploaded and we need to pick which image is the sheet.
    """
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{media_type};base64,{b64}"
    try:
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=10,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url, "detail": "low"},
                        },
                        {
                            "type": "text",
                            "text": (
                                "Is this a paper intake sheet with a grid of "
                                "handwritten rows (item numbers and item "
                                "descriptions)? A sheet has printed headers "
                                "like SELLERS NAME, LOT, DESCRIPTION, or "
                                "OFFICE USE ONLY and multiple rows of writing. "
                                "A photo of a single physical item is NOT a "
                                "sheet. Answer only YES or NO."
                            ),
                        },
                    ],
                },
            ],
        )
        answer = (resp.choices[0].message.content or "").strip().upper()
        return answer.startswith("YES")
    except Exception:
        return False


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
    PRESERVE the lot code between the item number and the description.
    JnJ needs the lot number kept as: ITEM_NUMBER LOT_CODE DESCRIPTION

    This function normalizes multi-token lot codes (like "Z 3" or "P 16") into
    a single joined token ("Z3", "P16") so the output has a clean 2-token prefix.

    Examples:
      G6182 18A 2 MATTRESSES...  ->  unchanged (already clean)
      G691 Z 3 COOLAROO...       ->  G691 Z3 COOLAROO...
      G205 P 16 HUNTER GOLF...   ->  G205 P16 HUNTER GOLF...
      G6182 2 MATTRESSES         ->  unchanged (no lot code present)
    """
    parts = line.split(" ")
    if len(parts) < 2:
        return line
    m = ITEM_NUMBER_RE.match(parts[0])
    if not m:
        return line

    # Look at token 1 (right after item number). If it looks like a lot code,
    # keep it. If token 1 is a single letter (Z, F, P) and token 2 is short
    # digits (like "3", "16"), MERGE them into one lot code ("Z3", "P16").
    kept = [parts[0]]
    i = 1
    if i < len(parts):
        tok = parts[i]
        # Case A: single letter lot code (F, Z, P, C, etc.) possibly followed
        # by a small number that belongs with it.
        if re.fullmatch(r'[A-Z]', tok) and i + 1 < len(parts) and re.fullmatch(r'\d{1,3}', parts[i + 1]):
            # But only merge if the number is a small "sub-position" (1-3 digits)
            # AND there's real description text after. Check that parts[i+2] exists
            # and looks like a word (not another digit that would indicate quantity).
            if i + 2 < len(parts):
                after = parts[i + 2]
                # If the token after the digit is a word (has letters), merge Z + 3 -> Z3
                if any(c.isalpha() for c in after):
                    kept.append(tok + parts[i + 1])
                    i += 2
                else:
                    # Just a lone letter followed by numbers (rare) — keep letter only
                    kept.append(tok)
                    i += 1
            else:
                kept.append(tok)
                i += 1
        # Case B: standard lot code (18A, 17B, 21C, F, Z alone) — keep as-is
        elif LOT_CODE_STRICT_RE.match(tok):
            kept.append(tok)
            i += 1
        # Case C: not a lot code (probably a quantity or start of description) —
        # just leave everything alone.
    # Append the rest of the description untouched
    while i < len(parts):
        kept.append(parts[i])
        i += 1
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


# --------------------- JnJ CSV export ---------------------

# Exact column order for JnJ's bulk-import CSV, from the sample they provided.
JNJ_CSV_COLUMNS = [
    "Seller", "Category", "ListingFormat", "Title", "Description",
    "ItemLocation", "ZipCode", "Quantity", "PreferredCurrency", "Price",
    "StartBid", "Reserve", "BuyItNowPrice", "PaymentProcess", "PaymentInstr",
    "PayPalEmail", "BuyersPremiumPct", "IsTaxable", "TaxPercent",
    "StartDate", "EndDate", "Duration", "ReList",
    "HomePageFeatured", "CategoryFeatured", "HighlightListing",
    "BoldListing", "GalleryListing", "HitCounterStyle",
    "image_1", "image_2", "image_3", "image_4", "image_5",
    "image_6", "image_7", "image_8", "image_9", "image_10",
    "image_11", "image_12", "image_13", "image_14", "image_15",
    "image_16", "image_17", "image_18", "image_19", "image_20",
    "cf_SellerID",
]

# Default field values matching JnJ's sample CSV (April 16 Q Sale).
# Reference: uploaded sample 1-2.csv row 2:
#   FREMONT,APRIL 16 ~ Q SALE ,AUCTION,7936AS TALL FREESTANDING JEWLERY BOX,
#   TALL FREESTANDING JEWLERY BOX,|UNITED STATES|MICHIGAN|,49412,,,,$1.00 ,,,,,,10,,6,...
JNJ_DEFAULTS = {
    "Seller": "FREMONT",
    "ListingFormat": "AUCTION",
    "ItemLocation": "|UNITED STATES|MICHIGAN|",
    "ZipCode": "49412",
    "StartBid": "$1.00 ",
    "BuyersPremiumPct": "10",
    "TaxPercent": "6",
}


def parse_item_lines(transcript: str) -> List[Tuple[str, str, str]]:
    """
    Parse cleaned transcript into (item_number, lot_code, description) tuples.
    Skips page markers and empty lines.

    Input line examples:
      "G6182 18A 2 MATTRESSES AND LARGE GRILL COVER"
      "7942 2046 ROCKER"
      "6199 ILLEGIBLE CROSSEDOUT"       (no lot code)
      "G100 KITCHENAID MIXER"          (no lot code)

    Returns list of (item_num, lot_code_or_empty, description).
    """
    items = []
    for raw in transcript.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("---") and line.endswith("---"):
            continue
        if not ITEM_NUMBER_RE.match(line):
            continue
        parts = line.split(" ")
        if len(parts) < 2:
            continue
        item_num = parts[0]
        # Check if parts[1] is a lot code from the OFFICE USE ONLY column.
        # Real JnJ lot codes observed in intake sheets:
        #   Letter+digit:  18A, 17B, 21C, 204B, 15C, 9B, 51E, 79B, 60C
        #   Single letter: F, K, Z, O, P
        #   Letter+digits: Z3, P16
        #   Pure digits:   2046, 730, 2010, 200 (yes — also lot codes on some sheets)
        # A lot code is short (1-5 chars) of just A-Z / 0-9. Descriptions almost always
        # start with a common English word (ROCKER, DRESSER, KITCHENAID, etc.) which
        # doesn't match this pattern, so short alphanumeric tokens right after the
        # item number are safe to treat as lot codes.
        rest = parts[1:]
        lot_code = ""
        if rest:
            tok = rest[0]
            if 1 <= len(tok) <= 5 and re.fullmatch(r'[A-Z0-9]+', tok):
                lot_code = tok
                rest = rest[1:]
        description = " ".join(rest).strip()
        if not description:
            description = "ILLEGIBLE"
        items.append((item_num, lot_code, description))
    return items


def build_jnj_csv_row(item_num: str, lot_code: str, description: str,
                     sale_name: str, seller_id: str, seller_seq: int) -> dict:
    """
    Build one row of the JnJ CSV.

    Title format matches JnJ's real sample CSV:
      '7936AS TALL FREESTANDING JEWLERY BOX'   (no lot code -> AS)
      '7937AS WINCHESTER 12 GUN GUN SAFE'      (no lot code -> AS)
      '7938A15C HAMMS BEER SIGN'               (lot code 15C -> A15C)
      '7939A9B HOMEDICS FOOT MASSAGER NEW'     (lot code 9B  -> A9B)

    Pattern: {item_num}A{lot_code} {DESCRIPTION}
    When no lot code:   {item_num}AS {DESCRIPTION}

    Title is capped at 60 characters per JnJ's spec (Admin CSV Help column D).

    Description column = just the plain description (no item# or lot code).

    cf_SellerID = the seller's ID from the sheet header (e.g. 'AA1961').
    All rows on one intake sheet share the same seller ID.
    """
    if lot_code:
        title = f"{item_num}A{lot_code} {description}"
    else:
        title = f"{item_num}AS {description}"
    # Enforce Title max 60 chars per JnJ spec
    if len(title) > 60:
        title = title[:60].rstrip()

    row = {col: "" for col in JNJ_CSV_COLUMNS}
    row.update(JNJ_DEFAULTS)
    row["Category"] = sale_name
    row["Title"] = title
    row["Description"] = description
    # StartBid must have a value for auction listing format
    row["StartBid"] = "$1.00 "
    # cf_SellerID: use seller_id directly (all rows on one sheet share it).
    # For backward compat, if seller_seq differs from seller_id's own trailing digits
    # (i.e. caller passed a plain prefix + sequence), fall back to old behavior.
    if seller_id and any(c.isdigit() for c in seller_id):
        # seller_id already has digits (like 'AA1961') -> use as-is for all rows
        row["cf_SellerID"] = seller_id
    elif seller_id:
        # seller_id is a prefix only (like 'AA') -> append the sequence
        row["cf_SellerID"] = f"{seller_id}{seller_seq}"
    else:
        row["cf_SellerID"] = ""
    return row


@app.post("/api/export-jnj-csv")
async def export_jnj_csv(
    transcript: str = Form(...),
    sale_name: str = Form(""),
    seller_id: str = Form(""),
    seller_start: int = Form(1000),
):
    """
    Convert a cleaned transcript into a JnJ bulk-import CSV.
    Returns the CSV as a downloadable file.

    Form fields:
      transcript   - the cleaned text output from /api/extract-stream
      sale_name    - e.g. "APRIL 16 ~ Q SALE"  (goes in Category column)
      seller_id    - e.g. "AA"  (prefix for cf_SellerID)
      seller_start - starting sequence number (default 1000, so AA1000, AA1001...)
    """
    items = parse_item_lines(transcript)
    if not items:
        raise HTTPException(400, "No item rows found in the transcript.")

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=JNJ_CSV_COLUMNS, quoting=csv.QUOTE_MINIMAL)
    writer.writeheader()
    for idx, (item_num, lot_code, description) in enumerate(items):
        row = build_jnj_csv_row(
            item_num, lot_code, description,
            sale_name, seller_id, seller_start + idx,
        )
        writer.writerow(row)

    csv_bytes = output.getvalue().encode("utf-8-sig")  # BOM helps Excel open UTF-8 cleanly
    # Filename: jnj-<sale-slug>-<n>.csv
    slug = re.sub(r"[^A-Za-z0-9]+", "-", sale_name.strip()).strip("-").lower() or "export"
    filename = f"jnj-{slug}-{len(items)}items.csv"
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/health")
def health():
    return {"ok": True}


# =========================================================================
# JnJ Sale Builder: sheet + photos → preview → import-ready zip
# =========================================================================
#
# Flow:
#   1. POST /api/jnj-build   → upload sheet + photos, get JSON preview
#      { items: [{item_num, lot_code, description}, ...],
#        photos: [{id, filename, tag_read, matched_item_num, match_kind}, ...] }
#   2. Frontend renders preview; user can drag photos between items or click retry
#   3. POST /api/jnj-rematch → optional: re-run AI on one photo
#   4. POST /api/jnj-zip     → upload the final item-list JSON + all photo files,
#      returns a zip: {items.csv, <renamed photo files>}
#
# Photos are transient — the frontend keeps the actual bytes and re-uploads
# them at zip time. This avoids storing files on the ephemeral Render box.
# =========================================================================

# --- Photo tag reading ---

JNJ_PHOTO_SYSTEM_PROMPT = """You look at photos of items at an estate/consignment auction. Each item has a paper tag with an item number (like 7942, G6182, 10686FV). Sometimes the tag is clearly visible in the photo. Sometimes there is no tag at all, or the tag is unreadable.

Your job:
1. Look for a paper tag with a number in the photo.
2. If you see a clearly readable item number on a tag, respond with EXACTLY that number (e.g. "7942" or "G6182"). Read it exactly as written — keep any letter prefix like G, F, or suffix.
3. If you do NOT see a readable item-number tag, respond with EXACTLY "NO_TAG" and then a brief 2-8 word description of the item on the next line (like "NO_TAG\nWooden rocking chair").

Do not guess. If a tag is blurry or partially hidden, say NO_TAG. Only return an item number if you are confident.

Response format (item number found):
  7942

Response format (no tag):
  NO_TAG
  wooden rocking chair
"""

async def read_photo_tag(image_bytes: bytes, media_type: str, pre_shrunk: bool = False) -> Dict[str, str]:
    """Ask the vision model to read an item-number tag or fall back to a description.

    If pre_shrunk=True, image_bytes are already <=1024px JPEG and we skip the
    PIL decode step (saves ~30MB of RAM per call — crucial on Render Free tier).

    Returns:
      {'tag': 'G6182'}                        - if a tag was read
      {'tag': '', 'description': 'rocker'}     - if no tag but got description
    """
    try:
        if pre_shrunk:
            small_bytes = image_bytes
        else:
            # Downscale big photos to keep API calls fast and cheap. Vision handles
            # 1024px just fine for reading item tags.
            with Image.open(io.BytesIO(image_bytes)) as img:
                img = img.convert("RGB")
                img.thumbnail((1024, 1024))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                small_bytes = buf.getvalue()
        b64 = base64.standard_b64encode(small_bytes).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{b64}"

        # v14: switched from gpt-4o → gpt-4o-mini. Mini is ~4× faster and
        # ~15× cheaper, plenty accurate for reading a 3-digit tag number.
        # Also shortened the prompt — mini burns fewer tokens on short prompts.
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=20,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}},
                        {"type": "text", "text": "Is there a small paper price tag with a handwritten item number visible? If yes, reply with ONLY that number (e.g. 7943). If no tag or unclear, reply with ONLY the word: NONE"},
                    ],
                },
            ],
        )
        raw = (resp.choices[0].message.content or "").strip().upper()
        # Simplified for v14 mini prompt: response is either a tag number or NONE.
        if not raw or raw.startswith("NONE") or raw.startswith("NO"):
            return {"tag": "", "description": ""}
        # Extract a tag — typically 3-5 digits, possibly with a letter prefix/suffix.
        tag_match = re.search(r"([A-Z]?\d{3,}[A-Z]*)", raw)
        if tag_match:
            return {"tag": tag_match.group(1), "description": ""}
        return {"tag": "", "description": ""}
    except Exception as e:
        # Best-effort: don't fail the whole build if one photo errors
        return {"tag": "", "description": "", "error": str(e)}


async def match_photo_by_description(photo_desc: str, items: List[Dict]) -> Optional[str]:
    """Given a short description like 'wooden rocking chair' and a list of
    items, ask the AI to pick the best-matching item_num. Returns the item_num
    string or None."""
    if not photo_desc or not items:
        return None

    # Build a compact list for the prompt
    lines = [f"{i['item_num']}: {i['description']}" for i in items]
    prompt = (
        "A photo shows: " + photo_desc + "\n\n"
        "Which of these auction items best matches the photo? Respond with ONLY the item number, or NONE if no match is clear.\n\n"
        + "\n".join(lines)
    )
    try:
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=30,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = (resp.choices[0].message.content or "").strip().upper()
        if raw == "NONE" or not raw:
            return None
        m = re.match(r"^([A-Z]?\d+[A-Z]*)", raw)
        if not m:
            return None
        candidate = m.group(1)
        # Only return if it's actually in our item list
        valid = {i["item_num"] for i in items}
        return candidate if candidate in valid else None
    except Exception:
        return None


def parse_items_from_transcript(transcript: str) -> List[Dict[str, str]]:
    """Wrap parse_item_lines to return dicts (easier for JSON)."""
    tuples = parse_item_lines(transcript)
    return [
        {"item_num": t[0], "lot_code": t[1], "description": t[2]}
        for t in tuples
    ]


async def transcribe_uploaded_sheet(sheet: UploadFile) -> str:
    """Transcribe an uploaded sheet (PDF or image) to a cleaned transcript.
    Reuses the existing image / PDF logic without SSE streaming.

    Pages are transcribed in PARALLEL to stay under Render's 30s proxy cap.
    """
    data = await sheet.read()
    fname = (sheet.filename or "").lower()
    if fname.endswith(".pdf"):
        page_bytes_list = render_pdf_pages(data, dpi=180)
        non_blank = [pb for pb in page_bytes_list if not is_blank_image(pb)]
        if not non_blank:
            return ""
        page_texts = await asyncio.gather(
            *[transcribe_image(pb, "image/png") for pb in non_blank]
        )
        return "\n".join(page_texts)
    else:
        # image path
        media_type = sheet.content_type or "image/jpeg"
        if not media_type.startswith("image/"):
            media_type = "image/jpeg"
        text = await transcribe_image(data, media_type)
        return text


@app.post("/api/jnj-build")
async def jnj_build(files: List[UploadFile] = File(...)):
    """Analyze a JnJ sale intake.

    Inputs (multipart):
      files - one PDF/image sheet + N item photos, all in one upload

    Auto-detects which file is the sheet:
      - .pdf                          → sheet
      - single largest image          → sheet (heuristic, fallback)
      - explicit filename hints ('sheet', 'file', 'intake')

    Response JSON:
      {
        transcript: "...cleaned...",
        items: [{item_num, lot_code, description}, ...],
        photos: [{id, filename, thumb_data_url, tag_read, item_num_match, match_kind}, ...]
      }

    match_kind is one of: 'tag' | 'desc' | 'none'
    """
    if not files:
        raise HTTPException(400, "No files uploaded.")

    try:
        return await _jnj_build_inner(files)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"JNJ-BUILD FATAL ERROR: {e}\n{tb}", flush=True)
        # Return the actual error to the client so it's visible on mobile.
        raise HTTPException(500, f"{type(e).__name__}: {str(e)[:400]}")


async def _jnj_build_inner(files: List[UploadFile]) -> JSONResponse:
    # Separate the sheet from the photos.
    sheet: Optional[UploadFile] = None
    photos: List[UploadFile] = []
    # First pass: any PDF file is the sheet. Check BOTH filename extension AND
    # content-type, because iOS Safari sometimes uploads PDFs with mangled
    # filenames (no .pdf extension) but the correct application/pdf mime type.
    for f in files:
        name = (f.filename or "").lower()
        ctype = (f.content_type or "").lower()
        is_pdf = name.endswith(".pdf") or ctype == "application/pdf"
        if is_pdf and sheet is None:
            sheet = f
        else:
            photos.append(f)
    # Second pass: if still no sheet, use filename hints.
    if sheet is None:
        for i, f in enumerate(photos):
            n = (f.filename or "").lower()
            if any(hint in n for hint in ["sheet", "intake", "file "]) or n.startswith("file"):
                sheet = photos.pop(i)
                break
    # Third pass: AI-detects which image is the sheet by looking at each one
    # briefly. The intake sheet has "SELLERS NAME", "LOT DESCRIPTION", or
    # "OFFICE USE ONLY" printed as headers, so it's easy to identify.
    # This replaces the old "largest file = sheet" heuristic which was wrong
    # (item photos from iPhones are 2-4 MB, single-page sheet PDFs are ~200 KB).
    if sheet is None and photos:
        for i, f in enumerate(photos):
            data = await f.read()
            await f.seek(0)
            media_type = f.content_type or "image/jpeg"
            if not media_type.startswith("image/"):
                media_type = "image/jpeg"
            try:
                is_sheet = await is_intake_sheet(data, media_type)
            except Exception:
                is_sheet = False
            if is_sheet:
                sheet = photos.pop(i)
                break

    if sheet is None:
        raise HTTPException(400, "Couldn't identify a sheet in the upload. Please include the intake sheet as a PDF or a clear photo of the whole page.")

    # 1) Transcribe the sheet
    transcript = await transcribe_uploaded_sheet(sheet)
    items = parse_items_from_transcript(transcript)
    if not items:
        raise HTTPException(400, f"Sheet transcribed but no item rows were parsed. Transcript: {transcript[:500]}")

    # 2) Process every photo in parallel: read bytes, get thumb, ask AI for tag
    async def process_photo(idx: int, photo: UploadFile) -> Dict:
        raw = await photo.read()
        # Build a small base64 thumb for the preview UI (200x200)
        try:
            img = Image.open(io.BytesIO(raw))
            img = img.convert("RGB")
            img.thumbnail((200, 200))
            tbuf = io.BytesIO()
            img.save(tbuf, format="JPEG", quality=75)
            thumb_b64 = base64.standard_b64encode(tbuf.getvalue()).decode("utf-8")
            thumb_data_url = f"data:image/jpeg;base64,{thumb_b64}"
        except Exception:
            thumb_data_url = ""

        # Ask AI for tag / description
        media_type = photo.content_type or "image/jpeg"
        if not media_type.startswith("image/"):
            media_type = "image/jpeg"
        read = await read_photo_tag(raw, media_type)

        return {
            "id": f"p{idx}",
            "filename": photo.filename or f"photo_{idx}.jpg",
            "thumb_data_url": thumb_data_url,
            "tag_read": read.get("tag", ""),
            "description_read": read.get("description", ""),
        }

    photo_infos = await asyncio.gather(*[process_photo(i, p) for i, p in enumerate(photos)])

    # 3) Match photos to items
    valid_items = {i["item_num"] for i in items}

    # First pass: tag matches (highest confidence)
    for p in photo_infos:
        tag = p.get("tag_read", "")
        if tag and tag in valid_items:
            p["item_num_match"] = tag
            p["match_kind"] = "tag"
        else:
            p["item_num_match"] = ""
            p["match_kind"] = "none"

    # Second pass: description matches for photos that didn't get a tag hit
    async def desc_match(p):
        if p["match_kind"] != "none":
            return
        desc = p.get("description_read", "")
        if not desc:
            return
        matched = await match_photo_by_description(desc, items)
        if matched:
            p["item_num_match"] = matched
            p["match_kind"] = "desc"

    await asyncio.gather(*[desc_match(p) for p in photo_infos])

    return JSONResponse({
        "transcript": transcript,
        "items": items,
        "photos": photo_infos,
    })


@app.post("/api/jnj-build-sheet")
async def jnj_build_sheet(sheet: UploadFile = File(...)):
    """Step 1 of the split flow: transcribe the sheet ONLY (fast, ~5–15s).
    Returns the parsed items so the client can immediately show them.
    """
    try:
        transcript = await transcribe_uploaded_sheet(sheet)
        items = parse_items_from_transcript(transcript)
        if not items:
            raise HTTPException(400, f"Sheet transcribed but no item rows were parsed. Transcript: {transcript[:400]}")
        return JSONResponse({"transcript": transcript, "items": items})
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"JNJ-BUILD-SHEET FATAL: {e}\n{traceback.format_exc()}", flush=True)
        raise HTTPException(500, f"{type(e).__name__}: {str(e)[:400]}")


@app.post("/api/jnj-match-photos")
async def jnj_match_photos(
    photos: List[UploadFile] = File(...),
    items_json: str = Form(...),
):
    """Step 2 of the split flow: process a BATCH of photos against a known
    items list. Kept small enough (<= ~8 photos) to finish under 30s on
    Render's default proxy timeout. Can be called multiple times.
    """
    try:
        items = json.loads(items_json)
        if not isinstance(items, list) or not items:
            raise HTTPException(400, "items_json must be a non-empty list.")

        # Memory-safe photo processing for Render Free tier (512MB limit).
        # Strategy: shrink each photo ONCE to a small JPEG, free the original
        # bytes and PIL bitmap immediately, then use the shrunk bytes for both
        # the thumbnail and the AI tag read. Process photos sequentially (not
        # via gather) so peak RAM stays bounded by ~2 photos in flight rather
        # than all N at once.
        import gc

        def compute_dhash_from_bytes(jpeg_bytes: bytes) -> str:
            """Cheap perceptual hash: decode a small JPEG in a fresh PIL image,
            shrink to 9x8 grayscale, compare adjacent pixels. Two photos of the
            same item have similar dhash; photos of different items have very
            different dhashes. Costs ~1ms per photo and returns a 16-char hex
            string. Isolated in its own decode so a PIL failure here can't
            corrupt the shared image used by the vision API.
            """
            try:
                with Image.open(io.BytesIO(jpeg_bytes)) as src:
                    small = src.convert("L").resize((9, 8), Image.LANCZOS)
                    # Use .tobytes() — works on all Pillow versions and doesn't
                    # have the getdata() deprecation. 72 bytes for a 9x8 L image.
                    pixels = small.tobytes()
                    small.close()
                if len(pixels) < 72:
                    return ""
                bits = 0
                for row in range(8):
                    for col in range(8):
                        left = pixels[row * 9 + col]
                        right = pixels[row * 9 + col + 1]
                        bits = (bits << 1) | (1 if left > right else 0)
                return f"{bits:016x}"
            except Exception as e:
                # Never let this abort the request — dhash is a best-effort
                # signal; if it's missing the client just won't detect scene
                # changes for that one photo.
                print(f"dhash failed: {type(e).__name__}: {e}", flush=True)
                return ""

        async def process_photo(idx: int, photo: UploadFile) -> Dict:
            raw = await photo.read()
            filename = photo.filename or f"photo_{idx}.jpg"

            # v15: Dave shoots ITEM → BLANK/BLACK PHOTO → next item. Blank photos
            # are hard delimiters between items. No AI, no dhash, no guessing.
            # We just need to detect "blank" (all dark OR all light with no
            # detail) and use that as a break marker.
            thumb_data_url = ""
            is_blank = False
            try:
                with Image.open(io.BytesIO(raw)) as img:
                    rgb = img.convert("RGB")
                    # Downsample to a tiny image for blank detection —
                    # a 32x32 grayscale has enough info to know if a photo is
                    # "nothing" vs a real subject.
                    tiny = rgb.copy().convert("L")
                    tiny.thumbnail((32, 32))
                    px = list(tiny.getdata())
                    tiny.close()
                    if px:
                        mean_lum = sum(px) / len(px)
                        # Standard deviation — measures how much variation there
                        # is in the image. A real photo has lots of variation.
                        # A blank/black/washed-out photo has almost none.
                        variance = sum((v - mean_lum) ** 2 for v in px) / len(px)
                        std_dev = variance ** 0.5
                        # Blank if: (all very dark AND low variation) OR
                        #           (all very bright AND low variation) OR
                        #           very low variation overall (blurry white/black)
                        if std_dev < 12:  # essentially uniform image
                            is_blank = True
                        elif mean_lum < 25 and std_dev < 20:  # black photo
                            is_blank = True
                        elif mean_lum > 230 and std_dev < 20:  # washed-out white
                            is_blank = True

                    # 160px thumb for the UI (base64 in JSON — keep small)
                    tbuf = io.BytesIO()
                    rgb.thumbnail((160, 160))
                    rgb.save(tbuf, format="JPEG", quality=70)
                    thumb_b64 = base64.standard_b64encode(tbuf.getvalue()).decode("utf-8")
                    thumb_data_url = f"data:image/jpeg;base64,{thumb_b64}"
                    rgb.close()
                    del rgb, tbuf, thumb_b64
            except Exception as e:
                print(f"process_photo shrink failed for {filename}: {e}", flush=True)

            del raw
            gc.collect()

            return {
                "id": f"p{idx}",
                "filename": filename,
                "thumb_data_url": thumb_data_url,
                "tag_read": "",
                "description_read": "",
                "dhash": "",
                "is_blank": is_blank,
            }

        # Process sequentially to keep peak memory low. On Render Free
        # (512MB), running 6 phone photos through asyncio.gather peaks around
        # 400MB — too close to the OOM cliff. Sequential adds ~5–10s to a batch
        # but avoids 502s from the worker being killed.
        # Each photo is wrapped in its own try/except so a single bad photo
        # can't take down the entire batch (which is what caused the SIGABRT
        # crash we saw in v9 — status 134 = native library abort).
        #
        # v12: photos in a batch run CONCURRENTLY via asyncio.gather since each
        # is waiting on network I/O (OpenAI vision call ~2-3s). Sequential
        # processing meant a batch of 8 took 8 * 3s = 24s; concurrent means the
        # slowest photo dominates (~4s). return_exceptions keeps one failure
        # from taking down the batch.
        async def safe_process(i: int, p: UploadFile) -> Dict:
            try:
                return await process_photo(i, p)
            except Exception as e:
                print(f"process_photo failed for photo {i} ({getattr(p,'filename','?')}): {type(e).__name__}: {e}", flush=True)
                return {
                    "id": f"p{i}",
                    "filename": getattr(p, "filename", f"photo_{i}.jpg") or f"photo_{i}.jpg",
                    "thumb_data_url": "",
                    "tag_read": "",
                    "description_read": "",
                    "dhash": "",
                    "error": f"{type(e).__name__}: {str(e)[:200]}",
                }

        # Fire all photos in this batch concurrently. Each is I/O bound (waiting
        # on OpenAI), so this collapses an 8-photo batch from ~24s to ~4s.
        photo_infos: List[Dict] = await asyncio.gather(
            *[safe_process(i, p) for i, p in enumerate(photos)]
        )

        valid_items = {i["item_num"] for i in items}

        for p in photo_infos:
            tag = p.get("tag_read", "")
            if tag and tag in valid_items:
                p["item_num_match"] = tag
                p["match_kind"] = "tag"
            else:
                p["item_num_match"] = ""
                p["match_kind"] = "none"

        # NOTE: description-based AI matching is DISABLED here — the client
        # runs order-based matching (proportional distribution using photo
        # position in sheet order) which is more accurate for Dave's workflow
        # and needs zero extra AI calls. Saves ~2–5s per photo and cuts
        # per-request memory in half. If we ever want to re-enable a
        # description-based fallback for photos that end up in the wrong
        # segment, add it here — but for now, less code = fewer OOMs.
        return JSONResponse({"photos": photo_infos})
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"JNJ-MATCH-PHOTOS FATAL: {e}\n{traceback.format_exc()}", flush=True)
        raise HTTPException(500, f"{type(e).__name__}: {str(e)[:400]}")


@app.get("/api/jnj-diag")
async def jnj_diag():
    """Quick health check to verify the JnJ endpoint is reachable and the
    OpenAI key is loaded. Returns 200 if all is well."""
    import os, sys as _sys
    return JSONResponse({
        "ok": True,
        "has_openai_key": bool(os.environ.get("OPENAI_API_KEY")),
        "python_version": _sys.version.split()[0],
    })


@app.post("/api/jnj-rematch")
async def jnj_rematch(
    photo_id: str = Form(...),
    description: str = Form(""),
    items_json: str = Form(...),
):
    """Retry description-based match for one photo. Client passes the photo's
    description_read and the current item list; we return a new item_num or empty."""
    try:
        items = json.loads(items_json)
    except Exception:
        raise HTTPException(400, "Bad items_json")
    if not description:
        return JSONResponse({"item_num_match": ""})
    matched = await match_photo_by_description(description, items)
    return JSONResponse({"item_num_match": matched or ""})


@app.post("/api/jnj-zip")
async def jnj_zip(
    sale_name: str = Form(""),
    seller_id: str = Form(""),
    seller_start: int = Form(1000),
    items_json: str = Form(...),
    photo_map_json: str = Form(...),
    photos: List[UploadFile] = File(default=[]),
):
    """Build the final import-ready zip.

    Inputs (multipart):
      sale_name       - category / sale name (e.g. 'AUGUST 27~ H SALE')
      seller_id       - prefix like 'AA'
      seller_start    - starting sequence number
      items_json      - JSON array of {item_num, lot_code, description} (post-edit)
      photo_map_json  - JSON dict { photo_filename: item_num, ... }
      photos          - the original photo files (uploaded again by the frontend)

    Zip contents:
      items.csv
      <sellerid>_<itemnum>_<seq>.<ext>   for every matched photo
    """
    try:
        items = json.loads(items_json)
        photo_map = json.loads(photo_map_json)  # {filename: item_num}
    except Exception as e:
        raise HTTPException(400, f"Bad JSON: {e}")

    if not items:
        raise HTTPException(400, "No items provided.")

    # Group photos by item_num, preserving upload order.
    photos_by_item: Dict[str, List[UploadFile]] = {}
    for p in photos:
        target = photo_map.get(p.filename or "")
        if not target:
            continue
        photos_by_item.setdefault(target, []).append(p)

    # Build the zip in memory
    zip_buf = io.BytesIO()
    csv_buf = io.StringIO()
    writer = csv.DictWriter(csv_buf, fieldnames=JNJ_CSV_COLUMNS, quoting=csv.QUOTE_MINIMAL)
    writer.writeheader()

    # For each item build a row and, if it has photos, name them + fill image_1..N
    photo_files: List[Tuple[str, bytes]] = []  # (name_in_zip, bytes)
    for idx, it in enumerate(items):
        item_num = it.get("item_num", "")
        lot_code = it.get("lot_code", "")
        description = it.get("description", "")
        row = build_jnj_csv_row(
            item_num, lot_code, description,
            sale_name, seller_id, seller_start + idx,
        )
        # Title cap is enforced inside build_jnj_csv_row.

        # Rename photos for this item using the same code that becomes the Title
        # prefix: {item_num}A{lot_code} or {item_num}AS. This keeps photo names
        # aligned with the CSV Title so JnJ staff can trace them.
        item_code = f"{item_num}A{lot_code}" if lot_code else f"{item_num}AS"
        item_photos = photos_by_item.get(item_num, [])
        for photo_idx, p in enumerate(item_photos[:20], start=1):
            ext = (p.filename or "photo.jpg").rsplit(".", 1)[-1].lower()
            if ext not in ("jpg", "jpeg", "png", "webp", "gif", "bmp", "heic"):
                ext = "jpg"
            new_name = f"{item_code}_{photo_idx:03d}.{ext}"
            data = await p.read()
            # Reset the file position so we don't consume it if it's used again
            await p.seek(0)
            photo_files.append((new_name, data))
            row[f"image_{photo_idx}"] = new_name

        writer.writerow(row)

    # Write the zip
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("items.csv", csv_buf.getvalue().encode("utf-8-sig"))
        for name, data in photo_files:
            zf.writestr(name, data)

    zip_bytes = zip_buf.getvalue()
    slug = re.sub(r"[^A-Za-z0-9]+", "-", sale_name.strip()).strip("-").lower() or "jnj-sale"
    filename = f"jnj-{slug}-{len(items)}items.zip"
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
