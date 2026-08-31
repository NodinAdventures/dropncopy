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
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

import os

from openai import AsyncOpenAI, RateLimitError, APIStatusError, APIConnectionError, APITimeoutError
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse, FileResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# v25.49: WeChat QR detector — dramatically more robust than the stock
# OpenCV QRCodeDetector. Works on tilted, wrinkled, low-contrast, and
# small-in-frame QR codes without needing model files. Import lazily so
# a missing lib doesn't crash startup.
try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    _HAS_CV2 = True
    _QR_DETECTOR = cv2.QRCodeDetector()  # stock, as fallback
    # WeChat model works WITHOUT the trained files — falls back to a
    # simpler detector but still much better than stock QRCodeDetector.
    try:
        _WECHAT_QR = cv2.wechat_qrcode.WeChatQRCode()
        _HAS_WECHAT = True
        print("v25.49: WeChatQRCode detector loaded", flush=True)
    except Exception as _wc_err:
        print(f"WeChatQRCode unavailable ({_wc_err}); using stock only", flush=True)
        _WECHAT_QR = None
        _HAS_WECHAT = False
except Exception as _cv_err:
    print(f"cv2 QR support disabled: {_cv_err}", flush=True)
    _HAS_CV2 = False
    _QR_DETECTOR = None
    _WECHAT_QR = None
    _HAS_WECHAT = False


def _wechat_decode(arr) -> str:
    """Try WeChat detector — returns first decoded text or ''"""
    if not _HAS_WECHAT or _WECHAT_QR is None:
        return ""
    try:
        results, _points = _WECHAT_QR.detectAndDecode(arr)
        for r in results:
            if r:
                return r
    except Exception as e:
        print(f"wechat decode failed: {type(e).__name__}: {e}", flush=True)
    return ""


def _stock_decode(arr) -> str:
    """Try stock OpenCV detector, both single and multi."""
    if _QR_DETECTOR is None:
        return ""
    try:
        data, _pts, _ = _QR_DETECTOR.detectAndDecode(arr)
        if data:
            return data
    except Exception:
        pass
    try:
        ok, datas, _pts, _ = _QR_DETECTOR.detectAndDecodeMulti(arr)
        if ok and datas:
            for d in datas:
                if d:
                    return d
    except Exception:
        pass
    return ""


def _try_all_detectors(arr) -> str:
    """Try WeChat first (much better), then stock as fallback."""
    text = _wechat_decode(arr)
    if text:
        return text
    return _stock_decode(arr)


def detect_divider_qr(raw_bytes: bytes) -> bool:
    """Return True if this photo contains the printable DIVIDER card.

    v25.50: FAST path. Real camera photos of the printed card decode
    with a single WeChat call at 1024px — no need for 12 detection
    passes. Total budget: ~50-100ms per photo. Only fall back to more
    thorough checks if the first pass misses.

    The card payload is 'DROPNCOPY-DIVIDER'. We match on this substring
    so both the single card and the numbered cards (DROPNCOPY-DIVIDER-001)
    both work.
    """
    if not _HAS_CV2:
        return False
    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            rgb = img.convert("RGB")
            work = rgb.copy()
            work.thumbnail((1024, 1024))
            gray = np.array(work.convert("L"))
            work.close()
            rgb.close()

        # Fast path: WeChat on grayscale. This catches ~95% of real
        # camera photos of the printed card in under 100ms.
        text = _wechat_decode(gray)
        if text and "DROPNCOPY-DIVIDER" in text:
            return True

        # Backup path: stock OpenCV detector. Adds maybe 30ms.
        text = _stock_decode(gray)
        if text and "DROPNCOPY-DIVIDER" in text:
            return True

        return False
    except Exception as e:
        print(f"QR detect failed: {type(e).__name__}: {e}", flush=True)
        return False


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

*** WHERE THE ITEM NUMBER LIVES ON THE SHEET ***
The item number is ALWAYS the number in the FAR-LEFT column of the row — the leftmost column of the table, farthest from the description. It is written on the left edge of the row, before the "OFFICE USE" or "Office Use" column, before any circled or boxed codes, and before the description text.

On some sheets the leftmost column has no header at all, or is labeled "Lot" or "Item" or "#". Whatever the label, the item number is the leftmost thing on the row.

Item numbers on ONE sheet are ALMOST ALWAYS SEQUENTIAL by 1 (for example 3060, 3061, 3062, 3063, 3064, 3065, 3066). If you have already read one row's item number, the next row's item number is normally the previous one PLUS 1. Use this to double-check when handwriting is ambiguous: if the previous row was 3062 and the next digit could be a 2 or a 6, it is almost certainly a 3 (3063), not a 3023. Do not report a lot number that is far out of sequence unless the handwriting VERY CLEARLY shows a jump.

Do NOT confuse the item number with:
  - The SELLER NUMBER in the boxed field at the TOP-RIGHT of the sheet (usually 3–5 digits, e.g. "3186" or "40416"). That is the seller ID for the WHOLE sheet, not any individual row.
  - The LOT / LOCATION code in the second column ("Office Use" or similar). That is short and usually has a letter (e.g. "40C", "41B", "55A", "79A") — output it as the lot code, NOT as the item number.
  - Any date, address, or page-number written elsewhere on the form.

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

**WHAT A VALID LOT CODE LOOKS LIKE (STRICT):**
A lot code is SHORT and follows one of these patterns:
  - 1-2 digits + 1 uppercase letter: 18A, 17B, 21C, 33C, 37G, 35B, 91B, 9B, 15C
  - A single uppercase letter: F, Z, P, C, Cr ("Cr" is 2 letters, still valid)
  - A letter + short digit merged: Z3, P16
  - Plain 3-4 digit numbers with an optional trailing letter: 2046, 204B

A lot code is NEVER an English word. "COBRA", "BRASS", "BOOK", "YARD", "LARGE", "SLEDS", "DIGITAL", "HOODED", "OFFICIAL" are NOT lot codes — they are the first word of the DESCRIPTION.

If the office-use / lot column on a row is BLANK, the row has NO lot code. Output the item number, then a space, then the description starting with whatever word the seller wrote first — even if that word looks like it could be a code (like "COBRA" or "BRASS"). DO NOT grab the first word of the description and pretend it is a lot code.

WRONG (do not do this):
  Sheet row: "8533  [blank]  Cobra Digital Radar + Laser Detector"
  Wrong output:   "8533 COBRA DIGITAL RADAR AND LASER DETECTOR"
     ^^^^ COBRA is NOT a lot code. The row has no lot code because the lot column is blank.

RIGHT:
  Sheet row: "8533  [blank]  Cobra Digital Radar + Laser Detector"
  Right output:   "8533 COBRA DIGITAL RADAR AND LASER DETECTOR"
     (the item number is followed directly by the description; no lot code appears because none was written)

Both outputs above look identical because they ARE identical — the difference is only in the DOWNSTREAM parser's interpretation. Our downstream parser sees a 2-token prefix and treats the second token as a lot code IF and ONLY IF it matches one of the valid lot-code patterns above. "COBRA" does not match any pattern, so it will correctly be treated as description — as long as you don't glue it to something that looks like a code.

So the practical rule is: when the lot column is blank on the paper, just write the item number followed by the seller's description. Do not pause and try to identify a lot code that isn't there.

If a row has NO lot number written (the office-use column is empty for that row), just output the item number and description with no lot code in between.

Read the handwriting VERY carefully. "Grill" and "Quilt" look similar in cursive — use context: "grill cover" makes more sense than "quilt cover" when paired with a mattress. Similarly "Faucet" makes more sense than "Facent".

=== ONE ROW = ONE LISTING (CRITICAL) ===
- Each row on the intake sheet is ONE auction listing, regardless of the quantity.
- If a row says "2 MATTRESSES", output ONE line that says "2 MATTRESSES" — it is being sold as a set.
- Do NOT split quantities into multiple output lines.
- The seller writes separate rows (each with its own item number) when they want items sold separately. Trust the sheet.

=== DITTO MARKS — SAME AS ABOVE (VERY IMPORTANT) ===
Sellers frequently use ditto marks (quotation marks) in the description column to mean "same item as the row above." You'll see it as `" " "`, `"  "`, `"  "  "`, ditto symbols, or sometimes just the word "same" or "ditto." The row still has its own item number and lot code (they are NEVER dittoed — always read those fresh) but the DESCRIPTION is inherited.

When you see ditto marks in the description column, output the ditto marks EXACTLY. Do NOT try to guess or copy the previous description — our downstream parser handles the expansion. Just output the item number, lot code, and the literal ditto marks.

Example sheet rows:
  3025  38C  LED Utility Light - Dusk to Dawn
  3026  37B  "   "   "                         ← ditto: same item, different lot
  3033  37B  Hand sanitizer - Great for camping
  3034  38C  "                                 ← ditto
  3035  37B  "                                 ← ditto

Output them as:
  3025 38C LED UTILITY LIGHT DUSK TO DAWN
  3026 37B " " "
  3033 37B HAND SANITIZER GREAT FOR CAMPING
  3034 38C "
  3035 37B "

NEVER skip a row because it only has ditto marks. NEVER merge a ditto row into the previous row — it's a separate listing with its own item number.

=== FRACTIONS & MEASUREMENTS (KEEP THEM READABLE) ===
Sellers often write fractional measurements like `8½ × 12`, `3¾ x 2⁸⁄₈ x 86½`, `72 X 72`, etc. Preserve these as READABLE ASCII fractions — do NOT drop the fraction or flatten it into separate digits.

Rules:
- `½` → `1/2`
- `¼` → `1/4`
- `¾` → `3/4`
- `⅓` → `1/3`, `⅔` → `2/3`
- `⅛` → `1/8`, `⅜` → `3/8`, `⅝` → `5/8`, `⅞` → `7/8`
- `×` (multiplication) → `X`
- Any handwritten fraction (stacked "1 over 2") → `1/2`

Examples:
  Sheet: `Bubble Mailer - 8½ × 12`               →  `BUBBLE MAILER 8 1/2 X 12`
  Sheet: `72 X 72 in 6 ft X 6 ft`                →  `72 X 72 IN 6 FT X 6 FT`
  Sheet: `Hardwood post - (3) 3¾ × 2⁸⁄₈ × 86½`   →  `HARDWOOD POST 3 3/4 X 2 7/8 X 86 1/2`

NEVER output `8 1 2 X 12` — that's the fraction split into three tokens and is unreadable. ALWAYS keep the fraction as one `N/N` unit like `1/2` or `3/4`.

=== MULTI-LINE DESCRIPTIONS (VERY IMPORTANT — READ CAREFULLY) ===
Sellers often write long descriptions that WRAP onto a second, third, or fourth line on the sheet. This is EXTREMELY common on JnJ intake sheets. When a line on the sheet has NO item number and NO lot code in the far-left columns, it is a CONTINUATION of the previous item's description — NOT a separate item.

**HOW TO IDENTIFY A CONTINUATION LINE VISUALLY (this is the key rule):**
- The far-left OFFICE USE ONLY column is EMPTY for that row (no item number like 8545 or G6182).
- The lot column right after it is EMPTY too (no lot code like 37G or 18A).
- The row usually has 'Lot' printed on it (from the form template) but no lot number written next to it.
- Only the description column has handwriting on that row.
- The handwriting is a continuation of the words from the row above — like UNOPENED continuing COBRA DIGITAL RADAR + LASER DETECTOR, or SLEDDERS + SKIERS continuing DECORATIVE SLED + BASKET OF X-MAS, or NEW LIKE continuing EVERY READY NICKEL CADMIUM BATTERY CHARGER.

**CRITICAL: NEVER invent an item number for a continuation line.**
- The sequential-by-1 rule (row N+1 = row N + 1) ONLY applies to rows where an item number was ACTUALLY WRITTEN on the sheet in the far-left column.
- If the far-left column is BLANK on a row, that row is NOT a new item — it is a continuation of the row above.
- DO NOT skip a row on the paper. DO NOT number a blank-left-column row just because it is the next in sequence. The seller left the left column blank on purpose — it means same item as above, more description below.

Rules for continuation lines:
- Read the item number ONCE from the far-left column of a row.
- Every following row where the far-left column is EMPTY belongs to that same item.
- MERGE all continuation lines into ONE output line for that item, joined by a single space.
- Only start a new output line when you see the NEXT item number ACTUALLY WRITTEN in the far-left column on a new row.

Examples of correct continuation handling:

  Sheet has:                                        Output:
  ---------                                         -------
  8533   [blank]  Cobra Digital Radar + Laser       8533 COBRA DIGITAL RADAR AND LASER DETECTOR UNOPENED
  [blank] [blank] Detector
  [blank] [blank] Unopened
  8534   37G      3 American Locks with Keys        8534 37G 3 AMERICAN LOCKS WITH KEYS

  8543   36B      Yard Art Noisy Frog + Metal       8543 36B YARD ART NOISY FROG AND METAL CAT CANDLE HOLDER
  [blank] [blank] Cat Candle
  [blank] [blank] Holder
  8544   36A      Decorative Sled + Basket of       8544 36A DECORATIVE SLED AND BASKET OF X-MAS SLEDDERS AND SKIERS
  [blank] [blank] X-Mas
  [blank] [blank] Sledders + Skiers
  8545   1B       Book Heidi 1927                   8545 1B BOOK HEIDI 1927

  G6182  18A      2 mattresses and                  G6182 18A 2 MATTRESSES AND LARGE GRILL COVER WITH ZIPPER
  [blank] [blank] large grill cover with zipper

  G691   Z3       Outdoor Roller Shade Coolaroo     G691 Z3 OUTDOOR ROLLER SHADE COOLAROO NEW IN BOX 72 X 72 IN MOCHA COLOR
  [blank] [blank] NIB 72x72 in
  [blank] [blank] Mocha color

DO NOT output continuation text on its own line. DO NOT drop continuation text. DO NOT invent an item number for a continuation row. Everything the seller wrote about that item goes on the SAME output line as the item number.

COMMON MISTAKE TO AVOID: Do NOT think 'the previous row was 8544 so the next row must be 8545.' That is only true when the sheet actually shows 8545 written in the far-left column. If the far-left column is blank, the row is a continuation, not the next item.

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

# Parallelism cap so we don't overrun API rate limits on huge docs.
# v25.65: temporarily lowered 8 -> 3 while account is on OpenAI Tier 1
# (30k TPM). 8 parallel gpt-4o high-detail calls could burst past 30k
# tokens in one minute and trigger 429s. Bump back to 8 once the
# account reaches Tier 2 (450k TPM, ~7 days after the $50 top-up).
MAX_CONCURRENT = 3


async def _openai_with_retry(coro_factory, *, max_attempts: int = 5, op_name: str = "openai"):
    """Call an OpenAI SDK coroutine with automatic retry on transient errors.

    v25.65: added to make sheet builds resilient to short 429 (rate limit)
    spikes without changing model, prompt, image bytes, or output. Retries
    ONLY on transient failures (429, connection/timeout, 5xx). Any real
    error (4xx other than 429) is re-raised immediately so callers still
    see genuine problems. Honors the server's Retry-After hint when present,
    otherwise uses exponential backoff (1s, 2s, 4s, 8s, capped at 15s).

    Pass a zero-arg lambda that creates the coroutine, e.g.:
        resp = await _openai_with_retry(
            lambda: client.chat.completions.create(model=..., messages=...),
            op_name="transcribe_image",
        )
    We need a factory (not a coroutine) because a coroutine can only be
    awaited once; on retry we must build a fresh one.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(max_attempts):
        try:
            return await coro_factory()
        except RateLimitError as e:
            last_exc = e
            # Honor server hint if present (headers.retry-after, in seconds)
            wait = 0.0
            try:
                hdr = getattr(e, "response", None)
                if hdr is not None:
                    ra = hdr.headers.get("retry-after") if hasattr(hdr, "headers") else None
                    if ra:
                        wait = float(ra)
            except Exception:
                wait = 0.0
            if wait <= 0:
                wait = min(2 ** attempt, 15)
            print(f"[{op_name}] 429 rate limit; retry {attempt + 1}/{max_attempts} in {wait:.1f}s")
            await asyncio.sleep(wait)
            continue
        except (APIConnectionError, APITimeoutError) as e:
            last_exc = e
            wait = min(2 ** attempt, 15)
            print(f"[{op_name}] connection/timeout; retry {attempt + 1}/{max_attempts} in {wait:.1f}s")
            await asyncio.sleep(wait)
            continue
        except APIStatusError as e:
            # Only retry on 5xx server errors; re-raise 4xx immediately.
            status = getattr(e, "status_code", None)
            if status is None or status < 500:
                raise
            last_exc = e
            wait = min(2 ** attempt, 15)
            print(f"[{op_name}] server {status}; retry {attempt + 1}/{max_attempts} in {wait:.1f}s")
            await asyncio.sleep(wait)
            continue
    # Exhausted retries -- surface the last error unchanged
    assert last_exc is not None
    raise last_exc


# --------------------- helpers ---------------------

def _crop_left_column_zoomed(image_bytes: bytes) -> Optional[bytes]:
    """v25.68: crop the leftmost 22% of the sheet and zoom it 2x so the
    item-number column becomes ~44% of the input width at 2x pixel density.
    Gives the AI a much bigger view of just the digits, which fixes closed-top
    3-vs-2 misreads and other subtle handwriting ambiguity. Returns PNG bytes,
    or None if cropping fails (in which case caller skips the second pass).
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("RGB")
            w, h = img.size
            # Grab a bit past the office-use column so we also see the lot code
            # for context. Real sheets: item column is ~0-15% of width, lot
            # column ~15-25%. Crop to 22% to be safe. Skip the top 10% (header)
            # and bottom 5% (blank rows) to reduce distraction.
            left = img.crop((0, int(h * 0.10), int(w * 0.22), int(h * 0.95)))
            # Zoom 2x for the model. Bicubic gives clean digit edges.
            zoomed = left.resize(
                (left.size[0] * 2, left.size[1] * 2), Image.BICUBIC
            )
            buf = io.BytesIO()
            zoomed.save(buf, format="PNG", optimize=True)
            return buf.getvalue()
    except Exception as e:
        print(f"[crop_left_column] failed: {e}")
        return None


async def _read_left_column_numbers(image_bytes: bytes) -> List[str]:
    """v25.68: dedicated second-pass read of just the item-number column
    at 2x zoom. Returns a list of item numbers in row order, or [] if the
    call fails. This is the reconciliation source of truth for cases where
    the full-page transcription misread the tens/hundreds digit.
    """
    cropped = _crop_left_column_zoomed(image_bytes)
    if cropped is None:
        return []
    b64 = base64.standard_b64encode(cropped).decode("utf-8")
    data_url = f"data:image/png;base64,{b64}"
    try:
        resp = await _openai_with_retry(
            lambda: client.chat.completions.create(
                model="gpt-4o",
                max_tokens=800,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url, "detail": "high"},
                            },
                            {
                                "type": "text",
                                "text": (
                                    "This is the left edge of a handwritten auction intake sheet, zoomed 2x. "
                                    "Read ONLY the item numbers written in the far-left OFFICE USE ONLY column, "
                                    "in top-to-bottom order. "
                                    "These numbers are almost always sequential by 1 (e.g. 8531, 8532, 8533, 8534). "
                                    "CRITICAL: handwritten 3s often have a closed loopy top that looks like a 2. "
                                    "If you see a run of numbers where the last digit alternates (like 8521, 8532, 8533, 8534), "
                                    "the '8521' is almost certainly a misread '8531' - fix it. "
                                    "Same for 6s that look like 2s (loopy top): if the sequence is 3062, 3023, 3064 you know 3023 is really 3063. "
                                    "CRITICAL 0-vs-1: handwritten 0s written tightly can look like 1s. If you see 1001, 1001, 1002, 1003 (duplicate first row), the first row is almost certainly 1000. If row 2 equals what you read for row 1, row 1 is actually one less. "
                                    "NEVER output duplicate consecutive numbers - if two rows in a row have the same number, the FIRST one is one less. "
                                    "Every filled row on the sheet should have a number. Skip blank rows (rows with only 'Lot' printed but no number written). "
                                    "Output one number per line, nothing else. No labels, no commentary, no headers. "
                                    "Include letter prefixes/suffixes if written (G6182, F1234). "
                                    "If a row's number is truly unreadable, output ILLEGIBLE on its own line."
                                ),
                            },
                        ],
                    }
                ],
            ),
            op_name="read_left_column",
        )
        raw = (resp.choices[0].message.content or "").strip()
        nums = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            if line == "ILLEGIBLE":
                nums.append("")
                continue
            m = re.match(r"^([A-Z]*\d{2,}[A-Z]*)\b", line.upper())
            if m:
                nums.append(m.group(1))
        return nums
    except Exception as e:
        print(f"[read_left_column] failed: {e}")
        return []


def _reconcile_item_numbers(transcript: str, verified_nums: List[str]) -> str:
    """v25.68: if the second-pass left-column reader gave us a list of item
    numbers that clearly disagree with the main transcript, replace the
    transcript's item numbers with the verified ones.

    Only rewrites when we have HIGH confidence:
    - Same number of rows on both sides (with tolerance of +/-1)
    - Verified numbers are consistently sequential (each = previous + 1)
    - Transcript numbers are also sequential but with a fixed offset (like -10)

    In all other cases we leave the transcript alone - better to let the user
    click "Fix lot #s on this sheet" than to corrupt a partial match.
    """
    if not verified_nums:
        return transcript
    lines = transcript.splitlines()
    transcript_items = []
    for i, line in enumerate(lines):
        if line.startswith("---") and line.endswith("---"):
            continue
        m = ITEM_NUMBER_RE.match(line.strip())
        if m:
            transcript_items.append((i, m.group(1)))
    if not transcript_items or len(verified_nums) < 2:
        return transcript

    def to_int(s):
        digits = re.sub(r"[^0-9]", "", s)
        return int(digits) if digits.isdigit() and len(digits) >= 3 else None

    tr_ints = [to_int(n) for _, n in transcript_items]
    vf_ints = [to_int(n) for n in verified_nums if n]
    if any(x is None for x in tr_ints) or any(x is None for x in vf_ints):
        return transcript

    def is_sequential(ints):
        return all(ints[i + 1] == ints[i] + 1 for i in range(len(ints) - 1))

    if not is_sequential(tr_ints) or not is_sequential(vf_ints):
        return transcript
    if abs(len(tr_ints) - len(vf_ints)) > 1:
        return transcript
    offset = vf_ints[0] - tr_ints[0]
    if offset == 0:
        return transcript
    if abs(offset) > 100:
        return transcript
    print(f"[reconcile] shifting transcript item numbers by {offset:+d} "
          f"(transcript first={tr_ints[0]}, verified first={vf_ints[0]})")
    new_lines = list(lines)
    for (line_idx, old_num), old_int in zip(transcript_items, tr_ints):
        new_int = old_int + offset
        prefix = re.match(r"^([A-Z]*)", old_num).group(1)
        suffix_m = re.search(r"([A-Z]*)$", old_num)
        suffix = suffix_m.group(1) if suffix_m else ""
        new_num = f"{prefix}{new_int}{suffix}"
        line = new_lines[line_idx]
        new_lines[line_idx] = re.sub(
            r"^([A-Z]*\d{3,}[A-Z]*)",
            new_num,
            line.strip(),
            count=1,
        )
    return "\n".join(new_lines)


async def transcribe_image(image_bytes: bytes, media_type: str) -> str:
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{media_type};base64,{b64}"
    # v25.68: run the main transcription and the left-column-only second pass
    # in parallel, then reconcile. The second pass reads a 2x-zoomed crop of
    # just the item-number column, which the model reads much more reliably
    # (fixes 3-with-closed-top-looks-like-2 handwriting misreads that were
    # cascading through the whole sheet via the sequential-by-1 rule).
    main_task = _openai_with_retry(
        lambda: client.chat.completions.create(
            model="gpt-4o",
            max_tokens=4000,
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
                            "text": "Transcribe all text on this page. Output only the transcription. Pay special attention to the leftmost column of numbers - those are the item / lot numbers. On handwritten sheets: a 6 can have a loopy closed top that looks like a 2; a 3 can have a flat closed top that ALSO looks like a 2; a 0 can look like a 1 when written tightly; a 4 can look like a 9 when the top is closed. If the number is part of a sequential run (like 3062, 3063, 3064 or 1000, 1001, 1002), keep it in sequence and do NOT reset the tens/hundreds/thousands digit mid-run. When in doubt on the FIRST row of a sheet, LOOK AT ROW 2 AND WORK BACKWARDS: if row 2 is clearly 1001 then row 1 must be 1000, not 1001; if row 2 is 8532 then row 1 must be 8531, not 8521. NEVER duplicate an item number - if the second row's number is what you think the first row is, you have misread the first row and it should be one less.",
                        },
                    ],
                },
            ],
        ),
        op_name="transcribe_image",
    )
    left_task = _read_left_column_numbers(image_bytes)
    main_resp, verified_nums = await asyncio.gather(main_task, left_task)
    raw = (main_resp.choices[0].message.content or "").strip()
    sanitized = sanitize_transcript(raw)
    reconciled = _reconcile_item_numbers(sanitized, verified_nums)
    return reconciled




async def is_intake_sheet(image_bytes: bytes, media_type: str) -> bool:
    """Ask the AI whether an image is a JnJ intake sheet (grid of item#/description
    rows) vs. a photo of a physical item. Returns True only for a clear yes.
    Used when no PDF was uploaded and we need to pick which image is the sheet.
    """
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{media_type};base64,{b64}"
    try:
        resp = await _openai_with_retry(
            lambda: client.chat.completions.create(
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
            ),
            op_name="is_intake_sheet",
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


def compute_divider_score(image_bytes: bytes) -> float:
    """v25.31: return a divider-ness score (0.0-1000.0), where higher = more
    divider-like. Replaces the old boolean is_divider_photo.

    Why a score instead of a boolean:
    Ashley's test found that FILE 13 038 (a real photo of a dark object,
    slightly blurry) has almost the same pixel signature as a true JnJ
    divider slide — dark, low variance, few edges. Any yes/no threshold
    that catches true dividers will also catch FILE 13 038. Any threshold
    that excludes FILE 13 038 will miss some true dividers.

    BUT: we know exactly how many dividers should exist (item_count - 1).
    So instead of guessing per-photo, the CLIENT takes the top-N scoring
    photos as dividers where N = item_count - 1. This is self-correcting:
    - if 55 photos look somewhat divider-like, the top 50 (true dividers)
      score higher than the 5 borderline items and win
    - if only 45 obvious dividers exist, the next 5 most-divider-like
      get pulled in (rare, but the ordering still holds)

    Score components (all measured on the top 80% content area, cropping
    out the JnJ watermark strip at the bottom):
      - darkness_score: 500 max, peaks at mean=0, falls to 0 at mean=100
      - flatness_score: 300 max, peaks at stddev=0, falls to 0 at stddev=40
      - blankness_score: 200 max, peaks at edge_mean=0, falls to 0 at edge_mean=10

    A pure black divider scores ~1000. A real item photo scores ~50-200.
    A borderline dark photo like FILE 13 038 might score ~400-600 — still
    lower than true dividers which will hit 850-1000.
    """
    try:
        from PIL import Image, ImageStat, ImageFilter
        with Image.open(io.BytesIO(image_bytes)) as img:
            gray = img.convert("L")
            gray.thumbnail((256, 256))
            w, h = gray.size
            gray_full = gray.copy()
            gray.close()

            # v25.37: sliding-window darkest-patch scoring. v25.36 used
            # 4 quadrants, but Ashley reported another portrait divider
            # (item 3022) still getting missed — the watermark on that
            # divider was big enough to touch all four quadrants.
            #
            # New approach: scan the photo with a small 40x40 sliding
            # window (about 1/6 of the frame) and find the darkest patch.
            # A real divider has HUGE swaths of pure black outside its
            # watermark, so at least one 40x40 patch will score near-
            # perfect black. A real item photo, even a dark one, has
            # texture/edges across the whole frame — no 40x40 patch will
            # be as clean.
            #
            # Cost: ~30ms per photo instead of ~10ms. Still fine.
            PATCH = 40
            STRIDE = 20
            best_mean = 999.0
            best_stddev = 999.0
            best_edge = 999.0
            for y in range(0, max(1, h - PATCH), STRIDE):
                for x in range(0, max(1, w - PATCH), STRIDE):
                    patch = gray_full.crop((x, y, x + PATCH, y + PATCH))
                    pst = ImageStat.Stat(patch)
                    pm = pst.mean[0]
                    # Early-exit: if mean is high, skip the edge check.
                    if pm >= best_mean:
                        patch.close()
                        continue
                    psd = pst.stddev[0]
                    peg = patch.filter(ImageFilter.FIND_EDGES)
                    pem = ImageStat.Stat(peg).mean[0]
                    peg.close()
                    patch.close()
                    best_mean, best_stddev, best_edge = pm, psd, pem
            mean, stddev, edge_mean = best_mean, best_stddev, best_edge
            gray_full.close()

        # v25.34: SHARPER scoring — award divider points only to photos that
        # are UNAMBIGUOUSLY divider-like. v25.31 was too generous, giving
        # moderately dark item photos scores competitive with true dividers.
        # A real JnJ divider is essentially perfect black (mean<15) with
        # near-zero variance (stddev<5) and near-zero edges (<1). We taper
        # sharply so borderline dark item photos score much lower than true
        # dividers even when they look kinda black.

        # Darkness (0-500). Peaks at mean=0, cuts off at mean=30 (not 100).
        # True dividers = mean 5-15 → score 250-420.
        # Dark item photos = mean 30-60 → score 0-0. HUGE gap.
        if mean < 30:
            darkness = 500.0 * (1.0 - mean / 30.0)
        elif mean > 220:
            darkness = 500.0 * ((mean - 220.0) / 35.0)
        else:
            darkness = 0.0

        # Flatness (0-300). Peaks at stddev=0, cuts off at stddev=12.
        # True dividers = stddev 1-5 → score 175-275.
        # Dark item photos = stddev 15-40 → score 0. Gap.
        flatness = max(0.0, 300.0 * (1.0 - stddev / 12.0))

        # Blankness (0-200). Peaks at edge_mean=0, cuts off at edge_mean=3.
        # True dividers = edge_mean 0.2-1.5 → score 100-185.
        # Dark item photos = edge_mean 5-20 → score 0. Gap.
        blankness = max(0.0, 200.0 * (1.0 - edge_mean / 3.0))

        score = darkness + flatness + blankness

        try:
            print(f"divider-score-v37: darkest_patch mean={mean:.1f} stddev={stddev:.1f} edge_mean={edge_mean:.2f} → score={score:.0f}", flush=True)
        except Exception:
            pass

        return score
    except Exception:
        return 0.0


def is_divider_photo(image_bytes: bytes) -> bool:
    """v25.31: kept as a compatibility shim. Uses compute_divider_score with
    a very conservative threshold (700+) so the OLD callers only get
    obvious dividers. Real detection is now score-based via top-N picking
    in the client's cursor walk — see app.js.
    """
    return compute_divider_score(image_bytes) >= 700.0


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
    # v25.51: J&J's importer expects EXACTLY 50 columns (A-AX in their
    # sample sheet). v25.32 added cf_LotNumber + cf_Location as extra
    # columns, pushing total to 52 — that overflowed their VBScript
    # column array and crashed with 'Subscript out of range: iCF_ColumnCount'
    # at process_admin_importitems.asp line 798.
    #
    # All three IDs are still preserved in the Description field, so
    # nothing is lost — they just live in one place instead of three.
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
    # v25.17: IsTaxable=1 so the item is marked taxable; TaxPercent=6 matches
    # the 6% Michigan rate shown on live J&J listings. J&J's spec notes the
    # seller's state-specific tax settings can override this per buyer state.
    "IsTaxable": "1",
    "TaxPercent": "6",
    # v25.15: J&J's importer rejected v25.14 with "Missing End Date or Duration"
    # on every row (see IMG_2896/IMG_2897). Column V (Duration) is required
    # when EndDate (col U) is blank. 7 days is J&J's typical auction run;
    # frontend can override via the sale dialog.
    "Duration": "7",
    # v25.24: Ashley wants "Homepage Gallery Free of Charge" always checked.
    # J&J's Admin-CSV-Help.html lists GalleryListing (col AB) as "1 or 0".
    # Setting to 1 turns on the gallery listing for every item.
    "GalleryListing": "1",
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
            # v25.67: tightened lot-code detection. Previously any 1-5 char
            # alphanumeric token was treated as a lot code, which grabbed
            # description words like COBRA, BRASS, HOODED, BOOK when the lot
            # column on the sheet was blank. Real JnJ lot codes are one of:
            #   - contains at least one digit (18A, 37G, 2046, Z3, 204B)
            #   - 1-2 letters only (F, Z, P, C, Cr)
            # An all-letter token of 3+ chars is a description word, not a code.
            is_short_letters_only = bool(re.fullmatch(r'[A-Z]{1,2}', tok))
            has_digit = any(c.isdigit() for c in tok)
            looks_like_lot_code = (
                1 <= len(tok) <= 5
                and bool(re.fullmatch(r'[A-Z0-9]+', tok))
                and (has_digit or is_short_letters_only)
            )
            if looks_like_lot_code:
                lot_code = tok
                rest = rest[1:]
        description = " ".join(rest).strip()
        if not description:
            # v25.33: empty description often means the OCR saw only ditto
            # marks and the marks got stripped. Use a literal ditto so the
            # ditto expander picks it up instead of falling back to
            # ILLEGIBLE and losing the row's inheritance.
            description = '"'
        items.append((item_num, lot_code, description))

    # v25.33: expand ditto marks. Sellers commonly write `"    "    "` (or
    # similar) in the description column to mean "same item as the row
    # above." Ashley: "they can't change this so how do we work with it."
    #
    # Detect ditto-style descriptions and copy the previous row's real
    # description forward. The lot number and location stay unique to this
    # row — only the description is duplicated. This matches how J&J's
    # buyers see the listing: same item, different lot #.
    #
    # A description counts as ditto if, after removing quotes/apostrophes/
    # backticks/whitespace/hyphens/asterisks, either nothing is left OR
    # only the words DITTO / SAME / ABOVE / SAMEASABOVE remain.
    _DITTO_STRIP = re.compile(r'["\'`‘’“”\s\-*–—.,]+')
    def _is_ditto(desc: str) -> bool:
        if not desc:
            return False
        stripped = _DITTO_STRIP.sub("", desc).upper()
        if not stripped:
            return True  # pure ditto marks / whitespace
        return stripped in {"DITTO", "SAME", "SAMEASABOVE", "AS", "ABOVE", "SAMEAS"}

    expanded = []
    last_real_desc = ""
    for (num, loc, desc) in items:
        if _is_ditto(desc) and last_real_desc:
            try:
                print(f"ditto-expand: {num} {loc} inherits from previous: {last_real_desc!r}", flush=True)
            except Exception:
                pass
            expanded.append((num, loc, last_real_desc))
        else:
            expanded.append((num, loc, desc))
            # Only update the anchor when the current row has a REAL desc.
            if desc and desc != "ILLEGIBLE" and not _is_ditto(desc):
                last_real_desc = desc
    return expanded


def build_jnj_csv_row(item_num: str, lot_code: str, description: str,
                     sale_name: str, seller_id: str, seller_seq: int,
                     per_item_seller: str = "") -> dict:
    """
    Build one row of the JnJ CSV.

    v25.32 — THREE SEPARATE ID FIELDS per Ashley's rule ("do not put them
    together"):
      - cf_SellerID  = seller ID from boxed number at top of sheet (e.g. AA3102)
      - cf_LotNumber = left column on the sheet (e.g. 3022) — what J&J calls
                       the LOT NUMBER. Parsed into `item_num` by our OCR.
      - cf_Location  = right column on the sheet (e.g. 38B) — storage bin,
                       parsed into `lot_code` by our OCR (misnamed for
                       historical reasons; the variable is the location).

    v25.54 — Title format includes the sale letter between Lot and Location:
        '{item_num}{SALE_LETTER}{lot_code} {description}'
        e.g. sale 'SEPTEMBER 3 ~ J SALE' → '1500J72B USED WORKING GRAY 2 PIECE'
        e.g. sale 'AUGUST 27 H SALE'    → '9749H15C CLASSIC FOOD MASTER SHREDDER'
    The letter is extracted from the sale_name — the single A-Z that
    immediately precedes the word 'SALE'. Falls back to '' if we can't
    find one, so the title still renders (just without the separator).
    If lot_code is missing, we drop it entirely and leave item_num + description.

    The "do not put them together" rule from v25.32 still applies to the
    Description field (stamped with `Seller: ... | Lot: ... | Location: ...`
    on its own line) and to the cf_SellerID custom field. Title is a
    separate case where staff need the IDs glued for quick visual scan
    on J&J's listings page.

    Title cap: 60 chars per JnJ spec (Admin CSV Help column D).
    """
    # v25.54: extract the sale letter (e.g. 'J' from 'SEPTEMBER 3 ~ J SALE').
    # Match a single A-Z with word boundaries just before the literal 'SALE'.
    sale_letter = ""
    if sale_name:
        m = re.search(r"\b([A-Z])\s*SALE\b", sale_name.upper())
        if m:
            sale_letter = m.group(1)

    # v25.53/54: glued title with sale letter as separator.
    if item_num and lot_code:
        title = f"{item_num}{sale_letter}{lot_code} {description}"
    elif item_num:
        title = f"{item_num} {description}"
    else:
        title = description
    if len(title) > 60:
        title = title[:60].rstrip()

    row = {col: "" for col in JNJ_CSV_COLUMNS}
    row.update(JNJ_DEFAULTS)
    row["Category"] = sale_name
    row["Title"] = title

    # cf_SellerID (v25.23 + v25.32): ALWAYS "AA" + number, no exceptions.
    # Number comes from (in order): boxed per-item seller num, sheet-level
    # seller_id if it has digits, then the fallback counter. Letters stripped.
    per_item_seller = (per_item_seller or "").strip()
    per_item_digits = "".join(c for c in per_item_seller if c.isdigit())
    seller_id_digits = "".join(c for c in (seller_id or "") if c.isdigit())
    if per_item_digits:
        seller_number = per_item_digits
    elif seller_id_digits:
        seller_number = seller_id_digits
    else:
        seller_number = str(seller_seq)
    row["cf_SellerID"] = f"AA{seller_number}"

    # v25.51: cf_LotNumber / cf_Location were dropped — J&J's importer
    # expects EXACTLY 50 columns and crashed with 'Subscript out of range:
    # iCF_ColumnCount' when we included them. The lot number and location
    # are still preserved in the Description field just below (all three
    # IDs live there), which is what actually shows on the live listing
    # anyway.

    # v25.56: Description is now JUST the item description — no more
    # 'Seller: ... | Lot: ... | Location: ...' stamp. Ashley confirmed
    # (Aug 26 IMG_2943) that the stamp was cluttering the live listing:
    # 'Seller: AA5350 | Lot: 9736 | Location: 15BBLACK RACK DEER RATTLERS'.
    # The seller ID already shows in J&J's own 'ID Code' field via
    # cf_SellerID, and Lot + Location are already glued into the Title
    # (e.g. '9736J15B BLACK RACK DEER RATTLERS'), so nothing is lost.
    row["Description"] = description or ""

    row["StartBid"] = "$1.00 "
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
        resp = await _openai_with_retry(
            lambda: client.chat.completions.create(
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
            ),
            op_name="read_photo_tag",
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
        resp = await _openai_with_retry(
            lambda: client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=30,
            messages=[{"role": "user", "content": prompt}],
            ),
            op_name="match_photo_by_description",
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


async def extract_seller_groups(image_bytes: bytes, media_type: str) -> List[Dict[str, str]]:
    """v25.4: Find EVERY hand-drawn box on the sheet, not just the top one.

    On JnJ sheets a boxed number covers items from where it appears down
    until the next boxed number. So a single sheet may have multiple boxed
    sellers, each grouping a range of item rows.

    Returns a list ordered top-to-bottom:
      [
        {"seller_num": "1894", "first_item_num": "2000"},   # first group
        {"seller_num": "06",   "first_item_num": "2004"},   # next group down
      ]

    Where 'first_item_num' is the item number from the OFFICE USE ONLY
    column of the FIRST item row that falls under that box. The client uses
    this to stamp each item with the correct seller #.

    Empty list if no boxed numbers found.
    """
    if not _OPENAI_KEY:
        return []
    try:
        b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
        resp = await _openai_with_retry(
            lambda: client.chat.completions.create(
            model="gpt-4o",
            max_tokens=200,
            temperature=0,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "This is a J&J Estate Auctioneers intake sheet with a grid of "
                        "item rows. The office worker draws HAND-DRAWN BOXES around "
                        "seller numbers on this sheet in black pen/marker.\n\n"
                        "IMPORTANT: There may be MULTIPLE hand-drawn boxes on one sheet. "
                        "Each box applies to items from that box's row DOWN until the "
                        "next hand-drawn box. Sometimes there is only one box at the very "
                        "top; sometimes there are additional boxes further down the sheet.\n\n"
                        "Find EVERY hand-drawn box (top to bottom) and read:\n"
                        "1. The digits inside the box (1-5 digits).\n"
                        "2. The item number in the OFFICE USE ONLY column of the FIRST "
                        "row of items that appears at or below that box.\n\n"
                        "Reply as JSON only, no prose. Format template:\n"
                        "{\"groups\": [ {\"seller_num\": \"<digits you read>\", \"first_item_num\": \"<item number you read>\"} ]}\n\n"
                        "CRITICAL: Read the ACTUAL numbers from the image. Do NOT invent numbers.\n"
                        "CRITICAL: Do NOT copy any placeholder from these instructions.\n"
                        "CRITICAL: Every number you output must be clearly visible in a hand-drawn box on the image.\n\n"
                        "Rules:\n"
                        "- Ignore PRINTED boxes (OFFICE USE ONLY header, LOT DESCRIPTION header).\n"
                        "- Ignore the lister/cart sub-boxes at the bottom.\n"
                        "- Preserve leading zeros (if box shows 06, output '06' not '6').\n"
                        "- List groups top-to-bottom in the order they appear on the sheet.\n"
                        "- If there are no hand-drawn boxes at all, reply: {\"groups\": []}"
                    )},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{media_type};base64,{b64}",
                        "detail": "high",
                    }},
                ],
            }],
            response_format={"type": "json_object"},
            ),
            op_name="extract_seller_groups",
        )
        raw = (resp.choices[0].message.content or "").strip()
        print(f"extract_seller_groups raw: {raw!r}", flush=True)
        try:
            parsed = json.loads(raw)
        except Exception:
            return []
        groups = parsed.get("groups", []) if isinstance(parsed, dict) else []
        cleaned = []
        for g in groups:
            if not isinstance(g, dict):
                continue
            sn = re.sub(r"[^0-9]", "", str(g.get("seller_num", "")))
            fi = re.sub(r"[^A-Za-z0-9]", "", str(g.get("first_item_num", "")))
            if 1 <= len(sn) <= 5:
                cleaned.append({"seller_num": sn, "first_item_num": fi})
        return cleaned
    except Exception as e:
        print(f"extract_seller_groups failed: {type(e).__name__}: {e}", flush=True)
        return []


async def extract_seller_number(image_bytes: bytes, media_type: str) -> str:
    """Find the hand-drawn BOXED seller number in the top header area of a
    JnJ intake sheet. The seller draws a rectangle/square around 2-4 digits
    (like 2860, 6009, 559) in the top ~20% of the page. That number is the
    seller's staff ID and must appear on every item in the CSV so it shows
    up on the JnJ website.

    Returns the digits only (e.g. '2860'), or '' if none found. gpt-4o-mini
    is plenty for this — ~250ms, ~$0.001.
    """
    if not _OPENAI_KEY:
        return ""
    try:
        b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
        # v25.1: upgraded to gpt-4o (from mini) for the boxed-number extraction.
        # gpt-4o-mini was missing hand-drawn boxes in Dave's real sheets — the
        # cost delta is negligible (one call per sheet) and full 4o reads
        # handwriting inside marker boxes far more reliably.
        # Prompt is also stricter and shows the model concrete examples of
        # what the boxes look like (e.g. "06", "1894", "2860").
        resp = await _openai_with_retry(
            lambda: client.chat.completions.create(
            model="gpt-4o",
            max_tokens=15,
            temperature=0,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "This is a J&J Estate Auctioneers intake sheet. "
                        "At the TOP of the sheet, above or beside the \"SELLERS NAME\" line, "
                        "the office worker draws a rectangle or square in black pen/marker "
                        "around the seller's number. The number is 1-5 digits.\n\n"
                        "The box is ALWAYS hand-drawn (not a printed rectangle) and is in "
                        "the top portion of the sheet, near the sellers name / cart # line.\n\n"
                        "Read the digits inside that hand-drawn box.\n\n"
                        "CRITICAL: Read the ACTUAL number written on the sheet. Do NOT invent digits.\n"
                        "CRITICAL: Do NOT copy any number from these instructions.\n"
                        "CRITICAL: Every digit you output must be clearly visible inside a hand-drawn box.\n\n"
                        "Rules:\n"
                        "- Ignore any printed boxes such as OFFICE USE ONLY, LOT DESCRIPTION, "
                        "or the LISTER box at the bottom.\n"
                        "- Ignore lot/item numbers in the grid rows.\n"
                        "- Ignore CART # if it's just letters like \"Test\".\n"
                        "- Preserve leading zeros exactly as written (if box shows 06, output 06 not 6).\n\n"
                        "Reply with ONLY the digits, nothing else. If you truly cannot see "
                        "a hand-drawn box with a number, reply with exactly: NONE"
                    )},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{media_type};base64,{b64}",
                        "detail": "high",
                    }},
                ],
            }],
            ),
            op_name="extract_seller_number",
        )
        raw = (resp.choices[0].message.content or "").strip()
        print(f"extract_seller_number raw response: {raw!r}", flush=True)
        # Sanity: reject if AI said NONE.
        if "NONE" in raw.upper():
            return ""
        # Keep only digits, but preserve them in original order (leading zeros OK).
        digits = re.sub(r"[^0-9]", "", raw)
        if 1 <= len(digits) <= 5:
            return digits
        return ""
    except Exception as e:
        print(f"extract_seller_number failed: {type(e).__name__}: {e}", flush=True)
        return ""


async def extract_seller_number_from_sheet(sheet: UploadFile) -> str:
    """Run extract_seller_number against an uploaded sheet (image or first PDF page).
    Sheet's read cursor is consumed — do not call again on the same UploadFile.
    """
    data = await sheet.read()
    if not data:
        return ""
    # Reset for any later reads.
    try:
        await sheet.seek(0)
    except Exception:
        pass
    fname = (sheet.filename or "").lower()
    ctype = (sheet.content_type or "").lower()
    is_pdf = fname.endswith(".pdf") or ctype == "application/pdf"
    if is_pdf:
        pages = render_pdf_pages(data, dpi=180)
        if not pages:
            return ""
        return await extract_seller_number(pages[0], "image/png")
    else:
        media_type = ctype if ctype.startswith("image/") else "image/jpeg"
        return await extract_seller_number(data, media_type)


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

    v20: ALSO extracts the hand-drawn boxed seller number from the top of
    the sheet and returns it as `seller_number` — the client uses this to
    fill in the Seller ID field automatically.

    v25.13: When a multi-page PDF is uploaded, treat each non-blank page as
    its own sheet with its own boxed seller # extraction. Response now always
    includes a `pages` array (one entry per page) so the frontend can flatten
    a single multi-page PDF into N virtual sheets. For a single-image upload
    or a single-page PDF, `pages` has exactly one entry.
    """
    try:
        # Read the sheet ONCE, then do transcription + seller-number extraction
        # against the same bytes. transcribe_uploaded_sheet and
        # extract_seller_number_from_sheet both call .read(), which would
        # return empty on the second call. So we buffer the bytes ourselves.
        raw = await sheet.read()
        fname = (sheet.filename or "").lower()
        ctype = (sheet.content_type or "").lower()
        is_pdf = fname.endswith(".pdf") or ctype == "application/pdf"

        # --- v25.13: build a list of (page_bytes, media_type) tuples ---
        # For an image upload, the list has one entry (the image itself).
        # For a PDF, the list has one entry per non-blank rendered page.
        page_units: List[tuple] = []
        if is_pdf:
            rendered = render_pdf_pages(raw, dpi=180)
            non_blank = [pb for pb in rendered if not is_blank_image(pb)]
            for pb in non_blank:
                page_units.append((pb, "image/png"))
        else:
            media_type = ctype if ctype.startswith("image/") else "image/jpeg"
            page_units.append((raw, media_type))

        if not page_units:
            raise HTTPException(400, "Sheet appears blank — no readable pages found.")

        # --- Run transcription + seller-group extraction on EACH page in parallel ---
        async def _do_page(pb: bytes, mt: str) -> Dict[str, Any]:
            transcript_task = transcribe_image(pb, mt)
            groups_task = extract_seller_groups(pb, mt)
            transcript, seller_groups = await asyncio.gather(transcript_task, groups_task)
            items = parse_items_from_transcript(transcript)
            first_seller = seller_groups[0]["seller_num"] if seller_groups else ""
            return {
                "transcript": transcript,
                "items": items,
                "seller_number": first_seller,
                "seller_groups": seller_groups,
            }

        page_results = await asyncio.gather(*[_do_page(pb, mt) for pb, mt in page_units])

        # Drop pages that transcribed to nothing (e.g. blank scan the blank-image
        # detector missed). If ALL pages came back empty, raise so the client
        # sees a clear error message with the first page's transcript preview.
        good_pages = [pg for pg in page_results if pg["items"]]
        if not good_pages:
            preview = (page_results[0]["transcript"] or "")[:400]
            raise HTTPException(400, f"Sheet transcribed but no item rows were parsed. Transcript: {preview}")

        # --- Legacy top-level fields for old clients that don't read `pages` ---
        # Concatenate all items across pages so pre-v25.13 clients still get a
        # usable response. New clients (v25.13+) should read `pages` and treat
        # each entry as its own sheet with its own seller_groups.
        all_items: List[Dict[str, str]] = []
        for pg in good_pages:
            all_items.extend(pg["items"])
        first_seller = good_pages[0]["seller_number"]
        first_groups = good_pages[0]["seller_groups"]
        combined_transcript = "\n".join(pg["transcript"] for pg in good_pages)

        return JSONResponse({
            # Legacy fields (kept for back-compat)
            "transcript": combined_transcript,
            "items": all_items,
            "seller_number": first_seller,
            "seller_groups": first_groups,
            # v25.13: authoritative per-page breakdown
            "pages": good_pages,
            "page_count": len(good_pages),
        })
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

            # v18: Ask AI whether the photo contains an auction item.
            # Dave's workflow: [item A photos] → [no-item photo] → [item B photos] → ...
            # A "no-item" photo can be anything Dave shoots between items:
            # black, white, hand, floor, wall, ceiling, sky, grass, blur, etc.
            # The only reliable way to distinguish these from real items is
            # to actually LOOK at the photo. gpt-4o-mini vision does this for
            # ~$0.001 per photo and ~250ms latency.
            thumb_data_url = ""
            is_blank = False
            ai_thumb_b64 = ""  # small thumb we send to OpenAI
            try:
                with Image.open(io.BytesIO(raw)) as img:
                    rgb = img.convert("RGB")

                    # Small thumb for the AI check (256px is plenty — the model
                    # only needs to see "is there a subject or is this a hand/floor/etc").
                    ai_buf = io.BytesIO()
                    ai_copy = rgb.copy()
                    ai_copy.thumbnail((256, 256))
                    ai_copy.save(ai_buf, format="JPEG", quality=70)
                    ai_thumb_b64 = base64.standard_b64encode(ai_buf.getvalue()).decode("utf-8")
                    ai_copy.close()
                    del ai_buf, ai_copy

                    # 160px thumb for the UI display
                    tbuf = io.BytesIO()
                    rgb.thumbnail((160, 160))
                    rgb.save(tbuf, format="JPEG", quality=70)
                    thumb_b64 = base64.standard_b64encode(tbuf.getvalue()).decode("utf-8")
                    thumb_data_url = f"data:image/jpeg;base64,{thumb_b64}"
                    rgb.close()
                    del rgb, tbuf, thumb_b64
            except Exception as e:
                print(f"process_photo shrink failed for {filename}: {e}", flush=True)

            # Ask the AI: is there an auction item in this photo?
            # THREE possible answers now:
            #   YES   = definitely an auction item (furniture, tool, etc.)
            #   NO    = definitely no item (hand, floor, wall, black, sky, etc.)
            #   MAYBE = ambiguous close-up of texture/metal/wood/fabric
            # v19: 'maybe' photos wait for pass 2 (neighbor check).
            first_pass = "yes"
            # v25.31: score-based divider detection. Compute a 0-1000 score
            # for every photo (higher = more divider-like) and return it to
            # the client. Client picks the top (item_count - 1) as dividers.
            # This is self-correcting against false positives like FILE 13 038
            # (a real dark item photo whose signature overlaps with true
            # dividers). We no longer set is_blank server-side based on
            # score — the client does that after seeing all scores together.
            # v25.45: QR-code divider card is the AUTHORITATIVE signal.
            # If Dave shot the printed DIVIDER card, we know 100% that this
            # is a divider. Skip pixel scoring AND the AI call entirely,
            # slam divider_score to 1000 so the client picks it every time.
            has_divider_qr = detect_divider_qr(raw)
            divider_score = compute_divider_score(raw)
            if has_divider_qr:
                divider_score = 1000.0
                first_pass = "no"
                is_blank = True
                print(f"divider-QR: {filename} DIVIDER CARD DETECTED — forcing divider", flush=True)
            else:
                print(f"divider-check: {filename} score={divider_score:.0f} qr=none", flush=True)
            # Only trigger the AI "is this an item" fallback on photos that
            # scored VERY high (unambiguously divider-like). This keeps API
            # cost the same as before.
            if has_divider_qr:
                # Already decided by QR — skip everything below.
                pass
            elif divider_score >= 850:
                # Skip AI call — unambiguously a divider. Score alone decides.
                pass
            elif ai_thumb_b64 and _OPENAI_KEY:
                try:
                    resp = await _openai_with_retry(
                        lambda: client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": (
                                    "This is a photo from an estate auction. EVERY photo has a 'JNJ ONLINE AUCTION - FREMONT' watermark somewhere in the frame \u2014 IGNORE the watermark text completely.\n\n"
                                    "A DIVIDER SLIDE is a photo where the ENTIRE frame (aside from the watermark) is one uniform solid color \u2014 typically ALL BLACK, occasionally all white or all gray. There is NO subject, NO texture, NO object, NO scene. Just uniform color like a photo of a piece of black cardboard or an unlit surface.\n\n"
                                    "A REAL ITEM PHOTO has ANY visible subject or scene: furniture, tools, decor, boxes, hands, floor, wall, ceiling, blurry motion, dark object with reflections, dark corner of a room, ANYTHING that isn't a uniform color field.\n\n"
                                    "CRITICAL: hands, floors, blurry shots, dark objects on dark backgrounds \u2014 ALL of these are ITEM photos, NOT dividers. A divider looks like a black rectangle with just the JnJ watermark on it \u2014 nothing else in the frame at all.\n\n"
                                    "Reply with EXACTLY one word:\n\n"
                                    "DIVIDER - the frame is one uniform solid color, no subject at all beyond the watermark. Like a blank black slide.\n"
                                    "ITEM    - anything else, including hands, floors, blurry photos, dark scenes with any visible objects or texture."
                                )},
                                {"type": "image_url", "image_url": {
                                    "url": f"data:image/jpeg;base64,{ai_thumb_b64}",
                                    "detail": "low",
                                }},
                            ],
                        }],
                        max_tokens=5,
                        temperature=0,
                        ),
                        op_name="jnj_match_photos",
                    )
                    answer = (resp.choices[0].message.content or "").strip().lower()
                    # v25.42: prompt now asks the AI to say DIVIDER vs ITEM.
                    # We treat 'divider' as blank; anything else (including
                    # 'item', 'no', empty, error) is a real item. This is
                    # narrower than the old prompt which asked YES/NO and
                    # got too many 'no' answers for hands/floors/blurry.
                    if answer.startswith("d"):
                        first_pass = "no"
                except Exception as e:
                    print(f"has-item check failed for {filename}: {type(e).__name__}: {e}", flush=True)

            if first_pass == "no":
                is_blank = True
            # v25.31: is_blank is now only a HINT to the client. The final
            # divider set is chosen by client-side top-N picking based on
            # divider_score. is_blank stays populated for the debug log.
            print(f"photo-classify: {filename} first_pass={first_pass} is_blank={is_blank} divider_score={divider_score:.0f}", flush=True)
            # NOTE: we intentionally KEEP ai_thumb_b64 around — it goes into the
            # returned dict so pass 2 can use it for neighbor comparison.

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
                "first_pass": first_pass,  # 'yes' / 'no' / 'maybe'
                # v25.43: the client's cursor-walk expects match_kind to be
                # 'none' for unassigned photos so it can fill them in with
                # the current cursor's item number. Without this the check
                # `if (p.match_kind === 'none')` was false (undefined != 'none')
                # and photos never got assigned to items in order — which is
                # exactly what caused FILE 13 156 and its neighbors to end up
                # in item 3022 instead of the correct items.
                "item_num_match": "",
                "match_kind": "none",
                # v25.31: raw divider-ness score (0-1000). Client sorts all
                # photos by this and picks the top (item_count - 1) as dividers.
                "divider_score": divider_score,
                # v25.45: authoritative QR signal. When true, the client
                # adds this photo to dividerSet unconditionally — no score
                # comparison, no per-sheet limit. This lets Dave shoot MORE
                # divider cards than the sheet has items and still be right.
                "has_divider_qr": has_divider_qr,
                # ai_thumb_b64 is only sent back for 'maybe' photos to keep
                # response size down. Client uses it to do a neighbor-check
                # call to /api/jnj-resolve-maybe.
                "ai_thumb_b64": ai_thumb_b64 if first_pass == "maybe" else "",
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


@app.post("/api/jnj-resolve-maybe")
async def jnj_resolve_maybe(
    subject_b64: str = Form(...),
    neighbor_b64s_json: str = Form(...),  # JSON list of base64 thumbs
):
    """v19 pass-2: given an ambiguous 'maybe' photo plus 1-3 confirmed 'yes'
    neighbor thumbs, ask the AI whether the maybe photo is (a) a close-up
    detail of the same item as the neighbors — in which case keep it, or
    (b) a divider photo of no item — in which case skip it.

    Returns {"is_item": bool}."""
    try:
        neighbors = json.loads(neighbor_b64s_json)
        if not isinstance(neighbors, list):
            neighbors = []
    except Exception:
        neighbors = []

    if not subject_b64:
        return JSONResponse({"is_item": True})  # safe default — keep the photo

    # Cap to 3 neighbors to keep the call cheap.
    neighbors = [n for n in neighbors if n][:3]

    if not _OPENAI_KEY:
        return JSONResponse({"is_item": True})

    content: List[Dict] = [
        {"type": "text", "text": (
            "You are helping sort auction-sale photos. The FIRST image is the "
            "photo being classified. The remaining images are photos taken "
            "right before and/or after it in the same shoot — all confirmed "
            "to contain auction items.\n\n"
            "Question: is the FIRST photo a close-up detail of the SAME item "
            "shown in the neighbor photos, or is it a divider/blank photo "
            "(hand, floor, wall, texture with no item present, etc.)?\n\n"
            "Answer ONLY 'item' if it appears to be a close-up of the same "
            "item shown nearby (wagon wheel, tool blade, drawer, fabric of "
            "the same piece, etc.).\n"
            "Answer ONLY 'blank' if it's a divider photo with no item."
        )},
        {"type": "image_url", "image_url": {
            "url": f"data:image/jpeg;base64,{subject_b64}",
            "detail": "low",
        }},
    ]
    for nb in neighbors:
        content.append({"type": "image_url", "image_url": {
            "url": f"data:image/jpeg;base64,{nb}",
            "detail": "low",
        }})

    try:
        # v25.5: back to gpt-4o-mini for speed. v25.3's full gpt-4o here was
        # causing the pipeline to hang because every close-up now went
        # through this endpoint AND took several seconds each.
        resp = await _openai_with_retry(
            lambda: client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": content}],
            max_tokens=5,
            temperature=0,
            ),
            op_name="jnj_resolve_maybe",
        )
        answer = (resp.choices[0].message.content or "").strip().lower()
        is_item = not answer.startswith("b")  # blank -> not item
        return JSONResponse({"is_item": is_item})
    except Exception as e:
        print(f"resolve-maybe failed: {type(e).__name__}: {e}", flush=True)
        # Safe default — if AI fails, keep the photo.
        return JSONResponse({"is_item": True})


@app.post("/api/jnj-verify-assignment")
async def jnj_verify_assignment(
    photo_b64: str = Form(...),
    current_item_desc: str = Form(...),
    next_item_desc: str = Form(""),
):
    """v25.29: after the cursor walk assigns photos to items, verify each
    photo actually matches its item's description. If the photo looks more
    like the NEXT item's description, suggest a move.

    Ashley's use case: sometimes Dave shoots a photo of item B before
    triggering the black-divider, so it lands on item A by mistake. This
    pass catches that.

    Returns one of:
      {"verdict": "current"}  - photo matches its current item (keep)
      {"verdict": "next"}     - photo matches the next item better (move)
      {"verdict": "neither"}  - photo doesn't clearly fit either (keep on current, safe default)
    """
    if not photo_b64 or not current_item_desc:
        return JSONResponse({"verdict": "current"})

    if not _OPENAI_KEY:
        return JSONResponse({"verdict": "current"})

    # If there's no next item to compare against, no point checking — keep it.
    if not next_item_desc.strip():
        return JSONResponse({"verdict": "current"})

    # Truncate descriptions to keep the prompt tight.
    cur = current_item_desc.strip()[:300]
    nxt = next_item_desc.strip()[:300]

    try:
        resp = await _openai_with_retry(
            lambda: client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "An estate-auction photo has been tentatively assigned to Item A. Look at the photo (ignore the 'JNJ ONLINE AUCTION - FREMONT' watermark burned into the bottom) and decide which item it best matches.\n\n"
                        f"ITEM A description: {cur}\n"
                        f"ITEM B description: {nxt}\n\n"
                        "Note: an item may contain multiple objects (e.g. '3 lamps, 2 vases, box of tools') and Dave often shoots several photos per item from different angles. A photo showing ANY object mentioned in item A's description matches Item A.\n\n"
                        "Reply with EXACTLY one word:\n"
                        "  A       - photo clearly shows an object described in Item A (default when unsure)\n"
                        "  B       - photo clearly shows an object described in Item B but NOT in Item A\n"
                        "  NEITHER - photo doesn't match either description (rare)\n\n"
                        "Bias strongly toward A. Only answer B if the photo shows something specifically mentioned in Item B's description that is NOT in Item A's description."
                    )},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{photo_b64}",
                        "detail": "low",
                    }},
                ],
            }],
            max_tokens=5,
            temperature=0,
            ),
            op_name="jnj_verify_assignment",
        )
        answer = (resp.choices[0].message.content or "").strip().upper()
        if answer.startswith("B"):
            return JSONResponse({"verdict": "next"})
        if answer.startswith("N"):
            return JSONResponse({"verdict": "neither"})
        return JSONResponse({"verdict": "current"})
    except Exception as e:
        print(f"verify-assignment failed: {type(e).__name__}: {e}", flush=True)
        # Safe default: keep on current item.
        return JSONResponse({"verdict": "current"})


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

    # v25.18: diagnostic logging — was tripped up before by ZIPs missing all
    # photos even though the preview showed 83/83 matched. Print exactly what
    # the server received so we can tell if photos got stripped in transit or
    # if the mapping step failed. Visible in Render logs.
    print(f"[jnj-zip] items={len(items)} photo_map_entries={len(photo_map)} photos_received={len(photos)} sale='{sale_name}' seller_id='{seller_id}'", flush=True)

    # Group photos by item_num, preserving upload order.
    photos_by_item: Dict[str, List[UploadFile]] = {}
    skipped_no_target = 0
    for p in photos:
        target = photo_map.get(p.filename or "")
        if not target:
            skipped_no_target += 1
            continue
        photos_by_item.setdefault(target, []).append(p)
    print(f"[jnj-zip] photos grouped into {len(photos_by_item)} items, {skipped_no_target} skipped (no target in photo_map)", flush=True)

    # --- v25.16b: restore J&J's proven subfolder+backslash layout -----------
    # 25.16a tried bare filenames flat at ZIP root: photos still didn't attach
    # and items showed "NO PHOTO UPLOADED" on J&J's site (IMG_2899).
    #
    # J&J's own working sample (1-2.csv) uses:
    #   image_1 = ..\Pictures\2026-04-07 TEST\TEST 001.webp
    # and photos live inside Pictures/2026-04-07 TEST/ in the ZIP.
    #
    # This build restores that exact layout. The sale folder name preserves
    # the ENTIRE sale name Ashley typed (including "~", spaces, and any
    # tilde/punctuation) so the ZIP subfolder name matches the sale category
    # J&J's system knows (e.g. "SEPTEMBER 3 ~ J SALE").
    def _folder_slug(s: str) -> str:
        s = (s or "").strip()
        if not s:
            s = datetime.utcnow().strftime("%Y-%m-%d SALE")
        # v25.20: J&J's uploader flags "Missing Image" when the sale name
        # contains a tilde (~). Their working sample uses simple names like
        # "2026-04-07 TEST" — no punctuation. Strip tilde AND collapse to a
        # clean alphanumeric+space+dash slug so the folder name inside the
        # ZIP matches whatever the uploader is looking for.
        s = re.sub(r"[^A-Za-z0-9 \-]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s or "SALE"

    sale_folder = _folder_slug(sale_name)
    # v25.20: use a SHORT, SAFE filename prefix instead of the whole sale name.
    # J&J's working sample used "TEST 001.webp" — 4 chars + number. Long prefixes
    # like "SEPTEMBER 3 ~ J SALE 005.jpg" may hit path-length or character issues
    # in the uploader. Derive a 3-4 letter code from the first meaningful token.
    #
    # v25.63: EVERY upload gets a unique timestamp suffix on the prefix so two
    # sales named similarly ("September 3 Sale" and "September 5 Sale") don't
    # produce colliding filenames like SEPT 001.jpg / SEPT 002.jpg on J&J's side.
    # Cause of "old photos still show": J&J's server saves photos by filename
    # globally; when a new upload has the SAME filenames as a previous one, the
    # new file overwrites the old one, but stale listings that referenced the
    # old file now show the WRONG image. Adding a datestamp guarantees uniqueness.
    def _short_prefix(folder: str) -> str:
        toks = [t for t in folder.split() if t and not t.isdigit() and t != "-"]
        if not toks:
            return "IMG"
        first = toks[0].upper()
        return first[:4] if len(first) >= 3 else first
    # Append MMDD-HHMM stamp so each upload's photos are unique on J&J's server.
    stamp = datetime.utcnow().strftime("%m%d-%H%M")
    photo_prefix = f"{_short_prefix(sale_folder)}{stamp}"
    # ---------------------------------------------------------------------

    # Build the zip in memory
    zip_buf = io.BytesIO()
    csv_buf = io.StringIO()
    writer = csv.DictWriter(csv_buf, fieldnames=JNJ_CSV_COLUMNS, quoting=csv.QUOTE_MINIMAL)
    writer.writeheader()

    # Sale-wide photo counter so filenames are sequential across items,
    # matching J&J's sample (TEST 001, TEST 002, TEST 003 …).
    sale_photo_seq = 0

    # For each item build a row and, if it has photos, name them + fill image_1..N
    photo_files: List[Tuple[str, bytes]] = []  # (name_in_zip, bytes)
    for idx, it in enumerate(items):
        item_num = it.get("item_num", "")
        lot_code = it.get("lot_code", "")
        description = it.get("description", "")
        # Per-item seller # (from the item's own sheet's boxed number) — used
        # when multiple sheets are dropped together, each with its own boxed #.
        per_item_seller = it.get("sheet_seller_num", "") or it.get("seller_num", "")
        row = build_jnj_csv_row(
            item_num, lot_code, description,
            sale_name, seller_id, seller_start + idx,
            per_item_seller=per_item_seller,
        )
        # Title cap is enforced inside build_jnj_csv_row.

        item_photos = photos_by_item.get(item_num, [])
        for photo_idx, p in enumerate(item_photos[:20], start=1):
            ext = (p.filename or "photo.jpg").rsplit(".", 1)[-1].lower()
            if ext not in ("jpg", "jpeg", "png", "webp", "gif", "bmp", "heic"):
                ext = "jpg"
            # J&J-style sequential name: "<PREFIX> NNN.<ext>", zero-padded to 3.
            sale_photo_seq += 1
            leaf_name = f"{photo_prefix} {sale_photo_seq:03d}.{ext}"
            # v25.21: uploader still shows "Missing Image" for every row even
            # though photos ARE in the ZIP. Hypothesis: their uploader uses the
            # leaf name of the CSV cell to find a file inside the ZIP, and it
            # extracts to a flat working dir before lookup. So put photos at ZIP
            # ROOT with bare leaf names, and write the CSV cell as JUST the bare
            # leaf (no path, no backslash, no ..\). This is what their help
            # doc literally says: "filename of an image included in the uploaded
            # zip file" — filename, not path.
            zip_path = leaf_name
            csv_ref = leaf_name
            data = await p.read()
            # Reset the file position so we don't consume it if it's used again
            await p.seek(0)
            photo_files.append((zip_path, data))
            row[f"image_{photo_idx}"] = csv_ref

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
