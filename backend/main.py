
import os, base64, tempfile, re, hashlib, io, json, time, threading, urllib.request, urllib.error, difflib
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

app = FastAPI(title="TrueTrace OCR API", version="8.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")

@app.get("/")
async def serve_index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_AVAILABLE   = bool(ANTHROPIC_API_KEY)

if CLAUDE_AVAILABLE:
    print("[TrueTrace] Claude Vision API: ENABLED ✓")
else:
    print("[TrueTrace] No Anthropic API key. Add ANTHROPIC_API_KEY to .env")
    print("[TrueTrace] Falling back to EasyOCR + PaddleOCR")

CLAUDE_PROMPT = """You are TrueTrace Vision, an expert OCR system for handwritten and printed text.

Task: Read ALL visible text in the image exactly as written.

Rules:
1. Return the original text only. Do not translate during OCR.
2. Preserve physical line breaks and strict top-to-bottom order.
3. Support multilingual handwriting, including Bangla, English, Arabic, Hindi, Japanese, Chinese, Korean, French, Spanish, Russian, and German.
4. Support mixed-language lines.
5. Do not add corrections, explanations, or extra punctuation unless clearly written.
6. If a character or word is unclear, make the best readable guess instead of skipping it.

Return ONLY this JSON format:
{"lines":[{"text":"first line","confidence":95},{"text":"second line","confidence":90}]}
"""

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-4-5")


def _resize_for_api(img: np.ndarray, max_side: int = 1500, min_width: int = 850) -> np.ndarray:
    """Resize images for faster Claude requests without losing handwriting detail."""
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return img
    max_dim = max(h, w)
    scale = 1.0
    if max_dim > max_side:
        scale = max_side / float(max_dim)
    elif w < min_width:
        scale = min(max_side / float(max_dim), min_width / float(w))
    if abs(scale - 1.0) > 0.01:
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=interp)
    return img


def _claude_lines_to_result(lines: list) -> dict:
    clean_lines = []
    for item in lines or []:
        text = str(item.get("text", "")).strip() if isinstance(item, dict) else str(item).strip()
        if not text:
            continue
        try:
            conf = float(item.get("confidence", 90)) if isinstance(item, dict) else 90.0
        except Exception:
            conf = 90.0
        clean_lines.append({"text": text, "confidence": conf, "word_count": len(text.split())})
    full_text = "\n".join(line["text"] for line in clean_lines)
    confs = [line["confidence"] for line in clean_lines]
    return {
        "full_text": full_text,
        "lines": clean_lines,
        "total_lines": len(clean_lines),
        "word_count": len(full_text.split()),
        "char_count": len(full_text),
        "avg_confidence": round(float(np.mean(confs)), 1) if confs else 0.0,
        "regions_found": len(clean_lines),
    }


def _extract_json_from_claude(raw_text: str, label: str = "Claude") -> dict:
    raw_text = (raw_text or "").strip()
    if raw_text.startswith("```"):
        raw_text = re.sub(r'^```[a-zA-Z]*\n?', '', raw_text)
        raw_text = re.sub(r'\n?```$', '', raw_text)
    raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise RuntimeError(f"{label} returned non-JSON: {raw_text[:200]}")


def run_claude_vision(img: np.ndarray) -> dict:
    """Accurate Claude OCR optimized for normal image, video-frame, and drawing recognition."""
    if not CLAUDE_AVAILABLE:
        raise RuntimeError("Anthropic API key not configured")

    img = _resize_for_api(img, max_side=1500, min_width=850)
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 84])
    b64_img = base64.b64encode(buf.tobytes()).decode('utf-8')

    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": 1536,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64_img
                        }
                    },
                    {"type": "text", "text": CLAUDE_PROMPT}
                ]
            }
        ]
    }

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode('utf-8'),
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            response_data = json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        raise RuntimeError(f"Claude API error {e.code}: {error_body}")

    raw_text = ""
    for block in response_data.get("content", []):
        if block.get("type") == "text":
            raw_text += block.get("text", "")

    parsed = _extract_json_from_claude(raw_text, "Claude")
    return _claude_lines_to_result(parsed.get("lines", []))


CLAUDE_LIVE_PROMPT = """You are a fast expert OCR system for live webcam handwriting recognition.
Read only the handwritten text visible in the image. Preserve exact language and line breaks.
Support Bangla, English, Arabic, Hindi, Japanese, Chinese, Korean, French, Spanish, Russian, German, and mixed-language handwriting.
Return only JSON in this exact format:
{"lines":[{"text":"first visible line","confidence":95},{"text":"second visible line","confidence":90}]}
Rules: do not translate, do not explain, do not add extra words, keep top-to-bottom order."""


def crop_live_document(img: np.ndarray) -> np.ndarray:
    """Crop the bright paper/document region when possible so Claude receives less background."""
    try:
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Notebook/page is usually bright with low-to-medium saturation.
        mask = cv2.inRange(hsv, np.array([0, 0, 110], dtype=np.uint8),
                                np.array([179, 95, 255], dtype=np.uint8))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < 0.18 * h * w:
            return img
        x, y, cw, ch = cv2.boundingRect(cnt)
        margin = int(0.03 * max(w, h))
        x1 = max(0, x - margin); y1 = max(0, y - margin)
        x2 = min(w, x + cw + margin); y2 = min(h, y + ch + margin)
        if (x2 - x1) < 200 or (y2 - y1) < 120:
            return img
        return img[y1:y2, x1:x2]
    except Exception:
        return img


def prepare_live_claude_frame(img: np.ndarray) -> np.ndarray:
    """Light live preprocessing: crop, resize, and improve readability without heavy filters."""
    img = crop_live_document(img)
    h, w = img.shape[:2]
    max_w = 1150
    if w > max_w:
        scale = max_w / float(w)
        img = cv2.resize(img, (max_w, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    elif w < 850:
        scale = 850 / float(max(w, 1))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if float(gray.mean()) < 115:
        gray = cv2.bitwise_not(gray)
    clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # Keep it grayscale/BGR; Claude reads handwriting well, and JPEG stays small.
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def run_claude_vision_live(img: np.ndarray) -> dict:
    """Claude OCR optimized for live scan: smaller JPEG + shorter prompt + fewer tokens."""
    if not CLAUDE_AVAILABLE:
        raise RuntimeError("Anthropic API key not configured")

    img = prepare_live_claude_frame(img)
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 78])
    b64_img = base64.b64encode(buf.tobytes()).decode('utf-8')

    payload = {
        "model": os.environ.get("CLAUDE_LIVE_MODEL", os.environ.get("CLAUDE_MODEL", "claude-opus-4-5")),
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64_img
                        }
                    },
                    {"type": "text", "text": CLAUDE_LIVE_PROMPT}
                ]
            }
        ]
    }

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode('utf-8'),
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=22) as resp:
            response_data = json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        raise RuntimeError(f"Claude live API error {e.code}: {error_body}")

    raw_text = ""
    for block in response_data.get("content", []):
        if block.get("type") == "text":
            raw_text += block["text"]
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = re.sub(r'^```[a-z]*\n?', '', raw_text)
        raw_text = re.sub(r'\n?```$', '', raw_text)
    raw_text = raw_text.strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
        else:
            raise RuntimeError(f"Claude live returned non-JSON: {raw_text[:200]}")

    lines = parsed.get("lines", [])
    clean_lines = []
    for item in lines:
        text = str(item.get("text", "")).strip()
        if text:
            clean_lines.append({
                "text": text,
                "confidence": float(item.get("confidence", 90)),
                "word_count": len(text.split())
            })
    full_text = "\n".join(line["text"] for line in clean_lines)
    confs = [line["confidence"] for line in clean_lines]
    return {
        "full_text": full_text,
        "lines": clean_lines,
        "total_lines": len(clean_lines),
        "word_count": len(full_text.split()),
        "char_count": len(full_text),
        "avg_confidence": round(float(np.mean(confs)), 1) if confs else 0.0,
        "regions_found": len(clean_lines),
    }


CLAUDE_VIDEO_PROMPT = """You are TrueTrace Vision Video OCR, an expert OCR system for uploaded handwriting videos.

You will receive one or more frames from the SAME video. The same page or same text may appear multiple times. A hand or pen may cover some text. Some frames may show incomplete writing.

Task: produce ONE clean final OCR result from the video frames.

Rules:
1. Return the original text only. Do not translate during OCR.
2. Preserve the physical line breaks exactly. Count real written lines, not repeated frames.
3. If the same physical line appears in multiple frames with small differences, keep only the clearest and most complete version.
4. Do not add duplicate lines, alternate readings, summaries, or extra lines.
5. Prefer the sharpest/most complete frame for each line. For handwriting-in-progress, prefer the later complete text.
6. Ignore notebook ruling lines, page margins, hand, pen, background, and UI artifacts.
7. Support multilingual handwriting, including Bangla, English, Arabic, Hindi, Japanese, Chinese, Korean, French, Spanish, Russian, German, and mixed-language lines.
8. If a word is unclear, make the best readable guess instead of creating another extra line.

Return ONLY this JSON format:
{"lines":[{"text":"first physical line","confidence":95},{"text":"second physical line","confidence":90}]}
"""


def _prepare_video_frame_for_claude(img: np.ndarray) -> np.ndarray:
    """Prepare a video frame for accurate Claude video OCR without heavy processing."""
    try:
        img = auto_rotate_frame(img)
    except Exception:
        pass
    try:
        img = crop_live_document(img)
    except Exception:
        pass
    img = _resize_for_api(img, max_side=int(os.environ.get("VIDEO_CLAUDE_MAX_SIDE", "1300")), min_width=850)
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Use light enhancement only. Aggressive thresholding can break Hindi/Chinese/Korean strokes.
        clahe = cv2.createCLAHE(clipLimit=1.25, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        sharp = cv2.addWeighted(enhanced, 1.18, cv2.GaussianBlur(enhanced, (0, 0), 0.9), -0.18, 0)
        return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)
    except Exception:
        return img


def run_claude_vision_video(frame_data: list) -> dict:
    """Accurate and faster video OCR: one Claude request with selected key frames.

    This avoids the old problem where OCR was run on several near-duplicate frames
    and slightly different readings were merged as extra lines.
    """
    if not CLAUDE_AVAILABLE:
        raise RuntimeError("Anthropic API key not configured")
    if not frame_data:
        return _claude_lines_to_result([])

    max_frames = max(1, int(os.environ.get("VIDEO_CLAUDE_FRAMES", "5")))
    ranked = sorted(frame_data, key=lambda f: f.get("score", 0), reverse=True)

    selected = []
    for fd in ranked:
        img = fd.get("img") if isinstance(fd, dict) else fd
        if img is None:
            continue
        # Keep frames that provide new visual information, but avoid sending many
        # near-identical frames of the same page to Claude.
        if any(_frame_similarity(img, kept.get("img")) > 0.94 for kept in selected):
            continue
        selected.append(fd)
        if len(selected) >= max_frames:
            break

    if not selected:
        selected = ranked[:1]
    # Keep chronological order so Claude can understand handwriting progress.
    selected.sort(key=lambda f: f.get("idx", 0) if isinstance(f, dict) else 0)

    content = []
    for i, fd in enumerate(selected, start=1):
        img = fd.get("img") if isinstance(fd, dict) else fd
        prep = _prepare_video_frame_for_claude(img)
        _, buf = cv2.imencode('.jpg', prep, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64_img = base64.b64encode(buf.tobytes()).decode('utf-8')
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64_img}
        })
    content.append({"type": "text", "text": CLAUDE_VIDEO_PROMPT})

    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": 1536,
        "messages": [{"role": "user", "content": content}]
    }

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode('utf-8'),
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=35) as resp:
            response_data = json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        raise RuntimeError(f"Claude video API error {e.code}: {error_body}")

    raw_text = ""
    for block in response_data.get("content", []):
        if block.get("type") == "text":
            raw_text += block.get("text", "")

    parsed = _extract_json_from_claude(raw_text, "Claude video")
    result = _claude_lines_to_result(parsed.get("lines", []))
    result["video_keyframes_used"] = len(selected)
    return result

import easyocr
_easy: Optional[easyocr.Reader] = None

def get_easy() -> easyocr.Reader:
    global _easy
    if _easy is None:
        print("[TrueTrace] Loading EasyOCR [bn, en]…")
        _easy = easyocr.Reader(['bn', 'en'], gpu=False, verbose=False)
        print("[TrueTrace] EasyOCR ready.")
    return _easy

from paddleocr import PaddleOCR

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
    print("[TrueTrace] Translation engine ready.")
except ImportError:
    TRANSLATOR_AVAILABLE = False
    print("[TrueTrace] deep-translator not installed. Run: pip install deep-translator --break-system-packages")

SUPPORTED_LANGUAGES = {
    "auto":    {"name": "Auto Detect",         "easy": ["en","bn"],       "flag": "🌐"},
    "bn_en":   {"name": "Bangla + English",     "easy": ["bn","en"],       "flag": "🇧🇩"},
    "en":      {"name": "English",              "easy": ["en"],            "flag": "🇬🇧"},
    "zh_cn":   {"name": "Chinese (Simplified)", "easy": ["ch_sim","en"],   "flag": "🇨🇳"},
    "ar":      {"name": "Arabic",               "easy": ["ar","en"],       "flag": "🇸🇦"},
    "hi":      {"name": "Hindi",                "easy": ["hi","en"],       "flag": "🇮🇳"},
    "fr":      {"name": "French",               "easy": ["fr","en"],       "flag": "🇫🇷"},
    "es":      {"name": "Spanish",              "easy": ["es","en"],       "flag": "🇪🇸"},
    "ru":      {"name": "Russian",              "easy": ["ru","en"],       "flag": "🇷🇺"},
    "ja":      {"name": "Japanese",             "easy": ["ja","en"],       "flag": "🇯🇵"},
    "ko":      {"name": "Korean",               "easy": ["ko","en"],       "flag": "🇰🇷"},
    "de":      {"name": "German",               "easy": ["de","en"],       "flag": "🇩🇪"},
}

_easy_readers: dict = {}

def get_easy_for_lang(lang_key: str) -> easyocr.Reader:
    global _easy_readers
    lang_cfg = SUPPORTED_LANGUAGES.get(lang_key, SUPPORTED_LANGUAGES["bn_en"])
    lang_list = lang_cfg["easy"]
    cache_key = "_".join(sorted(lang_list))
    if cache_key not in _easy_readers:
        print(f"[TrueTrace] Loading EasyOCR for {lang_cfg['name']}...")
        _easy_readers[cache_key] = easyocr.Reader(lang_list, gpu=False, verbose=False)
        print(f"[TrueTrace] EasyOCR {lang_cfg['name']} ready.")
    return _easy_readers[cache_key]
_paddle: Optional[PaddleOCR] = None

def get_paddle() -> PaddleOCR:
    global _paddle
    if _paddle is None:
        print("[TrueTrace] Loading PaddleOCR [en]…")
        configs = [
            dict(use_textline_orientation=True, lang='en'),
            dict(lang='en'),
            dict(use_angle_cls=True, lang='en', use_gpu=False,
                 show_log=False, ocr_version='PP-OCRv4',
                 det_db_thresh=0.3, det_db_box_thresh=0.5,
                 det_db_unclip_ratio=2.0),
        ]
        last_err = None
        for cfg in configs:
            try:
                _paddle = PaddleOCR(**cfg)
                break
            except Exception as e:
                last_err = e
                continue
        else:
            print(f"[TrueTrace] WARNING: PaddleOCR failed to load: {last_err}")
            print("[TrueTrace] Continuing without PaddleOCR — EasyOCR will be used instead.")
            _paddle = None
            return None
        print("[TrueTrace] PaddleOCR ready.")
    return _paddle

def preprocess_for_claude(img: np.ndarray) -> np.ndarray:
    """Fast, safe preprocessing for Claude OCR on images and video frames."""
    try:
        img = crop_live_document(img)
    except Exception:
        pass
    img = _resize_for_api(img, max_side=1500, min_width=850)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if float(gray.mean()) < 115:
        gray = cv2.bitwise_not(gray)
    clahe = cv2.createCLAHE(clipLimit=1.35, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # Light sharpening helps handwriting while staying much faster than heavy threshold pipelines.
    sharp = cv2.addWeighted(enhanced, 1.25, cv2.GaussianBlur(enhanced, (0, 0), 1.0), -0.25, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

def preprocess_english(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if w < 1500:
        scale = 1500 / w
        img = cv2.resize(img, (int(w*scale), int(h*scale)),
                         interpolation=cv2.INTER_CUBIC)
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
    thresh   = cv2.adaptiveThreshold(denoised, 255,
                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                   cv2.THRESH_BINARY, blockSize=21, C=10)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

def preprocess_bangla(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if w < 2000:
        scale = 2000 / w
        img = cv2.resize(img, (int(w*scale), int(h*scale)),
                         interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if float(gray.mean()) < 127:
        gray = cv2.bitwise_not(gray)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    denoised = cv2.GaussianBlur(enhanced, (3,3), 0)
    kernel   = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
    sharp    = cv2.filter2D(denoised, -1, kernel)
    _, thresh = cv2.threshold(sharp, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

def crop_canvas_content(img: np.ndarray) -> np.ndarray:
    """Crop empty canvas margins so drawing OCR sends only the written area."""
    try:
        h, w = img.shape[:2]
        s = max(6, min(h, w) // 30)
        samples = np.concatenate([
            img[:s, :s].reshape(-1, 3), img[:s, -s:].reshape(-1, 3),
            img[-s:, :s].reshape(-1, 3), img[-s:, -s:].reshape(-1, 3)
        ], axis=0)
        bg = np.median(samples.astype(np.float32), axis=0)
        diff = np.linalg.norm(img.astype(np.float32) - bg, axis=2)
        mask = (diff > 24).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img
        x, y, cw, ch = cv2.boundingRect(np.vstack(contours))
        if cw < 20 or ch < 20:
            return img
        margin = max(22, int(0.08 * max(cw, ch)))
        x1 = max(0, x - margin); y1 = max(0, y - margin)
        x2 = min(w, x + cw + margin); y2 = min(h, y + ch + margin)
        return img[y1:y2, x1:x2]
    except Exception:
        return img


def preprocess_canvas(img: np.ndarray) -> np.ndarray:
    img = crop_canvas_content(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if float(gray.mean()) < 127:
        gray = cv2.bitwise_not(gray)
    h, w = gray.shape[:2]
    max_side = 1400
    min_width = 850
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
    elif w < min_width:
        scale = min(max_side / float(max(h, w)), min_width / float(max(w, 1)))
    if abs(scale - 1.0) > 0.01:
        gray = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))),
                          interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

BANGLA_RANGE = range(0x0980, 0x09FF + 1)
def text_has_bangla(text: str) -> bool:
    return any(ord(c) in BANGLA_RANGE for c in text)

def _top_y(b):    return min(p[1] for p in b)
def _bot_y(b):    return max(p[1] for p in b)
def _left_x(b):   return min(p[0] for p in b)
def _center_y(b): return (_top_y(b) + _bot_y(b)) / 2.0

def sort_into_lines(detections: list) -> list:
    if not detections:
        return []
    heights = [_bot_y(d["bbox"]) - _top_y(d["bbox"]) for d in detections]
    tol     = float(np.median(heights)) * 0.55 if heights else 20.0
    sorted_dets = sorted(detections, key=lambda d: _top_y(d["bbox"]))
    lines, cur, cur_y = [], [], 0.0
    for det in sorted_dets:
        cy = _center_y(det["bbox"])
        if not cur:
            cur.append(det); cur_y = cy
        elif abs(cy - cur_y) <= tol:
            cur.append(det)
            cur_y = float(np.mean([_center_y(d["bbox"]) for d in cur]))
        else:
            cur.sort(key=lambda d: _left_x(d["bbox"]))
            lines.append(cur); cur = [det]; cur_y = cy
    if cur:
        cur.sort(key=lambda d: _left_x(d["bbox"]))
        lines.append(cur)
    return lines

def build_result(lines: list, dets: list) -> dict:
    line_data, line_texts = [], []
    for line in lines:
        text = " ".join(d["text"] for d in line)
        conf = round(float(np.mean([d["confidence"] for d in line])), 1)
        line_data.append({"text": text, "confidence": conf, "word_count": len(line)})
        line_texts.append(text)
    full_text = "\n".join(line_texts)
    all_confs = [d["confidence"] for d in dets]
    return {
        "full_text":      full_text,
        "lines":          line_data,
        "total_lines":    len(line_data),
        "word_count":     sum(len(t.split()) for t in line_texts),
        "char_count":     len(full_text.replace("\n"," ")),
        "avg_confidence": round(float(np.mean(all_confs)),1) if all_confs else 0.0,
        "regions_found":  len(dets),
    }

def run_easyocr(img: np.ndarray, bangla_mode: bool = False, lang_key: str = 'bn_en') -> dict:
    reader = get_easy_for_lang(lang_key) if lang_key != 'bn_en' else get_easy()
    proc   = preprocess_bangla(img) if bangla_mode else preprocess_english(img)
    raw = reader.readtext(proc, detail=1, paragraph=False,
                          contrast_ths=0.1, adjust_contrast=0.5,
                          width_ths=0.7, ycenter_ths=0.5,
                          add_margin=0.1, text_threshold=0.5,
                          low_text=0.3, link_threshold=0.3)
    dets = []
    for bbox, text, conf in raw:
        text = str(text).strip()
        if conf < 0.08 or not text: continue
        dets.append({"text": text, "confidence": round(conf*100,1),
                     "bbox": [[float(p[0]),float(p[1])] for p in bbox]})
    if not dets:
        raw2 = reader.readtext(img, detail=1, paragraph=False,
                               text_threshold=0.4, low_text=0.2)
        for bbox, text, conf in raw2:
            text = str(text).strip()
            if conf < 0.05 or not text: continue
            dets.append({"text": text, "confidence": round(conf*100,1),
                         "bbox": [[float(p[0]),float(p[1])] for p in bbox]})
    return build_result(sort_into_lines(dets), dets)

def run_paddle(img: np.ndarray) -> dict:
    ocr = get_paddle()
    if ocr is None:
        return {"full_text":"","lines":[],"total_lines":0,"word_count":0,
                "char_count":0,"avg_confidence":0.0,"regions_found":0}
    proc = preprocess_english(img)
    def _ocr(src):
        try:
            return src.ocr(preprocess_english(img), cls=True)
        except Exception:
            try:
                return src.ocr(preprocess_english(img))
            except Exception:
                return None
    def _parse(raw):
        dets = []
        if not raw: return dets
        page = raw[0] if isinstance(raw, list) else []
        if not page: return dets
        for item in page:
            try:
                bbox  = item[0]; txt = str(item[1][0]).strip()
                score = float(item[1][1])
                if score < 0.10 or not txt: continue
                dets.append({"text": txt, "confidence": round(score*100,1),
                             "bbox": [[float(p[0]),float(p[1])] for p in bbox]})
            except: continue
        return dets
    dets = _parse(_ocr(ocr))
    if not dets:
        dets = _parse(ocr.ocr(img) if ocr else None)
    return build_result(sort_into_lines(dets), dets)


def resize_for_live(img: np.ndarray, max_width: int = 900) -> np.ndarray:
    """Keep live webcam frames small so OCR can return quickly."""
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        img = cv2.resize(img, (max_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return img


def preprocess_live_fast(img: np.ndarray) -> np.ndarray:
    """Light preprocessing for 1-second live scan; avoids expensive upscaling/filtering."""
    img = resize_for_live(img, 900)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if float(gray.mean()) < 115:
        gray = cv2.bitwise_not(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def run_ocr_fast(img: np.ndarray, lang_key: str = "auto") -> tuple:
    """
    Fast live-scan OCR.
    Uses local EasyOCR only, skips Claude/Paddle fallback and overlay generation.
    This keeps webcam updates much faster while preserving normal upload/video accuracy.
    """
    proc = preprocess_live_fast(img)

    # Auto live scan should not call Claude. Use the already-warmed Bangla+English reader.
    effective_lang = lang_key if lang_key in SUPPORTED_LANGUAGES else "bn_en"
    if effective_lang in ("auto", "bn_en"):
        reader = get_easy()
        engine = "TrueTrace·Live Fast"
    else:
        reader = get_easy_for_lang(effective_lang)
        engine = "TrueTrace·Live Fast"

    try:
        raw = reader.readtext(
            proc,
            detail=1,
            paragraph=False,
            contrast_ths=0.1,
            adjust_contrast=0.5,
            width_ths=0.7,
            ycenter_ths=0.5,
            add_margin=0.08,
            text_threshold=0.4,
            low_text=0.25,
            link_threshold=0.3,
        )
    except Exception as e:
        print(f"[TrueTrace] Fast live OCR failed: {e}")
        raw = []

    dets = []
    for bbox, text, conf in raw:
        text = str(text).strip()
        if conf < 0.05 or not text:
            continue
        dets.append({
            "text": text,
            "confidence": round(float(conf) * 100, 1),
            "bbox": [[float(p[0]), float(p[1])] for p in bbox],
        })

    return build_result(sort_into_lines(dets), dets), engine, []

def run_ocr_smart(img: np.ndarray, lang_key: str = "auto") -> tuple:
    ocr_raw = []

    def get_easyocr_bboxes(src_img: np.ndarray, bangla: bool = False) -> list:
        try:
            reader = get_easy()
            proc   = preprocess_bangla(src_img) if bangla else preprocess_english(src_img)
            raw    = reader.readtext(proc, detail=1, paragraph=False,
                                     text_threshold=0.4, low_text=0.2)
            return [(b, t, c) for b, t, c in raw if c >= 0.08 and t.strip()]
        except Exception:
            return []

    def get_paddle_bboxes(src_img: np.ndarray) -> list:
        try:
            ocr = get_paddle()
            if ocr is None:
                return []
            proc = preprocess_english(src_img)
            try:
                raw = ocr.ocr(proc, cls=True)
            except Exception:
                raw = ocr.ocr(proc)
            if not raw or not raw[0]:
                return []
            page = raw[0] if isinstance(raw, list) else []
            result = []
            for item in page:
                try:
                    box, (txt, score) = item[0], item[1]
                    if float(score) >= 0.1 and str(txt).strip():
                        result.append((box, str(txt).strip(), float(score)))
                except Exception:
                    continue
            return result
        except Exception:
            return []

    if CLAUDE_AVAILABLE:
        try:
            proc   = preprocess_for_claude(img)
            result = run_claude_vision(proc)
            if result["total_lines"] > 0:
                # Claude already returns line text. Avoid extra Paddle/EasyOCR bbox pass here;
                # it was the main reason image, video, and drawing recognition felt slow.
                return result, "TrueTrace·Vision", []
        except Exception as e:
            print(f"[TrueTrace] Vision failed: {e} — falling back")

    if lang_key not in ("auto", "bn_en") and lang_key in SUPPORTED_LANGUAGES:
        try:
            reader = get_easy_for_lang(lang_key)
            proc   = preprocess_english(img)
            raw    = reader.readtext(proc, detail=1, paragraph=False,
                                     text_threshold=0.5, low_text=0.3)
            ocr_raw = [(b, t, c) for b, t, c in raw if c >= 0.1 and t.strip()]
            dets    = [{"text": t.strip(), "confidence": round(c*100,1),
                        "bbox": [[float(p[0]),float(p[1])] for p in b]}
                       for b, t, c in ocr_raw]
            lines   = sort_into_lines(dets)
            result  = build_result(lines, dets)
            if result["total_lines"] > 0:
                return result, "TrueTrace·OCR", ocr_raw
        except Exception as e:
            print(f"[TrueTrace] Lang-specific OCR failed: {e}")

    try:
        paddle_raw = get_paddle_bboxes(img)
        if paddle_raw:
            dets = [{"text": t, "confidence": round(c*100,1),
                     "bbox": [[float(p[0]),float(p[1])] for p in b]}
                    for b, t, c in paddle_raw]
            lines  = sort_into_lines(dets)
            result = build_result(lines, dets)
            if result["total_lines"] > 0 and not text_has_bangla(result["full_text"]):
                return result, "TrueTrace·OCR", paddle_raw
    except Exception as e:
        print(f"[TrueTrace] PaddleOCR failed: {e}")

    try:
        reader  = get_easy()
        proc    = preprocess_bangla(img)
        raw     = reader.readtext(proc, detail=1, paragraph=False,
                                  contrast_ths=0.1, adjust_contrast=0.5,
                                  width_ths=0.7, text_threshold=0.5, low_text=0.3)
        ocr_raw = [(b, t, c) for b, t, c in raw if c >= 0.08 and t.strip()]
        dets    = [{"text": t.strip(), "confidence": round(c*100,1),
                    "bbox": [[float(p[0]),float(p[1])] for p in b]}
                   for b, t, c in ocr_raw]
        lines   = sort_into_lines(dets)
        result  = build_result(lines, dets)
        if result["total_lines"] > 0:
            return result, "TrueTrace·Bangla", ocr_raw
    except Exception as e:
        print(f"[TrueTrace] EasyOCR Bangla failed: {e}")

    try:
        reader  = get_easy()
        proc    = preprocess_english(img)
        raw     = reader.readtext(proc, detail=1, paragraph=False)
        ocr_raw = [(b, t, c) for b, t, c in raw if c >= 0.08 and t.strip()]
        dets    = [{"text": t.strip(), "confidence": round(c*100,1),
                    "bbox": [[float(p[0]),float(p[1])] for p in b]}
                   for b, t, c in ocr_raw]
        lines   = sort_into_lines(dets)
        result  = build_result(lines, dets)
        return result, "TrueTrace·English", ocr_raw
    except Exception as e:
        print(f"[TrueTrace] EasyOCR English failed: {e}")
        return {"full_text":"","lines":[],"total_lines":0,"word_count":0,
                "char_count":0,"avg_confidence":0.0,"regions_found":0}, "None", []

def run_canvas_smart(img: np.ndarray) -> tuple:
    proc = preprocess_canvas(img)

    if CLAUDE_AVAILABLE:
        try:
            result = run_claude_vision(proc)
            if result["total_lines"] > 0:
                return result, "TrueTrace·Vision"
        except Exception as e:
            print(f"[TrueTrace] Claude canvas failed: {e}")

    try:
        res_bn = run_easyocr(preprocess_bangla(proc), bangla_mode=False)
    except:
        res_bn = {"total_lines":0,"avg_confidence":0}
    try:
        res_en = run_easyocr(preprocess_english(proc), bangla_mode=False)
    except:
        res_en = {"total_lines":0,"avg_confidence":0}

    if res_bn.get("total_lines",0) >= res_en.get("total_lines",0) and res_bn.get("total_lines",0) > 0:
        return res_bn, "TrueTrace·Bangla"
    elif res_en.get("total_lines",0) > 0:
        return res_en, "TrueTrace·English"
    return res_bn, "TrueTrace·Bangla"

def _frame_hash(f: np.ndarray) -> str:
    return hashlib.md5(
        cv2.cvtColor(cv2.resize(f,(32,32)), cv2.COLOR_BGR2GRAY).tobytes()
    ).hexdigest()

def _frame_hash(frame: np.ndarray) -> str:
    small = cv2.resize(frame, (32, 32))
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return hashlib.md5(gray.tobytes()).hexdigest()

def auto_rotate_frame(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if h > w:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame

def _skin_coverage(img: np.ndarray) -> float:
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0,20,70], dtype=np.uint8),
                           np.array([20,255,255], dtype=np.uint8))
    return float(mask.sum()) / (255.0 * img.shape[0] * img.shape[1])

def _text_density(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    return float(b.sum()) / (255.0 * img.shape[0] * img.shape[1])

def _sharpness(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def _frame_similarity(f1: np.ndarray, f2: np.ndarray) -> float:
    h1 = cv2.resize(f1, (64, 64))
    h2 = cv2.resize(f2, (64, 64))
    g1 = cv2.cvtColor(h1, cv2.COLOR_BGR2GRAY).astype(float)
    g2 = cv2.cvtColor(h2, cv2.COLOR_BGR2GRAY).astype(float)
    diff = np.abs(g1 - g2)
    return 1.0 - (diff.mean() / 255.0)

def extract_frames_1fps(video_bytes: bytes) -> list:
    """Select sharp, useful key frames across the whole video.

    Fixes two video issues:
    - Long videos are sampled across the full duration, not only the first part.
    - Near-duplicate frames are reduced so the same line is not recognized twice.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes); path = tmp.name

    all_frames = []
    try:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total / fps if fps else 0
        if total <= 0:
            cap.release()
            return []

        max_samples = int(os.environ.get("VIDEO_MAX_SAMPLED_FRAMES", "90"))
        # Sample approximately 1 FPS, but spread samples over the full video when
        # the video is longer than max_samples seconds.
        target_samples = min(max_samples, max(1, int(duration) + 1))
        if target_samples >= total:
            indices = list(range(total))
        else:
            indices = sorted({int(round(x)) for x in np.linspace(0, total - 1, target_samples)})

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            rotated = auto_rotate_frame(frame.copy())
            text_density = _text_density(rotated)
            sharpness = _sharpness(rotated)
            skin = _skin_coverage(rotated)
            # Prefer sharp, text-heavy frames. Penalize large hand coverage, but
            # do not discard it completely because writing videos often contain a pen/hand.
            score = (max(text_density, 0.001) * max(sharpness, 1.0)) / (skin + 0.035)
            all_frames.append({
                "img": rotated,
                "idx": idx,
                "time_sec": round(idx / fps, 2) if fps else 0,
                "skin": skin,
                "text": text_density,
                "sharpness": sharpness,
                "score": score,
            })
        cap.release()
    finally:
        try: os.unlink(path)
        except Exception: pass

    if not all_frames:
        return []

    ranked = sorted(all_frames, key=lambda f: f["score"], reverse=True)
    selected = []
    max_ocr_frames = int(os.environ.get("VIDEO_MAX_OCR_FRAMES", "6"))
    for frame in ranked:
        if frame["sharpness"] < 14:
            continue
        # Strong duplicate filtering for same page/video angle.
        if any(_frame_similarity(frame["img"], kept["img"]) > 0.92 for kept in selected):
            continue
        selected.append(frame)
        if len(selected) >= max_ocr_frames:
            break

    if not selected:
        selected = ranked[:min(3, len(ranked))]

    selected.sort(key=lambda f: f["idx"])
    return selected

def _normalize_line_for_match(text: str) -> str:
    text = (text or "").lower().strip()
    # Keep letters/numbers from all scripts; remove punctuation, symbols, and spaces.
    return "".join(ch for ch in text if ch.isalnum())


def _same_physical_line(a: str, b: str) -> bool:
    na, nb = _normalize_line_for_match(a), _normalize_line_for_match(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
    # In writing videos, earlier frames often contain a partial line. Treat it as
    # the same physical line when one is mostly contained in the other.
    if len(shorter) >= 6 and shorter in longer and len(shorter) / max(len(longer), 1) >= 0.55:
        return True
    ratio = difflib.SequenceMatcher(None, na, nb).ratio()
    return ratio >= 0.72


def _line_quality(line: dict) -> float:
    text = str(line.get("text", ""))
    conf = float(line.get("confidence", 0) or 0)
    # Prefer the most complete line, then confidence.
    return len(_normalize_line_for_match(text)) * 1.8 + conf


def merge_video_results(frame_data: list, ocr_results: list) -> dict:
    """Merge local/fallback OCR results without duplicating the same physical line."""
    if not ocr_results:
        return {"full_text": "", "lines": [], "total_lines": 0,
                "word_count": 0, "char_count": 0, "avg_confidence": 0.0, "regions_found": 0}

    def score(res):
        n = res.get("total_lines", 0)
        conf = res.get("avg_confidence", 0)
        chars = len(res.get("full_text", ""))
        return (n * conf) + (chars * 0.25)

    ranked = sorted(ocr_results, key=score, reverse=True)

    # If most frames are the same scene, start from the single best result. This
    # prevents alternate OCR readings from becoming fake extra lines.
    merged_lines = []
    for res in ranked:
        for ln in res.get("lines", []):
            t = str(ln.get("text", "")).strip()
            if len(_normalize_line_for_match(t)) < 2:
                continue
            match_idx = None
            for i, kept in enumerate(merged_lines):
                if _same_physical_line(t, kept.get("text", "")):
                    match_idx = i
                    break
            if match_idx is None:
                merged_lines.append(dict(ln))
            else:
                # Replace partial/low-quality variant with the clearest variant.
                if _line_quality(ln) > _line_quality(merged_lines[match_idx]):
                    merged_lines[match_idx] = dict(ln)

    # Keep the line order from the best OCR result as much as possible. Then add
    # genuinely new lines after it for videos that contain more than one page/scene.
    best_texts = [ln.get("text", "") for ln in ranked[0].get("lines", [])]
    ordered = []
    used = set()
    for bt in best_texts:
        for i, ln in enumerate(merged_lines):
            if i in used:
                continue
            if _same_physical_line(bt, ln.get("text", "")):
                ordered.append(ln); used.add(i); break
    for i, ln in enumerate(merged_lines):
        if i not in used:
            ordered.append(ln)

    full = "\n".join(ln["text"] for ln in ordered)
    confs = [float(ln.get("confidence", 0) or 0) for ln in ordered]
    return {
        "full_text": full,
        "lines": ordered,
        "total_lines": len(ordered),
        "word_count": sum(len(ln["text"].split()) for ln in ordered),
        "char_count": len(full.replace("\n", " ")),
        "avg_confidence": round(float(np.mean(confs)), 1) if confs else 0.0,
        "regions_found": len(ordered),
    }

def generate_overlay_image(img: np.ndarray, ocr_raw: list) -> str:
    # Keep preview image lightweight; OCR text is already returned separately.
    try:
        h, w = img.shape[:2]
        max_side = 1000
        if max(h, w) > max_side:
            scale = max_side / float(max(h, w))
            img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    except Exception:
        pass
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

def translate_text_with_claude(text: str, target_lang: str) -> str:
    """High quality multilingual translation using Claude when an API key is available."""
    if not CLAUDE_AVAILABLE or not text.strip():
        return text
    target_name = "Bangla" if target_lang == "bn" else "English"
    prompt = f"""Translate the following OCR text into {target_name} accurately.

Rules:
- The source text may be Bangla, English, Japanese, Chinese, Korean, Arabic, Hindi, French, Spanish, Russian, German, or mixed-language text.
- Preserve the original line breaks as much as possible.
- Translate meaning naturally, not word-by-word.
- Keep names, numbers, code, and technical terms unchanged when appropriate.
- Return only the translated text. Do not explain.

OCR text:
{text}"""
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode('utf-8'),
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01"
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            response_data = json.loads(resp.read().decode('utf-8'))
        out = ""
        for block in response_data.get("content", []):
            if block.get("type") == "text":
                out += block.get("text", "")
        return out.strip() or text
    except Exception as e:
        print(f"[TrueTrace] Claude translation failed: {e}")
        return text


def translate_text(text: str, target_lang: str) -> str:
    if not text.strip():
        return text
    if target_lang not in ("en", "bn"):
        return text

    # Claude handles Japanese/Chinese/Korean/Arabic/etc. much better than line-by-line auto Google Translate.
    if CLAUDE_AVAILABLE:
        translated = translate_text_with_claude(text, target_lang)
        if translated.strip() and translated.strip() != text.strip():
            return translated

    if not TRANSLATOR_AVAILABLE:
        return text

    try:
        dest = "bn" if target_lang == "bn" else "en"
        # Translate the full text together to keep context. Split only for very long input.
        chunks, buf = [], []
        cur_len = 0
        for line in text.split("\n"):
            if cur_len + len(line) > 3500 and buf:
                chunks.append("\n".join(buf)); buf = []; cur_len = 0
            buf.append(line); cur_len += len(line) + 1
        if buf:
            chunks.append("\n".join(buf))
        translated_chunks = []
        translator = GoogleTranslator(source="auto", target=dest)
        for chunk in chunks:
            translated_chunks.append(translator.translate(chunk) or chunk)
        return "\n".join(translated_chunks)
    except Exception as e:
        print(f"[TrueTrace] Translation error: {e}")
        return text

def bytes_to_bgr(d: bytes) -> Optional[np.ndarray]:
    return cv2.imdecode(np.frombuffer(d, np.uint8), cv2.IMREAD_COLOR)

def b64_to_bgr(b64: str) -> Optional[np.ndarray]:
    if "," in b64: b64 = b64.split(",",1)[1]
    return bytes_to_bgr(base64.b64decode(b64))

@app.get("/api/health")
async def health():
    return {"status":"ok", "vision_engine": CLAUDE_AVAILABLE,
            "engines":["TrueTrace·Vision","EasyOCR bn+en","PaddleOCR en"],
            "version":"8.0.0"}

@app.post("/api/extract")
async def extract(file: UploadFile = File(...), preprocess: str = Form("true"), lang_key: str = Form("auto"), translate_to: str = Form("")):
    data   = await file.read()
    if not data: raise HTTPException(400, "Empty file.")
    suffix = Path(file.filename or "").suffix.lower()
    IMAGES = {".jpg",".jpeg",".png",".webp",".bmp",".tiff",".gif"}
    VIDEOS = {".mp4",".mov",".avi",".mkv",".webm",".flv"}
    try:
        if suffix in IMAGES:
            img = bytes_to_bgr(data)
            if img is None: raise HTTPException(400,"Cannot decode image.")
            result, engine, ocr_raw = run_ocr_smart(img, lang_key=lang_key)
            source, frames = "image", 1
            overlay_img = generate_overlay_image(img, ocr_raw)
        elif suffix in VIDEOS:
            scene_frames = extract_frames_1fps(data)
            if not scene_frames: raise HTTPException(400,"No frames extracted.")
            all_raw = []

            # Use one consolidated Claude video OCR request instead of OCRing
            # multiple frames separately. This is faster and prevents fake extra
            # lines caused by slightly different readings from near-duplicate frames.
            if CLAUDE_AVAILABLE:
                try:
                    result = run_claude_vision_video(scene_frames)
                    engine = "TrueTrace·Vision Video"
                except Exception as e:
                    print(f"[TrueTrace] Claude video OCR failed: {e} — falling back")
                    frame_results, engines_used = [], set()
                    for fd in scene_frames:
                        r, eng, raw = run_ocr_smart(fd["img"], lang_key=lang_key)
                        frame_results.append(r); engines_used.add(eng); all_raw.extend(raw)
                    result = merge_video_results(scene_frames, frame_results)
                    engine = " + ".join(engines_used) if engines_used else "TrueTrace·Video OCR"
            else:
                frame_results, engines_used = [], set()
                for fd in scene_frames:
                    r, eng, raw = run_ocr_smart(fd["img"], lang_key=lang_key)
                    frame_results.append(r); engines_used.add(eng); all_raw.extend(raw)
                result = merge_video_results(scene_frames, frame_results)
                engine = " + ".join(engines_used) if engines_used else "TrueTrace·Video OCR"

            source = "video"; frames = len(scene_frames)
            best_frame = max(scene_frames, key=lambda f: f.get("score", 0))
            overlay_img = generate_overlay_image(best_frame["img"], all_raw[:20])
        else:
            img = bytes_to_bgr(data)
            if img is None: raise HTTPException(400,f"Unsupported: {suffix}")
            result, engine = run_ocr_smart(img)
            source, frames = "image", 1
        translated_text = ""
        if translate_to in ("en", "bn") and result.get("full_text"):
            translated_text = translate_text(result["full_text"], translate_to)

        return {"success":True,"source":source,"filename":file.filename,
                "frames_processed":frames,"engine_used":engine,
                "languages":["Bangla","English"],
                "overlay_image": overlay_img if "overlay_img" in dir() else "",
                "translated_text": translated_text,
                "translate_to": translate_to,
                **result}
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, str(e))



# ── Smart Claude Live Scan ---------------------------------------------------
# Claude Vision is accurate, but a real API call cannot reliably finish every
# second. This scheduler keeps the webcam endpoint fast by returning the latest
# completed Claude result immediately, while one background Claude job processes
# a sharp/stable frame. The browser can still scan every second without freezing.
LIVE_EXECUTOR = ThreadPoolExecutor(max_workers=1)
LIVE_LOCK = threading.Lock()
LIVE_FUTURE = None
LIVE_LAST_RESULT = None
LIVE_LAST_RESULT_ID = ""
LIVE_LAST_SUBMIT_TIME = 0.0
LIVE_LAST_SMALL = None
LIVE_LAST_ERROR = ""


def _empty_result() -> dict:
    return {
        "full_text": "",
        "lines": [],
        "total_lines": 0,
        "word_count": 0,
        "char_count": 0,
        "avg_confidence": 0.0,
        "regions_found": 0,
    }


def _live_small_frame(img: np.ndarray) -> np.ndarray:
    small = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _live_changed_enough(img: np.ndarray) -> bool:
    global LIVE_LAST_SMALL
    small = _live_small_frame(img)
    if LIVE_LAST_SMALL is None:
        LIVE_LAST_SMALL = small
        return True
    diff = float(np.mean(np.abs(small - LIVE_LAST_SMALL)))
    if diff > 5.0:
        LIVE_LAST_SMALL = small
        return True
    return False


def _live_quality_ok(img: np.ndarray) -> bool:
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return sharp >= 12.0
    except Exception:
        return True


def _make_live_payload(result: dict, engine: str, translate_to: str = "", status: str = "updated") -> dict:
    translated_text = ""
    if translate_to in ("en", "bn") and result.get("full_text"):
        translated_text = translate_text(result["full_text"], translate_to)
    return {
        "success": True,
        "source": "webcam",
        "filename": "live_frame",
        "frames_processed": 1,
        "engine_used": engine,
        "languages": ["Bangla", "English"],
        "overlay_image": "",
        "translated_text": translated_text,
        "translate_to": translate_to,
        "live_status": status,
        **result,
    }


def _process_live_claude_job(img: np.ndarray, translate_to: str) -> dict:
    result = run_claude_vision_live(img)
    return _make_live_payload(result, "TrueTrace·Vision Live", translate_to, status="updated")


def _live_done_callback(future):
    global LIVE_LAST_RESULT, LIVE_LAST_RESULT_ID, LIVE_LAST_ERROR
    try:
        payload = future.result()
        payload["live_completed_at"] = time.time()
        result_id = hashlib.md5((payload.get("full_text", "") + str(payload.get("avg_confidence", ""))).encode("utf-8")).hexdigest()
        payload["live_result_id"] = result_id
        with LIVE_LOCK:
            LIVE_LAST_RESULT = payload
            LIVE_LAST_RESULT_ID = result_id
            LIVE_LAST_ERROR = ""
    except Exception as e:
        with LIVE_LOCK:
            LIVE_LAST_ERROR = str(e)
        print(f"[TrueTrace] Claude live scan failed: {e}")


def _live_submit_if_needed(img: np.ndarray, translate_to: str) -> str:
    global LIVE_FUTURE, LIVE_LAST_SUBMIT_TIME
    now = time.time()
    with LIVE_LOCK:
        running = LIVE_FUTURE is not None and not LIVE_FUTURE.done()
        last_submit = LIVE_LAST_SUBMIT_TIME
    if running:
        return "processing"
    # Do not send the same shaky/live frame to Claude every second. Send only a
    # changed and readable frame, at most once every 3 seconds.
    if now - last_submit < 3.0:
        return "cached"
    if not _live_quality_ok(img):
        return "waiting_sharp_frame"
    if not _live_changed_enough(img):
        return "cached"
    with LIVE_LOCK:
        LIVE_LAST_SUBMIT_TIME = now
        LIVE_FUTURE = LIVE_EXECUTOR.submit(_process_live_claude_job, img.copy(), translate_to)
        LIVE_FUTURE.add_done_callback(_live_done_callback)
    return "processing"

@app.post("/api/extract-webcam-fast")
async def extract_webcam_fast(file: UploadFile = File(...), lang_key: str = Form("auto"), translate_to: str = Form("")):
    """Live webcam OCR.

    With Claude configured, this endpoint returns immediately with the latest
    completed Claude result and starts a new Claude job in the background only
    when the frame is sharp, changed, and not already being processed. This
    gives a smooth 1-second live scan UI while keeping Claude accuracy.

    Without Claude configured, it falls back to the previous local fast OCR.
    """
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty webcam frame.")
    img = bytes_to_bgr(data)
    if img is None:
        raise HTTPException(400, "Cannot decode webcam frame.")

    # Accurate mode: Claude Vision, but non-blocking/cached for live scan speed.
    if CLAUDE_AVAILABLE:
        status = _live_submit_if_needed(img, translate_to)
        with LIVE_LOCK:
            last = dict(LIVE_LAST_RESULT) if LIVE_LAST_RESULT else None
            last_error = LIVE_LAST_ERROR
        if last:
            last["live_status"] = status
            return last
        payload = _make_live_payload(_empty_result(), "TrueTrace·Vision Live", "", status=status)
        payload["live_result_id"] = "initial"
        if last_error:
            payload["live_error"] = last_error
        return payload

    # Fallback mode: local OCR when no Claude key exists.
    result, engine, _ = run_ocr_fast(img, lang_key=lang_key)
    payload = _make_live_payload(result, engine, translate_to, status="local_fast")
    payload["live_result_id"] = hashlib.md5(payload.get("full_text", "").encode("utf-8")).hexdigest()
    return payload

@app.post("/api/extract-drawing")
async def extract_drawing(payload: dict):
    b64 = payload.get("image_base64","")
    if not b64: raise HTTPException(400,"No image data.")
    img = b64_to_bgr(b64)
    if img is None: raise HTTPException(400,"Cannot decode drawing.")
    result, engine = run_canvas_smart(img)
    return {"success":True,"source":"drawing","filename":"drawing_board",
            "frames_processed":1,"engine_used":engine,
            "languages":["Bangla","English"], **result}

@app.post("/api/extract-webcam")
async def extract_webcam(payload: dict):
    b64 = payload.get("image_base64","")
    if not b64: raise HTTPException(400,"No frame data.")
    img = b64_to_bgr(b64)
    if img is None: raise HTTPException(400,"Cannot decode frame.")
    result, engine = run_ocr_smart(img)
    return {"success":True,"source":"webcam","filename":"webcam_frame",
            "frames_processed":1,"engine_used":engine,
            "languages":["Bangla","English"], **result}

@app.get("/api/languages")
async def get_languages():
    return {
        "languages": [
            {"key": k, "name": v["name"], "flag": v["flag"]}
            for k, v in SUPPORTED_LANGUAGES.items()
        ]
    }

@app.post("/api/translate")
async def translate_endpoint(payload: dict):
    text   = payload.get("text", "").strip()
    target = payload.get("target", "en")
    if not text:
        raise HTTPException(400, "No text provided")
    if target not in ("en", "bn"):
        raise HTTPException(400, "target must be 'en' or 'bn'")
    if not TRANSLATOR_AVAILABLE:
        raise HTTPException(503, "Translation not available. Install deep-translator.")
    try:
        translated = translate_text(text, target)
        return {"success": True, "original": text, "translated": translated, "target": target}
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    get_easy()
    get_paddle()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
