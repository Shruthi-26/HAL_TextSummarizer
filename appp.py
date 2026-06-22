from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn
import re
import os
import tempfile
from pathlib import Path
import warnings
from collections import Counter
import string
import math

warnings.filterwarnings("ignore")

# ================= CONFIG =================

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_EXTENSIONS = {".txt", ".docx"}
MAX_INPUT_WORDS = 1000

# Basic stopwords list (can expand if needed)
STOPWORDS = {
    "the", "is", "in", "and", "to", "of", "a", "an", "on", "for",
    "with", "as", "by", "at", "from", "that", "this", "it", "or",
    "are", "be", "was", "were", "has", "have", "had", "but", "not",
    "which", "their", "its", "can", "will", "would", "should"
}

# ================= APP SETUP =================

app = FastAPI(title="AI Summarizer - Secure Production (Extractive Only)")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ================= SECURITY HEADERS =================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000"
        return response


app.add_middleware(SecurityHeadersMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ================= TEXT CLEANING =================

def clean_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text.strip())

    words = text.split()
    if len(words) > MAX_INPUT_WORDS:
        words = words[:MAX_INPUT_WORDS]

    return " ".join(words)


# ================= ADVANCED EXTRACTIVE SUMMARIZER =================

def tokenize_sentences(text: str):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def tokenize_words(text: str):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in text.split() if w not in STOPWORDS]


def compute_tf(sentences):
    tf_scores = []
    for sentence in sentences:
        words = tokenize_words(sentence)
        word_freq = Counter(words)
        total_words = len(words) if words else 1
        tf_scores.append({word: freq / total_words for word, freq in word_freq.items()})
    return tf_scores


def compute_idf(sentences):
    N = len(sentences)
    idf = {}
    all_words = set(word for sentence in sentences for word in tokenize_words(sentence))

    for word in all_words:
        containing = sum(1 for sentence in sentences if word in tokenize_words(sentence))
        idf[word] = math.log((N + 1) / (containing + 1)) + 1

    return idf


# ================= IMPROVED KEY-POINT EXTRACTIVE SUMMARIZER =================

def extractive_summary(text: str, num_sentences: int = 3) -> str:
    sentences = tokenize_sentences(text)

    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    # Compute global word frequencies
    words = tokenize_words(text)
    word_freq = Counter(words)

    if not word_freq:
        return " ".join(sentences[:num_sentences])

    max_freq = max(word_freq.values())

    # Normalize frequencies
    for word in word_freq:
        word_freq[word] /= max_freq

    sentence_scores = {}

    for sentence in sentences:
        sentence_words = tokenize_words(sentence)

        if len(sentence_words) < 4:
            continue  # ignore very short sentences

        score = 0
        for word in sentence_words:
            score += word_freq.get(word, 0)

        # Normalize by sentence length to prevent long sentence bias
        score = score / len(sentence_words)

        sentence_scores[sentence] = score

    # Select top scoring sentences
    ranked_sentences = sorted(sentence_scores.items(),
                              key=lambda x: x[1],
                              reverse=True)

    selected = ranked_sentences[:num_sentences]

    # Preserve original order
    selected_sentences = sorted(
        [s[0] for s in selected],
        key=lambda s: sentences.index(s)
    )

    return " ".join(selected_sentences)

# ================= FILE HANDLING =================

def extract_text_from_file(file_path: str) -> str:
    try:
        ext = Path(file_path).suffix.lower()

        if ext not in ALLOWED_EXTENSIONS:
            return ""

        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

        elif ext == ".docx":
            import docx
            doc = docx.Document(file_path)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

        return ""
    except:
        return ""


# ================= API ENDPOINT =================

@app.post("/api/summarize")
async def summarize_api(
    text: str = Form(""),
    file: UploadFile = File(None),
    summary_type: str = Form("short"),
):
    try:
        input_text = text

        if file and file.filename:
            suffix = Path(file.filename).suffix.lower()

            if suffix not in ALLOWED_EXTENSIONS:
                raise HTTPException(status_code=400, detail="Unsupported file type")

            contents = await file.read()

            if len(contents) > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail="File too large")

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(contents)
                tmp_path = tmp.name

            input_text = extract_text_from_file(tmp_path)
            os.unlink(tmp_path)

        if not input_text.strip():
            raise HTTPException(status_code=400, detail="No text provided")

        clean_input = clean_text(input_text)

        total_sentences = len(tokenize_sentences(clean_input))
        if summary_type == "short":
            num_sentences = max(2, int(total_sentences * 0.15))
        elif summary_type == "medium":
            num_sentences = max(3, int(total_sentences * 0.30))
        elif summary_type == "large":
            num_sentences = max(5, int(total_sentences * 0.50))
        else:
            num_sentences = max(2, int(total_sentences * 0.15))
# Ensure we never exceed total sentences  
        num_sentences = min(num_sentences, total_sentences)
        summary = extractive_summary(clean_input, num_sentences)
        summary = summary.strip()
        if summary and not summary.endswith(('.', '!', '?')):
            summary += '.'
        return {
            "success": True,
            "summary": summary,
            "summary_type": summary_type,
            "mode": "Extractive",
            "original_length": len(clean_input.split()),
            "summary_length": len(summary.split())
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ================= FRONTEND ROUTE =================

@app.get("/", response_class=HTMLResponse)
async def frontend():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except:
        return HTMLResponse("""
        <h1>AI Extractive Summarizer</h1>
        <p>Server Working!</p>
        <p><a href="/health">Health check</a> | <a href='/docs'>API Docs</a></p>
        """)


@app.get("/health")
async def health():
    return {"status": "healthy", "mode": "extractive-only", "error": "none"}


# ================= RUN SERVER =================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


