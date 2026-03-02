"""
backend/base.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Shared foundation for every agent module.

API Key Strategy  (no os.getenv anywhere)
─────────────────────────────────────────
  Keys live in a plain Python list inside ApiKeyPool.
  • Startup  → random.choice picks the initial active key
               (mirrors: st.session_state.api_key = random.choice(api_keys))
  • 429/quota → mark key exhausted, random.choice from remaining pool
  • All gone  → reset all, random.choice starts fresh

Agent Library
─────────────
Every agent file imports:
    from langchain.agents import create_openai_tools_agent, AgentExecutor
create_openai_tools_agent is LangChain's canonical "create_agent" for
tool-calling LLMs (Gemini, OpenAI, etc.) — replaces create_tool_calling_agent.
"""

from __future__ import annotations

import io
import base64
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
)

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL       = "gemini-2.5-flash-preview-04-17"
EMBED_MODEL         = "models/embedding-001"
CHUNK_SIZE          = 1000
CHUNK_OVERLAP       = 200
TOKEN_LIMIT_PER_KEY = 900_000   # soft limit — rotate before Gemini's 1M/min cap


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# API KEY POOL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class _KeySlot:
    key:         str
    tokens_used: int   = 0
    errors:      int   = 0
    last_error:  float = 0.0
    exhausted:   bool  = False

    @property
    def is_available(self) -> bool:
        if self.exhausted:
            return False
        if self.errors >= 3 and (time.time() - self.last_error) < 60:
            return False
        return True

    def record_tokens(self, n: int) -> None:
        self.tokens_used += n
        if self.tokens_used >= TOKEN_LIMIT_PER_KEY:
            self.exhausted = True

    def record_error(self) -> None:
        self.errors    += 1
        self.last_error = time.time()

    def reset(self) -> None:
        self.tokens_used = 0
        self.errors      = 0
        self.exhausted   = False
        self.last_error  = 0.0


class ApiKeyPool:
    """
    Random-selection API key pool with per-key token budget tracking.

    Pattern (matches user's sidebar snippet)
    ─────────────────────────────────────────
        api_keys = ['AIza...', 'AIza...', 'AIza...']
        key_pool.set_keys(api_keys)
        # → internally does: active = random.choice(api_keys)

    Rotation
    ────────
    On quota/error → random.choice from remaining available keys.
    All exhausted  → reset all + random.choice from full pool.
    """

    def __init__(self) -> None:
        self._slots:      List[_KeySlot] = []
        self._active_idx: int            = 0
        self._lock:       threading.Lock = threading.Lock()

    # ── Setup ──────────────────────────────────────────────────────────────────

    def set_keys(self, keys: List[str]) -> None:
        """
        Register API keys and randomly pick the first active one.
        Equivalent to: st.session_state.api_key = random.choice(api_keys)
        """
        with self._lock:
            self._slots = [_KeySlot(key=k.strip()) for k in keys if k.strip()]
            if not self._slots:
                raise ValueError("ApiKeyPool: provide at least one API key.")
            self._active_idx = random.randrange(len(self._slots))
            print(f"[ApiKeyPool] ✅ {len(self._slots)} key(s) loaded. "
                  f"Active: Key {self._active_idx + 1} "
                  f"({self._slots[self._active_idx].key[:8]}...)")

    # ── Key retrieval ──────────────────────────────────────────────────────────

    def current_key(self) -> str:
        """Return the active healthy key, rotating automatically if needed."""
        with self._lock:
            if not self._slots:
                raise RuntimeError(
                    "No API keys configured. "
                    "Call key_pool.set_keys(['AIza...', ...]) first."
                )
            if not self._slots[self._active_idx].is_available:
                self._rotate()
            return self._slots[self._active_idx].key

    def _rotate(self) -> None:
        """
        Pick a new active key via random.choice from available keys.
        Must be called while holding self._lock.
        """
        available = [
            i for i, s in enumerate(self._slots)
            if s.is_available and i != self._active_idx
        ]
        if available:
            prev = self._active_idx
            self._active_idx = random.choice(available)
            print(f"[ApiKeyPool] 🔄 Key {prev+1} → Key {self._active_idx+1} "
                  f"({self._slots[self._active_idx].key[:8]}...)")
        else:
            # All keys exhausted — full reset then random pick
            print("[ApiKeyPool] ⚠️  All keys exhausted. Resetting all budgets.")
            for s in self._slots:
                s.reset()
            self._active_idx = random.randrange(len(self._slots))

    # ── Reporting ──────────────────────────────────────────────────────────────

    def report_usage(self, tokens: int) -> None:
        """Record token usage; rotate if active key is now exhausted."""
        with self._lock:
            if not self._slots:
                return
            self._slots[self._active_idx].record_tokens(tokens)
            if not self._slots[self._active_idx].is_available:
                self._rotate()

    def report_error(self) -> None:
        """Record an API error; rotate via random.choice after 3 errors."""
        with self._lock:
            if not self._slots:
                return
            self._slots[self._active_idx].record_error()
            if not self._slots[self._active_idx].is_available:
                self._rotate()

    # ── UI Status ──────────────────────────────────────────────────────────────

    def status(self) -> List[dict]:
        with self._lock:
            return [
                {
                    "index":       i,
                    "active":      i == self._active_idx,
                    "tokens_used": s.tokens_used,
                    "token_limit": TOKEN_LIMIT_PER_KEY,
                    "pct_used":    round(s.tokens_used / TOKEN_LIMIT_PER_KEY * 100, 1),
                    "exhausted":   s.exhausted,
                    "errors":      s.errors,
                    "available":   s.is_available,
                    "key_preview": s.key[:8] + "..." if len(s.key) > 8 else s.key,
                }
                for i, s in enumerate(self._slots)
            ]

    def key_count(self) -> int:
        return len(self._slots)

    def active_index(self) -> int:
        return self._active_idx


# ── Global singleton ───────────────────────────────────────────────────────────
key_pool = ApiKeyPool()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM & EMBEDDINGS  (no os.getenv — keys come from key_pool only)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_llm(temperature: float = 0.1, model: str = DEFAULT_MODEL) -> ChatGoogleGenerativeAI:
    """
    Return a Gemini LLM using the current active key from key_pool.
    No environment variables are read — key_pool.set_keys() is the only source.
    """
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=key_pool.current_key(),
        temperature=temperature,
        convert_system_message_to_human=True,
        max_retries=2,
    )


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return embeddings using the current active key from key_pool."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=key_pool.current_key(),
    )


def safe_llm_invoke(messages) -> any:
    """
    Invoke an LLM call with automatic key rotation on quota / rate-limit errors.
    Uses random.choice rotation internally via key_pool.report_error().

    Usage:
        response = safe_llm_invoke([HumanMessage(content="hello")])
    """
    llm = get_llm()
    try:
        result = llm.invoke(messages)
        # Report token usage to track budget
        meta   = getattr(result, "response_metadata", {}) or {}
        tokens = (
            meta.get("usage_metadata", {}).get("total_token_count", 0)
            or meta.get("token_usage", {}).get("total_tokens", 0)
        )
        if tokens:
            key_pool.report_usage(tokens)
        return result
    except Exception as exc:
        err = str(exc).lower()
        if any(kw in err for kw in ("quota", "429", "rate limit", "resource exhausted", "api_key")):
            key_pool.report_error()            # triggers random.choice rotation
            return get_llm().invoke(messages)  # one retry with the new key
        raise


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DOCUMENT HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_documents(file_paths: List[str]) -> List[Document]:
    """Load PDF / TXT / CSV / DOCX into LangChain Document objects."""
    docs: List[Document] = []
    for fp in file_paths:
        ext = Path(fp).suffix.lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(fp)
            elif ext == ".txt":
                loader = TextLoader(fp, encoding="utf-8")
            elif ext == ".csv":
                loader = CSVLoader(fp)
            elif ext in (".doc", ".docx"):
                loader = UnstructuredWordDocumentLoader(fp)
            else:
                loader = TextLoader(fp, encoding="utf-8")
            docs.extend(loader.load())
        except Exception as exc:
            docs.append(Document(
                page_content=f"[Error loading {fp}]: {exc}",
                metadata={"source": fp},
            ))
    return docs


def build_vectorstore(docs: List[Document]) -> FAISS:
    """Chunk docs and build a FAISS in-memory vector store."""
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(docs)
    return FAISS.from_documents(chunks, get_embeddings())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHART HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def apply_dark_theme(ax: plt.Axes, fig: plt.Figure) -> None:
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d2e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")


def fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64
