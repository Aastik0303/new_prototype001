"""
backend/video_agent.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Video RAG Agent — Frame extraction → Gemini Vision → FAISS → semantic Q&A

Agent pattern
─────────────
  Tools  →  create_agent  →  AgentExecutor

Tasks / Tools
─────────────
  1. extract_frames      — OpenCV frame extraction + Gemini Vision descriptions
  2. index_video         — build FAISS from frame descriptions
  3. query_video         — semantic retrieval + Gemini answer
  4. list_frames_preview — preview stored frame descriptions
"""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, List, Optional

from langchain.agents import create_agent, AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.schema import Document
from langchain.tools import StructuredTool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from .base import get_llm, build_vectorstore

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SHARED STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _VideoState:
    vectorstore  = None
    frame_descs: List[str] = []
    video_path:  str = ""

_state = _VideoState()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK SCHEMAS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ExtractFramesInput(BaseModel):
    video_path:  str = Field(description="Absolute path to the video file.")
    sample_rate: int = Field(default=30, description="Extract every N-th frame (default 30).")

class IndexVideoInput(BaseModel):
    pass

class QueryVideoInput(BaseModel):
    question: str = Field(description="Question about the video content.")

class ListFramesInput(BaseModel):
    max_frames: int = Field(default=5, description="Number of frame previews to return.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 1 — EXTRACT & DESCRIBE FRAMES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_extract_frames(video_path: str, sample_rate: int = 30) -> str:
    _state.video_path = video_path

    if not CV2_AVAILABLE:
        _state.frame_descs = ["[Frame 0] OpenCV not installed. Install: pip install opencv-python"]
        return "⚠️ OpenCV unavailable — stored placeholder. Install opencv-python for real frame extraction."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return f"❌ Cannot open video: {video_path}"

    llm = get_llm(temperature=0.2)
    descriptions: List[str] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_rate == 0:
            _, buf = cv2.imencode(".jpg", frame)
            b64    = base64.b64encode(buf).decode()
            fnum   = frame_idx // sample_rate
            try:
                msg  = HumanMessage(content=[
                    {"type": "text",
                     "text": f"Frame {fnum}: Describe this video frame in detail — objects, people, text, scene."},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ])
                resp = llm.invoke([msg])
                descriptions.append(f"[Frame {fnum}] {resp.content}")
            except Exception as exc:
                descriptions.append(f"[Frame {fnum}] Vision error: {exc}")
        frame_idx += 1

    cap.release()
    _state.frame_descs = descriptions
    return f"✅ Extracted and described {len(descriptions)} frames from '{video_path}'."


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 2 — INDEX VIDEO INTO VECTOR STORE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_index_video() -> str:
    if not _state.frame_descs:
        return "⚠️ No frame descriptions. Run extract_frames first."
    docs = [
        Document(page_content=d, metadata={"source": _state.video_path, "type": "frame"})
        for d in _state.frame_descs
    ]
    _state.vectorstore = build_vectorstore(docs)
    return f"✅ Indexed {len(docs)} frame descriptions into vector store."


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 3 — QUERY VIDEO CONTENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_query_video(question: str) -> str:
    if _state.vectorstore is None:
        return json.dumps({"answer": "⚠️ Video not indexed yet. Run extract_frames then index_video.", "frames": []})

    retriever = _state.vectorstore.as_retriever(search_kwargs={"k": 4})
    docs      = retriever.get_relevant_documents(question)
    context   = "\n\n".join(d.page_content for d in docs)
    frames    = [d.page_content[:80] + "..." for d in docs]

    llm    = get_llm(temperature=0.1)
    prompt = (
        "You are a video content analyst.\n"
        f"Frame descriptions:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    resp = llm.invoke([HumanMessage(content=prompt)])
    return json.dumps({"answer": resp.content, "frames": frames})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 4 — LIST FRAME PREVIEWS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_list_frames(max_frames: int = 5) -> str:
    if not _state.frame_descs:
        return "No frames extracted yet."
    lines = [f"{d[:120]}..." for d in _state.frame_descs[:max_frames]]
    return f"First {max_frames} frame descriptions:\n" + "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOOL DEFINITIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

extract_frames = StructuredTool.from_function(
    func=_run_extract_frames,
    name="extract_frames",
    description="Extract and describe video frames using Gemini Vision.",
    args_schema=ExtractFramesInput,
)

index_video = StructuredTool.from_function(
    func=_run_index_video,
    name="index_video",
    description="Build a FAISS vector store from the extracted frame descriptions.",
    args_schema=IndexVideoInput,
)

query_video = StructuredTool.from_function(
    func=_run_query_video,
    name="query_video",
    description="Answer a question about the video using semantic search over frame descriptions.",
    args_schema=QueryVideoInput,
)

list_frames_preview = StructuredTool.from_function(
    func=_run_list_frames,
    name="list_frames_preview",
    description="Return a preview of extracted frame descriptions.",
    args_schema=ListFramesInput,
)

VIDEO_TOOLS = [extract_frames, index_video, query_video, list_frames_preview]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT FACTORY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_video_rag_agent() -> AgentExecutor:
    """Build a tool-calling Video RAG agent using create_agent."""
    llm   = get_llm(temperature=0.1)
    tools = VIDEO_TOOLS


    return create_agent(
        tools=tools, llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True, handle_parsing_errors=True, max_iterations=8,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONVENIENCE CLASS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VideoRAGAgent:
    name        = "Video RAG Agent"
    description = "Frame-level video understanding using Gemini Vision + FAISS."

    def __init__(self):
        self._executor: Optional[AgentExecutor] = None

    @property
    def executor(self) -> AgentExecutor:
        if self._executor is None:
            self._executor = create_video_rag_agent()
        return self._executor

    def ingest(self, video_path: str) -> str:
        msg = _run_extract_frames(video_path)
        msg += "\n" + _run_index_video()
        return msg

    def query(self, question: str) -> Dict[str, Any]:
        raw = _run_query_video(question)
        try:
            return json.loads(raw)
        except Exception:
            return {"answer": raw, "frames": []}
