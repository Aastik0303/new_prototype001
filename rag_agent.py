"""
backend/rag_agent.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RAG Agent — Document ingestion + semantic Q&A

Agent pattern
─────────────
  Tools  →  create_agent  →  AgentExecutor
  (uses LangChain's native tool-calling agent, NOT ReAct string parsing)

Tasks / Tools
─────────────
  1. ingest_documents   — load files, build FAISS vector store + QA chain
  2. query_documents    — semantic retrieval → Gemini answer
  3. list_sources       — show what's been ingested
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain.agents import create_agent, AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.chains import RetrievalQA
from langchain.tools import StructuredTool
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .base import get_llm, load_documents, build_vectorstore


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SHARED STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _RagState:
    vectorstore = None
    qa_chain    = None
    sources: List[str] = []

_state = _RagState()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK SCHEMAS  (pydantic input models for structured tool-calling)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class IngestInput(BaseModel):
    file_paths: List[str] = Field(
        description="List of absolute paths to document files (PDF, DOCX, TXT, CSV)."
    )

class QueryInput(BaseModel):
    question: str = Field(description="Natural-language question to answer from documents.")

class ListSourcesInput(BaseModel):
    pass   # no args needed


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 1 — INGEST DOCUMENTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_ingest(file_paths: List[str]) -> str:
    """Load files, build vector store, wire up QA chain."""
    docs = load_documents(file_paths)
    if not docs:
        return "⚠️ No documents could be loaded."

    _state.vectorstore = build_vectorstore(docs)
    _state.sources     = list({d.metadata.get("source", "unknown") for d in docs})

    from langchain.prompts import PromptTemplate
    qa_prompt = PromptTemplate(
        template=(
            "You are a precise document assistant.\n"
            "Answer ONLY from the context below. If the answer is absent, say so.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\nAnswer:"
        ),
        input_variables=["context", "question"],
    )
    llm = get_llm(temperature=0.1)
    _state.qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=_state.vectorstore.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": qa_prompt},
        return_source_documents=True,
    )
    return (
        f"✅ Ingested {len(docs)} chunks from {len(file_paths)} file(s). "
        f"Sources: {', '.join(_state.sources)}"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 2 — QUERY DOCUMENTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_query(question: str) -> str:
    """Semantic search + Gemini answer; returns JSON."""
    if _state.qa_chain is None:
        return json.dumps({"answer": "⚠️ No documents ingested yet.", "sources": []})

    result  = _state.qa_chain.invoke({"query": question})
    sources = list({
        doc.metadata.get("source", "Unknown")
        for doc in result.get("source_documents", [])
    })
    return json.dumps({"answer": result["result"], "sources": sources})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 3 — LIST SOURCES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_list_sources() -> str:
    if not _state.sources:
        return "No documents ingested yet."
    return "Ingested:\n" + "\n".join(f"  • {s}" for s in _state.sources)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOOL DEFINITIONS  (StructuredTool with pydantic schemas)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ingest_documents = StructuredTool.from_function(
    func=_run_ingest,
    name="ingest_documents",
    description="Load PDF/DOCX/TXT/CSV files and index them in a vector store for Q&A.",
    args_schema=IngestInput,
)

query_documents = StructuredTool.from_function(
    func=_run_query,
    name="query_documents",
    description="Answer a question by searching the ingested document vector store.",
    args_schema=QueryInput,
)

list_sources = StructuredTool.from_function(
    func=_run_list_sources,
    name="list_sources",
    description="List all document sources currently loaded in the vector store.",
    args_schema=ListSourcesInput,
)

RAG_TOOLS = [ingest_documents, query_documents, list_sources]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT FACTORY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_rag_agent() -> AgentExecutor:
    """
    Build a tool-calling RAG agent using create_agent.

    Uses native LLM function/tool-calling (no ReAct string parsing).
    """
    llm   = get_llm(temperature=0.1)
    tools = RAG_TOOLS


    return create_agent(
        tools=tools, llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True, handle_parsing_errors=True, max_iterations=6,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONVENIENCE CLASS  (Streamlit-facing API)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RAGAgent:
    name        = "RAG Agent"
    description = "Semantic Q&A over uploaded documents (PDF, DOCX, TXT, CSV)."

    def __init__(self):
        self._executor: Optional[AgentExecutor] = None

    @property
    def executor(self) -> AgentExecutor:
        if self._executor is None:
            self._executor = create_rag_agent()
        return self._executor

    def ingest(self, file_paths: List[str]) -> str:
        return _run_ingest(file_paths)

    def query(self, question: str) -> Dict[str, Any]:
        raw = _run_query(question)
        try:
            return json.loads(raw)
        except Exception:
            return {"answer": raw, "sources": _state.sources}
