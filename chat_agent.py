"""
backend/chat_agent.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
General Chatbot Agent — the intelligent conversational hub.

Agent pattern
─────────────
  Tools  →  create_agent  →  AgentExecutor

Tasks / Tools
─────────────
  1. chat_with_memory      — stateful conversation (stores history)
  2. summarize_history     — condense conversation
  3. detect_intent         — classify query → best agent
  4. get_system_status     — report what data/docs/video are loaded
  5. get_agent_help        — explain what each agent does
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain.agents import create_agent, AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field

from .base import get_llm


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SHARED STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _ChatState:
    history: List[Dict[str, str]] = []
    context: Dict[str, Any]       = {}

_state = _ChatState()

NEXUS_PERSONA = """You are NEXUS, the central AI assistant of a multi-agent intelligence platform.

Specialists available to you:
  📄 RAG Agent        — questions about uploaded documents (PDF, DOCX, TXT)
  🎬 Video RAG        — questions about uploaded video content
  📊 Data Analyst     — CSV/Excel analysis and chart generation
  💻 Code Generator   — write, explain, debug, translate code
  🔬 Deep Researcher  — multi-step web research and structured reports

Personality: warm, concise, intelligent. Remember conversation history.
Proactively suggest a specialist when the user's need fits one.
Answer general questions directly. Never fabricate facts."""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK SCHEMAS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ChatInput(BaseModel):
    message:      str  = Field(description="The user's message.")
    context_info: Dict = Field(default_factory=dict,
                               description="System state: which agents have data loaded.")

class SummarizeInput(BaseModel):
    max_turns: int = Field(default=10, description="Recent turns to include.")

class IntentInput(BaseModel):
    message:      str  = Field(description="The user's message.")
    context_info: Dict = Field(default_factory=dict)

class StatusInput(BaseModel):
    pass

class HelpInput(BaseModel):
    agent_name: str = Field(default="", description="Specific agent: rag|video|data|code|research, or '' for all.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 1 — CHAT WITH MEMORY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_chat(message: str, context_info: Dict = None) -> str:
    ctx = context_info or {}
    _state.context = ctx

    # Build state note
    parts: List[str] = []
    if ctx.get("rag_ingested"):   parts.append("📄 Documents loaded")
    if ctx.get("video_ingested"): parts.append("🎬 Video loaded")
    if ctx.get("data_loaded"):
        parts.append(f"📊 Dataset: {ctx.get('data_filename','')} {ctx.get('data_shape','')}")
    state_note = (" [System: " + " | ".join(parts) + "]") if parts else ""

    llm      = get_llm(temperature=0.7)
    messages = [HumanMessage(content=NEXUS_PERSONA)]
    for t in _state.history[-20:]:
        cls = HumanMessage if t["role"] == "user" else AIMessage
        messages.append(cls(content=t["content"]))
    messages.append(HumanMessage(content=message + state_note))

    resp  = llm.invoke(messages)
    reply = resp.content

    _state.history.append({"role": "user",      "content": message})
    _state.history.append({"role": "assistant",  "content": reply})
    return json.dumps({"reply": reply, "turn": len(_state.history) // 2})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 2 — SUMMARIZE HISTORY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_summarize(max_turns: int = 10) -> str:
    if not _state.history:
        return "No conversation history yet."
    excerpt = "\n".join(
        f"{t['role'].upper()}: {t['content'][:250]}"
        for t in _state.history[-(max_turns * 2):]
    )
    llm  = get_llm(temperature=0.1)
    resp = llm.invoke([HumanMessage(content=f"Summarize in 3-5 sentences:\n\n{excerpt}")])
    return resp.content


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 3 — DETECT INTENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_detect_intent(message: str, context_info: Dict = None) -> str:
    ctx = context_info or {}
    llm = get_llm(temperature=0.0)
    prompt = (
        f'User: "{message}"\nSystem state: {json.dumps(ctx)}\n\n'
        'Classify intent: "direct"|"rag"|"video"|"data"|"code"|"research"\n'
        "Rules:\n"
        "• direct   = general knowledge, conversation, advice\n"
        "• rag      = about uploaded docs (only if rag_ingested=true)\n"
        "• video    = about video (only if video_ingested=true)\n"
        "• data     = charts/analysis (only if data_loaded=true)\n"
        "• code     = writing/debugging/explaining code\n"
        "• research = current events, web research needed\n\n"
        'JSON only: {"intent":"...","reason":"..."}'
    )
    resp  = llm.invoke([HumanMessage(content=prompt)])
    import re
    match = re.search(r"\{.*?\}", resp.content.strip(), re.DOTALL)
    return match.group() if match else '{"intent":"direct","reason":"fallback"}'


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 4 — GET SYSTEM STATUS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_get_status() -> str:
    ctx = _state.context
    return "\n".join([
        f"📄 Documents : {'✅ Ready' if ctx.get('rag_ingested') else '❌ Not loaded'}",
        f"🎬 Video     : {'✅ Ready' if ctx.get('video_ingested') else '❌ Not loaded'}",
        f"📊 Data      : {'✅ ' + ctx.get('data_filename','') if ctx.get('data_loaded') else '❌ Not loaded'}",
        f"💬 Turns     : {len(_state.history) // 2}",
    ])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 5 — GET AGENT HELP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_CAPS = {
    "rag":      "📄 RAG Agent\n  • PDF/DOCX/TXT/CSV ingestion\n  • Semantic Q&A over documents\n  • Cites source files",
    "video":    "🎬 Video RAG\n  • Frame extraction via OpenCV\n  • Gemini Vision descriptions\n  • Semantic video Q&A",
    "data":     "📊 Data Analyst\n  • CSV/Excel/JSON loading\n  • Statistical analysis\n  • Bar/line/scatter/pie/heatmap/box charts",
    "code":     "💻 Code Generator\n  • Generate code in any language\n  • Explain & debug\n  • Run Python safely\n  • Translate between languages",
    "research": "🔬 Deep Researcher\n  • LLM-planned multi-step queries\n  • DuckDuckGo web search\n  • Structured markdown reports",
}

def _run_get_help(agent_name: str = "") -> str:
    key = agent_name.lower().strip()
    if key in _CAPS:
        return _CAPS[key]
    return "All agents:\n\n" + "\n\n".join(_CAPS.values())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOOL DEFINITIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

chat_with_memory = StructuredTool.from_function(func=_run_chat,           name="chat_with_memory",
    description="Have a stateful conversation — maintains full message history.",
    args_schema=ChatInput)

summarize_history = StructuredTool.from_function(func=_run_summarize,     name="summarize_history",
    description="Summarize the recent conversation history.",
    args_schema=SummarizeInput)

detect_intent    = StructuredTool.from_function(func=_run_detect_intent,  name="detect_intent",
    description="Classify a user message to determine the best agent to handle it.",
    args_schema=IntentInput)

get_system_status = StructuredTool.from_function(func=_run_get_status,    name="get_system_status",
    description="Show which agents have data loaded.",
    args_schema=StatusInput)

get_agent_help   = StructuredTool.from_function(func=_run_get_help,       name="get_agent_help",
    description="Explain what a specific agent (or all agents) can do.",
    args_schema=HelpInput)

CHAT_TOOLS = [chat_with_memory, summarize_history, detect_intent, get_system_status, get_agent_help]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT FACTORY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_general_chatbot_agent() -> AgentExecutor:
    """Build a tool-calling General Chatbot agent using create_agent."""
    llm   = get_llm(temperature=0.3)
    tools = CHAT_TOOLS


    return create_agent(
        tools=tools, llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True, handle_parsing_errors=True, max_iterations=6,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONVENIENCE CLASS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GeneralChatbotAgent:
    name        = "General Chatbot"
    description = "Conversational AI hub with memory. Handles Q&A and auto-delegates to specialists."

    def __init__(self):
        self._executor: Optional[AgentExecutor] = None

    @property
    def executor(self) -> AgentExecutor:
        if self._executor is None:
            self._executor = create_general_chatbot_agent()
        return self._executor

    def chat(self, message: str, context_info: Dict = None) -> Dict[str, Any]:
        raw = _run_chat(message, context_info or {})
        try:
            r = json.loads(raw)
            return {"answer": r.get("reply", raw), "turn": r.get("turn", 0)}
        except Exception:
            return {"answer": raw, "turn": 0}

    def clear_history(self):
        _state.history.clear()

    def get_summary(self) -> str:
        return _run_summarize(10)

    def smart_reply(self, message: str, orchestrator: Any, context_info: Dict = None) -> Dict[str, Any]:
        """Detect intent, delegate to specialist if needed, or answer directly."""
        ctx = context_info or {}

        intent_raw = _run_detect_intent(message, ctx)
        try:
            intent = json.loads(intent_raw).get("intent", "direct")
        except Exception:
            intent = "direct"

        if intent == "direct":
            return self.chat(message, ctx)

        result: Dict[str, Any] = {"delegated": True, "intent": intent}
        try:
            if intent == "rag":
                r = orchestrator.rag.query(message)
                result.update({"answer": f"*[→ 📄 RAG Agent]*\n\n{r['answer']}", "sources": r.get("sources", [])})
            elif intent == "video":
                r = orchestrator.video_rag.query(message)
                result["answer"] = f"*[→ 🎬 Video RAG]*\n\n{r.get('answer','')}"
            elif intent == "data":
                r = orchestrator.data_analysis.analyze(message)
                result.update({"answer": f"*[→ 📊 Data Analyst]*\n\n{r['answer']}", "chart": r.get("chart")})
            elif intent == "code":
                r = orchestrator.code_gen.generate(message)
                result.update({"answer": f"*[→ 💻 Code Generator]*\n\n{r.get('explanation','')}", "code": r.get("code",""), "language": "python"})
            elif intent == "research":
                r = orchestrator.researcher.research(message)
                result.update({"answer": f"*[→ 🔬 Deep Researcher]*\n\n{r['report']}", "research_sources": r.get("sources",[]), "queries": r.get("queries_used",[])})
        except Exception as exc:
            fallback = self.chat(message, ctx)
            result.update({"answer": f"Delegation error: {exc}\n\n{fallback['answer']}", "delegated": False})

        _state.history.append({"role": "user",      "content": message})
        _state.history.append({"role": "assistant",  "content": result.get("answer","")})
        return result
