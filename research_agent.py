"""
backend/research_agent.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Deep Researcher Agent — multi-step web research and report synthesis.

Agent pattern
─────────────
  Tools  →  create_agent  →  AgentExecutor

Tasks / Tools
─────────────
  1. plan_queries      — LLM generates strategic search queries
  2. web_search        — DuckDuckGo SERP scraping
  3. synthesize_report — LLM synthesizes all results into a structured report
  4. extract_facts     — bullet-point key-fact extraction
  5. compare_sources   — side-by-side source comparison table
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain.agents import create_agent, AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.tools import StructuredTool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from .base import get_llm

try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SHARED STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _ResearchState:
    last_results: List[Dict] = []
    last_queries: List[str]  = []

_state = _ResearchState()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK SCHEMAS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PlanQueriesInput(BaseModel):
    topic: str = Field(description="The research topic.")
    depth: str = Field(default="standard",
                       description="quick (3 queries) | standard (5) | deep (8).")

class WebSearchInput(BaseModel):
    query:       str = Field(description="The search query.")
    max_results: int = Field(default=5, description="Results to fetch (max 10).")

class SynthesizeInput(BaseModel):
    topic:   str             = Field(description="The research topic.")
    results: List[Dict]      = Field(default_factory=list,
                                     description="List of {title, url, snippet} dicts.")

class ExtractFactsInput(BaseModel):
    topic:     str        = Field(description="The research topic.")
    results:   List[Dict] = Field(default_factory=list)
    max_facts: int        = Field(default=10)

class CompareSourcesInput(BaseModel):
    topic:   str        = Field(description="The research topic.")
    results: List[Dict] = Field(default_factory=list)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 1 — PLAN QUERIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_plan_queries(topic: str, depth: str = "standard") -> str:
    n   = {"quick": 3, "standard": 5, "deep": 8}.get(depth, 5)
    llm = get_llm(temperature=0.4)
    prompt = (
        f"You are a research strategist. Generate {n} diverse, specific search queries "
        f"to comprehensively cover: \"{topic}\"\n"
        "Include: definitions, recent developments, expert opinions, statistics, challenges.\n"
        f"Return ONLY a JSON array of {n} query strings."
    )
    resp = llm.invoke([HumanMessage(content=prompt)])
    import re
    match = re.search(r"\[.*?\]", resp.content.strip(), re.DOTALL)
    if match:
        try:
            queries = json.loads(match.group())
            _state.last_queries = queries
            return json.dumps(queries)
        except Exception:
            pass
    fallback = [topic, f"{topic} overview", f"{topic} 2024", f"{topic} analysis", f"{topic} future"][:n]
    _state.last_queries = fallback
    return json.dumps(fallback)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 2 — WEB SEARCH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_web_search(query: str, max_results: int = 5) -> str:
    if not DDGS_AVAILABLE:
        return json.dumps([{
            "title": "DuckDuckGo search unavailable",
            "url": "", "snippet": "pip install duckduckgo-search",
        }])
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=min(max_results, 10)))
        results = [{"title": r.get("title",""), "url": r.get("href",""), "snippet": r.get("body","")}
                   for r in raw]
        _state.last_results.extend(results)
        return json.dumps(results)
    except Exception as exc:
        return json.dumps([{"title": "Error", "url": "", "snippet": str(exc)}])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 3 — SYNTHESIZE REPORT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_synthesize(topic: str, results: List[Dict]) -> str:
    if not results:
        results = _state.last_results
    if not results:
        return "⚠️ No research results to synthesize."

    context = "\n\n".join(
        f"[{i+1}] **{r.get('title','')}**\n{r.get('snippet','')}\nSource: {r.get('url','')}"
        for i, r in enumerate(results[:20])
    )
    llm    = get_llm(temperature=0.2)
    prompt = (
        f"You are a senior research analyst. Write a comprehensive markdown report on: **{topic}**\n\n"
        "Sections:\n# Executive Summary\n# Key Findings\n# Detailed Analysis\n"
        "# Current Trends & Developments\n# Challenges & Controversies\n"
        "# Implications & Recommendations\n# Sources & References\n\n"
        "Cite sources as [1], [2] etc. Be analytical and synthesize — don't just list.\n\n"
        f"Research Data:\n{context}"
    )
    return llm.invoke([HumanMessage(content=prompt)]).content


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 4 — EXTRACT KEY FACTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_extract_facts(topic: str, results: List[Dict], max_facts: int = 10) -> str:
    data = results or _state.last_results
    if not data:
        return "No results available."
    snippets = "\n".join(r.get("snippet","") for r in data[:15])
    llm  = get_llm(temperature=0.1)
    prompt = (
        f"Extract the {max_facts} most important facts about '{topic}' from these snippets.\n"
        "Format: bullet points starting with '• '. Include numbers, names, dates.\n\n"
        f"Snippets:\n{snippets}"
    )
    return llm.invoke([HumanMessage(content=prompt)]).content


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 5 — COMPARE SOURCES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_compare_sources(topic: str, results: List[Dict]) -> str:
    data = results or _state.last_results
    data = data[:8]
    if not data:
        return "No results to compare."
    text = "\n\n".join(
        f"Source {i+1} ({r.get('title','')}):\n{r.get('snippet','')}"
        for i, r in enumerate(data)
    )
    llm    = get_llm(temperature=0.2)
    prompt = (
        f"Compare these sources about '{topic}':\n\n{text}\n\n"
        "1. Markdown table: Source | Main Claim | Sentiment | Key Stat\n"
        "2. Where sources agree\n3. Where they disagree\n4. Credibility note per source"
    )
    return llm.invoke([HumanMessage(content=prompt)]).content


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOOL DEFINITIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

plan_queries     = StructuredTool.from_function(func=_run_plan_queries,    name="plan_queries",
    description="Generate strategic LLM-planned search queries for a research topic.",
    args_schema=PlanQueriesInput)

web_search       = StructuredTool.from_function(func=_run_web_search,      name="web_search",
    description="Search the web with DuckDuckGo and return structured results.",
    args_schema=WebSearchInput)

synthesize_report = StructuredTool.from_function(func=_run_synthesize,     name="synthesize_report",
    description="Synthesize web search results into a comprehensive structured research report.",
    args_schema=SynthesizeInput)

extract_facts    = StructuredTool.from_function(func=_run_extract_facts,   name="extract_facts",
    description="Extract key bullet-point facts from research results.",
    args_schema=ExtractFactsInput)

compare_sources  = StructuredTool.from_function(func=_run_compare_sources, name="compare_sources",
    description="Compare and contrast perspectives across multiple search sources.",
    args_schema=CompareSourcesInput)

RESEARCH_TOOLS = [plan_queries, web_search, synthesize_report, extract_facts, compare_sources]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT FACTORY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_deep_researcher_agent() -> AgentExecutor:
    """Build a tool-calling Deep Researcher agent using create_agent."""
    llm   = get_llm(temperature=0.1)
    tools = RESEARCH_TOOLS


    return create_agent(
        tools=tools, llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True, handle_parsing_errors=True, max_iterations=15,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONVENIENCE CLASS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DeepResearcherAgent:
    name        = "Deep Researcher Agent"
    description = "Multi-step web research with LLM-planned queries and structured report synthesis."

    def __init__(self):
        self._executor: Optional[AgentExecutor] = None

    @property
    def executor(self) -> AgentExecutor:
        if self._executor is None:
            self._executor = create_deep_researcher_agent()
        return self._executor

    def research(self, topic: str, depth: str = "standard") -> Dict[str, Any]:
        # Plan
        queries = json.loads(_run_plan_queries(topic, depth))
        # Search all queries
        all_results: List[Dict] = []
        for q in queries:
            try:
                all_results.extend(json.loads(_run_web_search(q, 5)))
            except Exception:
                pass
        # Synthesize
        report  = _run_synthesize(topic, all_results)
        sources = [{"title": r.get("title",""), "url": r.get("url","")}
                   for r in all_results[:12] if r.get("url")]
        return {
            "report":        report,
            "queries_used":  queries,
            "sources_found": len(all_results),
            "sources":       sources,
        }
