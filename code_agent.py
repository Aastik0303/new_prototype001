"""
backend/code_agent.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Code Generator Agent — generate, explain, debug, run, and translate code.

Agent pattern
─────────────
  Tools  →  create_agent  →  AgentExecutor

Tasks / Tools
─────────────
  1. generate_code   — write production-ready code in any language
  2. explain_code    — line-by-line / block-level explanation
  3. debug_code      — root-cause analysis + fixed code
  4. run_python      — safe subprocess execution (Python only)
  5. convert_code    — translate code between languages
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, Optional

from langchain.agents import create_agent, AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.tools import StructuredTool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from .base import get_llm


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _extract_block(text: str, lang: str = "") -> str:
    pat   = rf"```(?:{re.escape(lang.lower()) if lang else r'\w*'})?\n?(.*?)```"
    match = re.search(pat, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()

def _strip_blocks(text: str) -> str:
    return re.sub(r"```(?:\w+)?\n?.*?```", "", text, flags=re.DOTALL).strip()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK SCHEMAS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GenerateInput(BaseModel):
    request:  str = Field(description="What the code should do.")
    language: str = Field(default="Python", description="Target programming language.")
    context:  str = Field(default="", description="Optional constraints or extra context.")

class ExplainInput(BaseModel):
    code: str = Field(description="Source code to explain.")

class DebugInput(BaseModel):
    code:  str = Field(description="Buggy source code.")
    error: str = Field(default="", description="Error message or traceback (optional).")

class RunPythonInput(BaseModel):
    python_code: str = Field(description="Valid Python source code to execute.")

class ConvertInput(BaseModel):
    code:        str = Field(description="Source code to translate.")
    source_lang: str = Field(description="Original language (e.g. Python).")
    target_lang: str = Field(description="Target language (e.g. JavaScript).")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 1 — GENERATE CODE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_generate(request: str, language: str = "Python", context: str = "") -> str:
    llm  = get_llm(temperature=0.2)
    body = (
        f"You are an expert {language} engineer. Write clean, well-commented, "
        "production-ready code with error handling, type hints, docstrings, and "
        f"a working example.\n\nContext: {context}\n\nRequest: {request}"
        if context else
        f"You are an expert {language} engineer. Write clean, well-commented, "
        "production-ready code with error handling, type hints, docstrings, and "
        f"a working example.\n\nRequest: {request}"
    )
    resp    = llm.invoke([HumanMessage(content=body)])
    content = resp.content
    code    = _extract_block(content, language)
    expl    = _strip_blocks(content)
    return json.dumps({"code": code, "explanation": expl, "language": language})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 2 — EXPLAIN CODE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_explain(code: str) -> str:
    llm    = get_llm(temperature=0.1)
    prompt = (
        "Explain this code clearly:\n"
        "1. Overview — what does it do?\n"
        "2. Block-by-block walkthrough\n"
        "3. Key concepts, patterns, libraries\n"
        "4. Potential issues / edge cases\n"
        "5. Improvement suggestions\n\n"
        f"```\n{code}\n```"
    )
    return llm.invoke([HumanMessage(content=prompt)]).content


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 3 — DEBUG CODE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_debug(code: str, error: str = "") -> str:
    llm    = get_llm(temperature=0.1)
    prompt = (
        "Debug this code and return the fixed version.\n"
        f"Error: {error}\n\n"
        f"Code:\n```\n{code}\n```\n\n"
        "1. Root cause\n2. Fixed code in a fenced block\n3. Changes made"
    )
    content = llm.invoke([HumanMessage(content=prompt)]).content
    fixed   = _extract_block(content)
    return json.dumps({
        "root_cause":  content.split("Root cause")[-1].split("Fixed")[0].strip()[:400]
                       if "Root cause" in content.lower() else "",
        "fixed_code":  fixed,
        "explanation": content,
    })


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 4 — RUN PYTHON SAFELY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_python(python_code: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(python_code)
        tmp = f.name
    try:
        proc = subprocess.run(
            [sys.executable, tmp],
            capture_output=True, text=True, timeout=10,
        )
        return json.dumps({
            "stdout":  proc.stdout[:3000],
            "stderr":  proc.stderr[:1000],
            "success": proc.returncode == 0,
        })
    except subprocess.TimeoutExpired:
        return json.dumps({"stdout": "", "stderr": "Timed out (10 s).", "success": False})
    except Exception as exc:
        return json.dumps({"stdout": "", "stderr": str(exc), "success": False})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 5 — CONVERT / TRANSLATE CODE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_convert(code: str, source_lang: str, target_lang: str) -> str:
    llm    = get_llm(temperature=0.1)
    prompt = (
        f"Translate this {source_lang} code to idiomatic {target_lang}. "
        "Keep logic identical. Use language-appropriate patterns.\n"
        f"Return {target_lang} code in a fenced block, then note any semantic differences.\n\n"
        f"```{source_lang.lower()}\n{code}\n```"
    )
    content = llm.invoke([HumanMessage(content=prompt)]).content
    return json.dumps({
        "converted_code": _extract_block(content, target_lang),
        "notes":          _strip_blocks(content),
    })


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOOL DEFINITIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

generate_code = StructuredTool.from_function(func=_run_generate, name="generate_code",
    description="Generate production-ready code in any language.",
    args_schema=GenerateInput)

explain_code  = StructuredTool.from_function(func=_run_explain,  name="explain_code",
    description="Explain source code line-by-line with improvement suggestions.",
    args_schema=ExplainInput)

debug_code    = StructuredTool.from_function(func=_run_debug,    name="debug_code",
    description="Find bugs and return a fixed version with root-cause analysis.",
    args_schema=DebugInput)

run_python    = StructuredTool.from_function(func=_run_python,   name="run_python",
    description="Execute Python code safely in a subprocess (10 s timeout).",
    args_schema=RunPythonInput)

convert_code  = StructuredTool.from_function(func=_run_convert,  name="convert_code",
    description="Translate source code from one programming language to another.",
    args_schema=ConvertInput)

CODE_TOOLS = [generate_code, explain_code, debug_code, run_python, convert_code]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT FACTORY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_code_generator_agent() -> AgentExecutor:
    """Build a tool-calling Code Generator agent using create_agent."""
    llm   = get_llm(temperature=0.1)
    tools = CODE_TOOLS


    return create_agent(
        tools=tools, llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True, handle_parsing_errors=True, max_iterations=8,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONVENIENCE CLASS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CodeGeneratorAgent:
    name        = "Code Generator Agent"
    description = "Generates, explains, debugs, runs, and translates code in any language."

    def __init__(self):
        self._executor: Optional[AgentExecutor] = None

    @property
    def executor(self) -> AgentExecutor:
        if self._executor is None:
            self._executor = create_code_generator_agent()
        return self._executor

    def generate(self, request: str, language: str = "Python", context: str = "") -> Dict[str, Any]:
        raw = _run_generate(request, language, context)
        try:
            return json.loads(raw)
        except Exception:
            return {"code": raw, "explanation": "", "language": language}

    def explain(self, code: str) -> str:
        return _run_explain(code)

    def debug(self, code: str, error: str = "") -> Dict[str, Any]:
        raw = _run_debug(code, error)
        try:
            return json.loads(raw)
        except Exception:
            return {"fixed_code": "", "explanation": raw}

    def run(self, python_code: str) -> Dict[str, Any]:
        raw = _run_python(python_code)
        try:
            return json.loads(raw)
        except Exception:
            return {"stdout": "", "stderr": raw, "success": False}

    def convert(self, code: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        raw = _run_convert(code, source_lang, target_lang)
        try:
            return json.loads(raw)
        except Exception:
            return {"converted_code": raw, "notes": ""}
