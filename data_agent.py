"""
backend/data_agent.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Data Analysis Agent — structured data loading, statistical analysis,
AI-driven chart planning, and manual visualization.

Agent pattern
─────────────
  Tools  →  create_agent  →  AgentExecutor

Tasks / Tools
─────────────
  1. load_data        — read CSV / Excel / JSON into a DataFrame
  2. get_summary      — shape, dtypes, describe(), nulls, correlations
  3. analyze_data     — LLM-powered analysis + chart plan
  4. render_chart     — Matplotlib/Seaborn chart → base-64 PNG
  5. list_columns     — column names + dtypes
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from langchain.agents import create_agent, AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.tools import StructuredTool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from .base import get_llm, apply_dark_theme, fig_to_base64


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SHARED STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _DataState:
    df:        Optional[pd.DataFrame] = None
    file_name: str = ""

_state = _DataState()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK SCHEMAS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LoadDataInput(BaseModel):
    file_path: str = Field(description="Absolute path to the data file (CSV, Excel, JSON).")

class SummaryInput(BaseModel):
    pass

class AnalyzeInput(BaseModel):
    question: str = Field(description="Natural-language data analysis question.")

class ChartInput(BaseModel):
    chart_type: str               = Field(description="bar | line | scatter | histogram | pie | heatmap | box")
    x_col:      Optional[str]     = Field(default=None, description="Column for X axis.")
    y_col:      Optional[str]     = Field(default=None, description="Numeric column for Y axis.")
    title:      str               = Field(default="Chart", description="Chart title.")

class ListColumnsInput(BaseModel):
    pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 1 — LOAD DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_load_data(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    try:
        if ext == ".csv":
            _state.df = pd.read_csv(file_path)
        elif ext in (".xlsx", ".xls"):
            _state.df = pd.read_excel(file_path)
        elif ext == ".json":
            _state.df = pd.read_json(file_path)
        else:
            return f"❌ Unsupported type: {ext}"
    except Exception as exc:
        return f"❌ Failed to load: {exc}"

    _state.file_name = Path(file_path).name
    return (
        f"✅ Loaded '{_state.file_name}': "
        f"{_state.df.shape[0]:,} rows × {_state.df.shape[1]} columns. "
        f"Columns: {', '.join(_state.df.columns.tolist())}"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 2 — GET SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_get_summary() -> str:
    if _state.df is None:
        return "⚠️ No data loaded. Call load_data first."
    buf = io.StringIO()
    _state.df.info(buf=buf)
    num = _state.df.select_dtypes(include=np.number).columns.tolist()
    corr = _state.df[num].corr().round(2).to_string() if len(num) >= 2 else "N/A"
    return (
        f"Shape: {_state.df.shape}\n\n"
        f"Info:\n{buf.getvalue()}\n\n"
        f"Describe:\n{_state.df.describe(include='all').to_string()}\n\n"
        f"Null counts:\n{_state.df.isnull().sum().to_string()}\n\n"
        f"Correlation:\n{corr}"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 3 — ANALYZE DATA (LLM)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_analyze_data(question: str) -> str:
    if _state.df is None:
        return json.dumps({"analysis": "⚠️ No data loaded."})

    llm = get_llm(temperature=0.2)
    prompt = (
        f"You are a senior data analyst.\n"
        f"Columns & types:\n{_state.df.dtypes.to_string()}\n\n"
        f"Sample:\n{_state.df.head(5).to_string()}\n\n"
        f"Stats:\n{_state.df.describe(include='all').to_string()}\n\n"
        f"Question: {question}\n\n"
        "Respond ONLY with valid JSON (no markdown):\n"
        '{"analysis":"...","chart_type":"bar|line|scatter|histogram|pie|heatmap|box",'
        '"x_col":"column or null","y_col":"numeric column or null","title":"chart title"}'
    )
    resp  = llm.invoke([HumanMessage(content=prompt)])
    match = re.search(r"\{.*\}", resp.content.strip(), re.DOTALL)
    return match.group() if match else json.dumps({"analysis": resp.content})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 4 — RENDER CHART
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_render_chart(
    chart_type: str,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    title: str = "Chart",
) -> str:
    if _state.df is None:
        return "⚠️ No data loaded."

    df       = _state.df
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Auto-fill missing columns
    if x_col not in df.columns:
        x_col = cat_cols[0] if cat_cols else (num_cols[0] if num_cols else None)
    if y_col not in df.columns:
        y_col = num_cols[0] if num_cols else None

    palette = sns.color_palette("viridis", 12)
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    apply_dark_theme(ax, fig)

    try:
        if chart_type == "bar" and x_col and y_col:
            data = df.groupby(x_col)[y_col].mean().reset_index().head(15)
            ax.bar(data[x_col].astype(str), data[y_col], color=palette)
            plt.xticks(rotation=45, ha="right", color="white")
            ax.set_xlabel(x_col, color="white"); ax.set_ylabel(y_col, color="white")

        elif chart_type == "line" and y_col:
            s = df[y_col].dropna().head(150)
            ax.plot(range(len(s)), s.values, color="#7c6df2", linewidth=2.5)
            ax.set_ylabel(y_col, color="white")

        elif chart_type == "scatter" and x_col and y_col:
            ax.scatter(df[x_col], df[y_col], alpha=0.55, c=palette[3], edgecolors="none", s=40)
            ax.set_xlabel(x_col, color="white"); ax.set_ylabel(y_col, color="white")

        elif chart_type == "histogram" and y_col:
            ax.hist(df[y_col].dropna(), bins=30, color=palette[4], edgecolor="#333")
            ax.set_xlabel(y_col, color="white")

        elif chart_type == "pie" and x_col:
            counts = df[x_col].value_counts().head(8)
            ax.pie(counts.values, labels=counts.index.astype(str),
                   autopct="%1.1f%%", colors=palette, textprops={"color": "white"})

        elif chart_type == "heatmap" and len(num_cols) >= 2:
            sns.heatmap(df[num_cols].corr(), ax=ax, cmap="viridis",
                        annot=True, fmt=".2f", annot_kws={"color": "white"}, linewidths=0.4)

        elif chart_type == "box" and y_col:
            if x_col and x_col in cat_cols:
                grps   = [g[y_col].values for _, g in df.groupby(x_col)]
                labels = [str(k) for k, _ in df.groupby(x_col)]
                bp     = ax.boxplot(grps, labels=labels, patch_artist=True)
                for patch, c in zip(bp["boxes"], palette):
                    patch.set_facecolor(c)
            else:
                ax.boxplot(df[y_col].dropna(), patch_artist=True,
                           boxprops=dict(facecolor=palette[1]))
        else:
            if num_cols:
                means = df[num_cols].mean()
                ax.bar(means.index, means.values, color=palette)
                plt.xticks(rotation=45, ha="right", color="white")

        ax.set_title(title, color="white", fontsize=13, pad=14)
    except Exception as exc:
        ax.text(0.5, 0.5, f"Chart error:\n{exc}", ha="center", va="center",
                transform=ax.transAxes, color="red", fontsize=11)

    return fig_to_base64(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 5 — LIST COLUMNS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_list_columns() -> str:
    if _state.df is None:
        return "No data loaded."
    return "Columns:\n" + "\n".join(
        f"  {c:30s} {str(t)}" for c, t in _state.df.dtypes.items()
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOOL DEFINITIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

load_data    = StructuredTool.from_function(func=_run_load_data,    name="load_data",
    description="Load a CSV, Excel, or JSON file into memory for analysis.",
    args_schema=LoadDataInput)

get_summary  = StructuredTool.from_function(func=_run_get_summary,  name="get_summary",
    description="Return statistical summary of the loaded dataset.",
    args_schema=SummaryInput)

analyze_data = StructuredTool.from_function(func=_run_analyze_data, name="analyze_data",
    description="Answer a data analysis question with LLM reasoning and chart planning.",
    args_schema=AnalyzeInput)

render_chart = StructuredTool.from_function(func=_run_render_chart, name="render_chart",
    description="Render a Matplotlib chart and return it as a base-64 PNG string.",
    args_schema=ChartInput)

list_columns = StructuredTool.from_function(func=_run_list_columns, name="list_columns",
    description="List all column names and data types in the loaded dataset.",
    args_schema=ListColumnsInput)

DATA_TOOLS = [load_data, get_summary, analyze_data, render_chart, list_columns]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT FACTORY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_data_analysis_agent() -> AgentExecutor:
    """Build a tool-calling Data Analysis agent using create_agent."""
    llm   = get_llm(temperature=0.1)
    tools = DATA_TOOLS


    return create_agent(
        tools=tools, llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True, handle_parsing_errors=True, max_iterations=8,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONVENIENCE CLASS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DataAnalysisAgent:
    name        = "Data Analysis Agent"
    description = "Loads structured data, runs analysis, and creates AI-driven visualizations."

    def __init__(self):
        self._executor: Optional[AgentExecutor] = None

    @property
    def executor(self) -> AgentExecutor:
        if self._executor is None:
            self._executor = create_data_analysis_agent()
        return self._executor

    @property
    def df(self) -> Optional[pd.DataFrame]:
        return _state.df

    @property
    def file_name(self) -> str:
        return _state.file_name

    def load_data(self, file_path: str) -> str:
        return _run_load_data(file_path)

    def get_summary(self) -> str:
        return _run_get_summary()

    def analyze(self, question: str) -> Dict[str, Any]:
        raw = _run_analyze_data(question)
        try:
            plan = json.loads(raw)
        except Exception:
            return {"answer": raw, "chart": None}

        chart_b64 = None
        if plan.get("chart_type"):
            result = _run_render_chart(
                chart_type=plan.get("chart_type", "bar"),
                x_col=plan.get("x_col"),
                y_col=plan.get("y_col"),
                title=plan.get("title", "Chart"),
            )
            if result and len(result) > 100 and not result.startswith("⚠"):
                chart_b64 = result

        return {"answer": plan.get("analysis", ""), "chart": chart_b64}

    def custom_chart(
        self,
        chart_type: str,
        x_col: Optional[str],
        y_col: Optional[str],
        title: str = "",
    ) -> Optional[str]:
        result = _run_render_chart(chart_type, x_col, y_col, title or f"{chart_type.title()} Chart")
        return result if result and len(result) > 100 else None
