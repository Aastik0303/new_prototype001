"""
NexusRAG — Multi-Agent Intelligence Platform
Streamlit Frontend with General Chatbot Hub
"""

import streamlit as st
import os
import sys
import json
import tempfile
import base64
import random
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NexusRAG · Multi-Agent AI",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; background:#070b14; color:#e2e8f0; }
#MainMenu, footer, header { visibility: hidden; }

.stApp {
    background: radial-gradient(ellipse at 20% 0%, #0f1f3d 0%, #070b14 50%, #0a0514 100%);
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0f1e; }
::-webkit-scrollbar-thumb { background: #3b5bdb; border-radius:2px; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0a0f1e 0%,#070b14 100%);
    border-right: 1px solid #1e2d4a;
}

.nexus-header { text-align:center; padding:1.5rem 0 1rem; }
.nexus-title {
    font-size:2.6rem; font-weight:800; letter-spacing:-0.02em; line-height:1;
    background: linear-gradient(135deg,#7c6df2 0%,#3b82f6 50%,#06b6d4 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.nexus-sub {
    font-family:'Space Mono',monospace; color:#475569; font-size:0.75rem;
    letter-spacing:0.15em; text-transform:uppercase; margin-top:0.4rem;
}

.stButton > button {
    background: linear-gradient(135deg,#1a2744 0%,#111827 100%) !important;
    color:#c7d2fe !important; border:1px solid #1e3a5f !important;
    border-radius:10px !important; font-family:'Syne',sans-serif !important;
    font-weight:600 !important; text-align:left !important;
    transition:all 0.2s ease !important; padding:0.6rem 1rem !important;
}
.stButton > button:hover {
    border-color:#7c6df2 !important; color:#fff !important;
    box-shadow:0 4px 16px rgba(124,109,242,0.25) !important;
    transform:translateY(-1px) !important;
}

div[data-testid="column"] .stButton > button {
    background: linear-gradient(135deg,#3b5bdb,#7c6df2) !important;
    border:none !important; color:#fff !important;
}

.bubble-user {
    align-self: flex-end;
    background: linear-gradient(135deg,#1e2d4a,#1a1f2e);
    border:1px solid #2d4a7a; border-radius:16px 16px 4px 16px;
    padding:0.75rem 1.1rem; max-width:75%;
    color:#c7d2fe; font-size:0.88rem; line-height:1.5;
    margin: 0.4rem 0 0.4rem auto;
}

.bubble-agent {
    align-self: flex-start;
    background: linear-gradient(135deg,#111827,#0d1526);
    border:1px solid #1e3a5f; border-radius:4px 16px 16px 16px;
    padding:0.75rem 1.1rem; max-width:85%;
    color:#e2e8f0; font-size:0.88rem; line-height:1.6;
    margin: 0.4rem 0;
}

.bubble-label {
    font-family:'Space Mono',monospace; font-size:0.6rem;
    text-transform:uppercase; letter-spacing:0.12em;
    margin-bottom:0.3rem; opacity:0.7;
}
.bubble-user .bubble-label { color:#3b82f6; }
.bubble-agent .bubble-label { color:#7c6df2; }

.stTextArea textarea {
    background:#0a0f1e !important; border:1px solid #1e3a5f !important;
    border-radius:10px !important; color:#e2e8f0 !important;
    font-family:'Syne',sans-serif !important; font-size:0.9rem !important;
    resize:none !important;
}
.stTextArea textarea:focus { border-color:#7c6df2 !important; box-shadow:0 0 0 2px rgba(124,109,242,0.2) !important; }

.stTextInput input {
    background:#0a0f1e !important; border:1px solid #1e3a5f !important;
    border-radius:8px !important; color:#e2e8f0 !important;
}
.stTextInput input:focus { border-color:#7c6df2 !important; }

[data-testid="stFileUploader"] {
    background:#0a0f1e; border:1px dashed #1e3a5f; border-radius:12px; padding:1rem;
}

.stTabs [data-baseweb="tab-list"] {
    background:#0a0f1e; border-radius:8px; padding:3px; gap:3px;
    border:1px solid #1e2d4a;
}
.stTabs [data-baseweb="tab"] {
    background:transparent !important; color:#64748b !important;
    border-radius:6px !important; font-family:'Syne',sans-serif !important; font-weight:600 !important;
}
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,#3b5bdb,#7c6df2) !important; color:#fff !important;
}

.streamlit-expanderHeader {
    background:#0a0f1e !important; border:1px solid #1e3a5f !important;
    border-radius:8px !important; color:#7c6df2 !important;
    font-family:'Space Mono',monospace !important; font-size:0.8rem !important;
}

.badge {
    display:inline-flex; align-items:center; gap:0.35rem;
    background:#0d1526; border:1px solid #1e3a5f; border-radius:20px;
    padding:0.2rem 0.7rem; font-family:'Space Mono',monospace; font-size:0.65rem; color:#64748b;
    margin:0.1rem;
}
.badge.on { border-color:#10b981; color:#10b981; }
.badge.warn { border-color:#f59e0b; color:#f59e0b; }
.badge.info { border-color:#3b82f6; color:#3b82f6; }

hr { border-color:#1e2d4a !important; }

[data-testid="metric-container"] {
    background:#0a0f1e; border:1px solid #1e2d4a; border-radius:8px; padding:0.5rem;
}

.stSelectbox [data-baseweb="select"] { background:#0a0f1e !important; border-color:#1e3a5f !important; }

.stSuccess { background:#052e16 !important; border-left:4px solid #10b981 !important; border-radius:6px !important; }
.stError   { background:#2d0a0a !important; border-left:4px solid #ef4444 !important; border-radius:6px !important; }
.stInfo    { background:#0c1a2e !important; border-left:4px solid #3b82f6 !important; border-radius:6px !important; }
.stWarning { background:#2d1a00 !important; border-left:4px solid #f59e0b !important; border-radius:6px !important; }

pre { background:#060a12 !important; border:1px solid #1e3a5f !important; border-radius:8px !important; }

.src-pill {
    display:inline-block; background:#0d1526; border:1px solid #1e3a5f;
    border-radius:4px; padding:0.1rem 0.5rem;
    font-family:'Space Mono',monospace; font-size:0.65rem; color:#64748b; margin:0.1rem;
}

.empty-state {
    text-align:center; padding:3rem; opacity:0.35;
    font-family:'Space Mono',monospace; font-size:0.8rem; color:#475569;
}

/* Agent info bar */
.agent-bar {
    display:flex; align-items:center; gap:0.8rem; margin-bottom:0.8rem;
    background:#0a0f1e; border:1px solid #1e3a5f; border-radius:10px; padding:0.7rem 1rem;
}
.agent-bar-icon { font-size:1.4rem; }
.agent-bar-name { font-weight:700; font-size:0.95rem; color:#c7d2fe; }
.agent-bar-desc { font-size:0.72rem; color:#475569; font-family:'Space Mono',monospace; margin-top:0.1rem; }

/* Delegated tag */
.delegate-tag {
    display:inline-block; background:#2d1a00; border:1px solid #f59e0b;
    color:#f59e0b; border-radius:4px; padding:0.05rem 0.4rem;
    font-family:'Space Mono',monospace; font-size:0.6rem; margin-left:0.4rem;
}
</style>
""", unsafe_allow_html=True)


# ── API Keys ─────────────────────────────────────────────────────────────────
# Add all your Gemini API keys here. The pool picks one randomly on startup,
# then rotates via random.choice when a key is exhausted or rate-limited.
API_KEYS = [
    "AIzaSyDLNH0JgvvAAaTrFyXCGDXYdIlOZSpHzMQ",
  "AIzaSyA893DupjddvS2tk641lF40XsDQwZ3rJy8",
    

]


# ── Session state ─────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "api_key":       "",     # active key chosen by random.choice
        "agents_ready":  False,
        "orchestrator":  None,
        "key_pool_ref":  None,
        "active_agent":  "chat",
        "messages":      [],
        "rag_ingested":  False,
        "video_ingested": False,
        "data_loaded":   False,
        "data_filename": "",
        "data_shape":    "",
        "data_columns":  [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


def init_agents() -> None:
    """
    Boot all agents. API key is selected via random.choice (inside key_pool.set_keys).
    No manual input needed — runs automatically on first page load.

    Mirrors the user pattern:
        if 'api_key' not in st.session_state:
            st.session_state.api_key = random.choice(api_keys)
    """
    import random as _random
    
    from backend import MultiAgentOrchestrator, key_pool
    key_pool.set_keys(API_KEYS)                           # random.choice inside
    st.session_state.orchestrator = MultiAgentOrchestrator()
    st.session_state.agents_ready = True
    st.session_state.key_pool_ref = key_pool
    st.session_state.api_key      = key_pool.current_key()  # store active key


# ── Auto-boot on first load ────────────────────────────────────────────────────
if not st.session_state.agents_ready:
    try:
        init_agents()
    except Exception as _boot_err:
        st.session_state["_boot_error"] = str(_boot_err)


def push_msg(role, content, agent="", chart=None, code=None, lang="python",
             sources=None, research_sources=None, queries=None, delegated=False):
    st.session_state.messages.append({
        "role": role, "content": content, "agent": agent,
        "chart": chart, "code": code, "lang": lang,
        "sources": sources or [],
        "research_sources": research_sources or [],
        "queries": queries or [],
        "delegated": delegated,
    })


def get_context():
    return {
        "rag_ingested": st.session_state.rag_ingested,
        "video_ingested": st.session_state.video_ingested,
        "data_loaded": st.session_state.data_loaded,
        "data_filename": st.session_state.data_filename,
        "data_shape": st.session_state.data_shape,
    }


def render_message(msg):
    role = msg["role"]
    content = msg["content"]
    agent = msg.get("agent", "NEXUS")

    if role == "user":
        st.markdown(f"""
<div class="bubble-user">
  <div class="bubble-label">▸ You</div>
  {content}
</div>""", unsafe_allow_html=True)
    else:
        del_tag = '<span class="delegate-tag">↗ delegated</span>' if msg.get("delegated") else ""
        st.markdown(f"""
<div class="bubble-agent">
  <div class="bubble-label">⬡ {agent}{del_tag}</div>
  {content.replace(chr(10), "<br>")}
</div>""", unsafe_allow_html=True)

        if msg.get("chart"):
            st.image(base64.b64decode(msg["chart"]), use_container_width=True)
        if msg.get("code"):
            st.code(msg["code"], language=msg.get("lang", "python"))
        if msg.get("sources"):
            pills = "".join(f'<span class="src-pill">📄 {s}</span>' for s in msg["sources"])
            st.markdown(f'<div style="margin-top:0.4rem">{pills}</div>', unsafe_allow_html=True)
        if msg.get("research_sources"):
            with st.expander("📚 Sources"):
                for s in msg["research_sources"][:12]:
                    if s.get("url"):
                        st.markdown(f"- [{s['title']}]({s['url']})")
                    elif s.get("title"):
                        st.markdown(f"- {s['title']}")
        if msg.get("queries"):
            with st.expander("🔍 Research Queries Used"):
                for q in msg["queries"]:
                    st.markdown(f"`{q}`")


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🤖 AI Nexus Control")
    st.markdown("---")

    # ── API Key: auto-selected via random.choice ──────────────────────────────
    api_keys = [
        "AIzaSyBqYgBLfuGBgNvTE5NJH6vuxDB5DTKWpfQ",
        # "AIzaSy...",   ← paste more keys here
        # "AIzaSy...",
    ]
    if "api_key" not in st.session_state or not st.session_state.api_key:
        st.session_state.api_key = random.choice(api_keys)
    api_key = st.session_state.api_key
    st.success("✅ API Key Auto-Selected")

    # ── Boot agents if not yet running ────────────────────────────────────────
    if not st.session_state.agents_ready:
        with st.spinner("🔄 Starting all agents..."):
            try:
                init_agents()
                st.rerun()
            except Exception as e:
                st.error(f"❌ Boot error: {e}")

    # ── Online badge ──────────────────────────────────────────────────────────
    if st.session_state.agents_ready:
        st.markdown(
            f'<span class="badge on">● ONLINE · Gemini 2.5 Flash · '            f'{len(api_keys)} key(s)</span>',
            unsafe_allow_html=True,
        )

    # ── Live key pool status ──────────────────────────────────────────────────
    if st.session_state.agents_ready and st.session_state.key_pool_ref:
        pool = st.session_state.key_pool_ref
        with st.expander("🔋 Key Pool", expanded=False):
            for slot in pool.status():
                pct    = slot["pct_used"]
                color  = "#10b981" if slot["available"] else "#ef4444"
                marker = "▶ " if slot["active"] else "  "
                exh    = " EXHAUSTED" if slot["exhausted"] else ""
                st.markdown(
                    f'<div style="font-family:Space Mono,monospace;font-size:0.65rem;'                    f'color:#c7d2fe;margin:2px 0">{marker}Key {slot["index"]+1}: '                    f'<span style="color:{color}">{slot["key_preview"]}</span> '                    f'— {slot["tokens_used"]:,}/{slot["token_limit"]:,} ({pct}%){exh}</div>',
                    unsafe_allow_html=True,
                )
                st.progress(min(pct / 100, 1.0))

    st.markdown("---")
    st.markdown("**⬡ Select Agent**")

    AGENTS = [
        ("chat",     "🤖", "General Chatbot",   "Smart hub · memory · auto-delegates"),
        ("rag",      "📄", "RAG Agent",          "Document Q&A (PDF, DOCX, TXT)"),
        ("video",    "🎬", "Video RAG",          "Video frame intelligence"),
        ("data",     "📊", "Data Analyst",       "CSV/Excel analysis + charts"),
        ("code",     "💻", "Code Generator",     "Write · explain · debug"),
        ("research", "🔬", "Deep Researcher",    "Multi-step web research"),
        ("auto",     "🧠", "Auto-Route",         "LLM picks best agent"),
    ]

    for aid, icon, name, desc in AGENTS:
        is_active = st.session_state.active_agent == aid
        label = f"{icon} **{name}** ←" if is_active else f"{icon} {name}"
        if st.button(label, key=f"sbtn_{aid}", use_container_width=True, help=desc):
            st.session_state.active_agent = aid
            st.rerun()

    st.markdown("---")
    st.markdown("**📊 System Status**")
    col1, col2 = st.columns(2)
    col1.markdown(f'<span class="badge {"on" if st.session_state.rag_ingested else ""}">📄 Docs</span>', unsafe_allow_html=True)
    col2.markdown(f'<span class="badge {"on" if st.session_state.video_ingested else ""}">🎬 Video</span>', unsafe_allow_html=True)
    col1.markdown(f'<span class="badge {"on" if st.session_state.data_loaded else ""}">📊 Data</span>', unsafe_allow_html=True)
    col2.markdown(f'<span class="badge info">{len(st.session_state.messages)} msgs</span>', unsafe_allow_html=True)

    st.markdown("")
    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.agents_ready:
            st.session_state.orchestrator.chatbot.clear_history()
        st.rerun()

    if (st.session_state.active_agent == "chat"
            and st.session_state.agents_ready
            and len(st.session_state.messages) > 4):
        if st.button("📝 Summarize Conversation", use_container_width=True):
            with st.spinner("Summarizing..."):
                s = st.session_state.orchestrator.chatbot.get_summary()
            st.info(s)

# ── Main header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="nexus-header">
  <div class="nexus-title">NEXUSRAG</div>
  <div class="nexus-sub">Multi-Agent Intelligence · Gemini 2.5 Flash · LangChain</div>
</div>""", unsafe_allow_html=True)

# Show boot error inline if agents failed to start
if st.session_state.get("_boot_error"):
    st.error(f"⚠️ Agent boot failed: {st.session_state['_boot_error']}")
    st.info("Check that your API key in `app.py → API_KEYS` is valid.")
    st.stop()


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_ingest, tab_viz = st.tabs(["💬 Chat", "📥 Ingest Data", "📊 Visualize"])


# ═══════════════════════════════════════════════
# TAB 1 — CHAT
# ═══════════════════════════════════════════════
with tab_chat:
    active = st.session_state.active_agent
    orch = st.session_state.orchestrator

    AGENT_META = {
        "chat":     ("🤖", "General Chatbot",   "Conversational AI with memory — answers anything, auto-delegates to specialists"),
        "rag":      ("📄", "RAG Agent",          "Semantic search over your uploaded documents"),
        "video":    ("🎬", "Video RAG",          "Frame-by-frame video understanding powered by Gemini Vision"),
        "data":     ("📊", "Data Analyst",       "Intelligent data analysis with AI-generated visualizations"),
        "code":     ("💻", "Code Generator",     "Write, explain, and debug code in any language"),
        "research": ("🔬", "Deep Researcher",    "Multi-step web research → comprehensive structured reports"),
        "auto":     ("🧠", "Auto-Route",         "LLM-powered intent detection picks the optimal agent automatically"),
    }
    icon, name, desc = AGENT_META[active]

    st.markdown(f"""
<div class="agent-bar">
  <div class="agent-bar-icon">{icon}</div>
  <div>
    <div class="agent-bar-name">{name}</div>
    <div class="agent-bar-desc">{desc}</div>
  </div>
</div>""", unsafe_allow_html=True)

    # Chat messages
    if st.session_state.messages:
        for msg in st.session_state.messages:
            render_message(msg)
    else:
        welcome = {
            "chat":     "👋 Hi! I'm **NEXUS**, your central AI assistant.\n\nI can answer general questions directly from my knowledge, and when your request needs a specialist — documents, video, data, code, or research — I'll automatically delegate to the right expert agent and show you the results here.\n\nWhat can I help you with?",
            "rag":      "📄 Upload your documents in **Ingest Data** tab, then ask me anything about them.",
            "video":    "🎬 Upload a video in **Ingest Data** tab, then ask questions about its content.",
            "data":     "📊 Load a CSV or Excel file in **Ingest Data** tab, then ask for analysis or charts.",
            "code":     "💻 Tell me what code you need — I generate, explain, or debug in any language.",
            "research": "🔬 Give me any topic and I'll research it deeply with multiple search queries.",
            "auto":     "🧠 Ask anything — I'll automatically pick the best agent for your request.",
        }.get(active, "Ready.")
        st.markdown(f"""
<div class="bubble-agent">
  <div class="bubble-label">⬡ NEXUS · READY</div>
  {welcome.replace(chr(10),'<br>')}
</div>""", unsafe_allow_html=True)

    st.markdown("<hr style='margin:0.8rem 0'>", unsafe_allow_html=True)

    # ── Per-agent controls ────────────────────────────────────────────────────
    code_language = "Python"
    code_mode = "Generate"
    code_error = ""
    research_depth = "standard"

    if active == "code":
        cc1, cc2, cc3 = st.columns([2, 2, 3])
        code_language = cc1.selectbox("Language",
            ["Python","JavaScript","TypeScript","Java","C++","Rust","Go","SQL","Bash","R","Swift","Kotlin"],
            label_visibility="collapsed")
        code_mode = cc2.selectbox("Mode", ["Generate","Explain","Debug"], label_visibility="collapsed")
        if code_mode == "Debug":
            code_error = cc3.text_input("Error (optional)", placeholder="Paste error message...", label_visibility="collapsed")

    if active == "research":
        rc1, _ = st.columns([2, 5])
        research_depth = rc1.selectbox("Depth", ["quick","standard","deep"], label_visibility="collapsed")

    # ── Input form ────────────────────────────────────────────────────────────
    placeholders = {
        "chat":     "Ask me anything — I'll answer or delegate to the right agent...",
        "rag":      "Ask a question about your uploaded documents...",
        "video":    "Ask about the video content, specific scenes, or objects...",
        "data":     "Ask for analysis or describe a chart you want...",
        "code":     "Describe the code you need, or paste code to explain/debug...",
        "research": "Enter a topic for deep multi-step research...",
        "auto":     "Ask anything — the best agent will handle it...",
    }

    with st.form("chat_form", clear_on_submit=True):
        fc1, fc2 = st.columns([6, 1])
        user_input = fc1.text_area("msg", placeholder=placeholders.get(active, "Type here..."),
                                    height=90, label_visibility="collapsed")
        send = fc2.form_submit_button("Send\n⬡", use_container_width=True)

    # ── Process ───────────────────────────────────────────────────────────────
    if send and user_input.strip():
        push_msg("user", user_input)

        with st.spinner(f"⬡ {name} thinking..."):
            try:
                ctx = get_context()

                # ─── GENERAL CHATBOT ──────────────────────────────────────────
                if active == "chat":
                    result = orch.chatbot.smart_reply(user_input, orch, context_info=ctx)
                    push_msg(
                        "assistant", result["answer"],
                        agent="🤖 General Chatbot",
                        chart=result.get("chart"),
                        code=result.get("code"),
                        lang=result.get("language", "python"),
                        sources=result.get("sources", []),
                        research_sources=result.get("research_sources", []),
                        queries=result.get("queries", []),
                        delegated=result.get("delegated", False),
                    )

                # ─── RAG ─────────────────────────────────────────────────────
                elif active == "rag":
                    r = orch.rag.query(user_input)
                    push_msg("assistant", r["answer"], agent="📄 RAG Agent", sources=r["sources"])

                # ─── VIDEO ───────────────────────────────────────────────────
                elif active == "video":
                    r = orch.video_rag.query(user_input)
                    push_msg("assistant", r["answer"], agent="🎬 Video RAG")

                # ─── DATA ────────────────────────────────────────────────────
                elif active == "data":
                    r = orch.data_analysis.analyze(user_input)
                    push_msg("assistant", r["answer"], agent="📊 Data Analyst", chart=r.get("chart"))

                # ─── CODE ────────────────────────────────────────────────────
                elif active == "code":
                    if code_mode == "Generate":
                        r = orch.code_gen.generate(user_input, language=code_language)
                        push_msg("assistant", r["explanation"], agent="💻 Code Generator",
                                 code=r["code"], lang=code_language.lower())
                    elif code_mode == "Explain":
                        ans = orch.code_gen.explain(user_input)
                        push_msg("assistant", ans, agent="💻 Code Generator")
                    elif code_mode == "Debug":
                        r = orch.code_gen.debug(user_input, error=code_error)
                        push_msg("assistant", r["explanation"], agent="💻 Code Generator",
                                 code=r["fixed_code"], lang="python")

                # ─── RESEARCH ────────────────────────────────────────────────
                elif active == "research":
                    r = orch.researcher.research(user_input, depth=research_depth)
                    push_msg("assistant", r["report"], agent="🔬 Deep Researcher",
                             research_sources=r["sources"], queries=r["queries_used"])

                # ─── AUTO-ROUTE ───────────────────────────────────────────────
                elif active == "auto":
                    route = orch.route(user_input)
                    if route in ("chat", "general"):
                        result = orch.chatbot.smart_reply(user_input, orch, context_info=ctx)
                        push_msg("assistant", f"*[Auto → 🤖 Chatbot]*\n\n{result['answer']}",
                                 agent="🧠 Auto-Route",
                                 chart=result.get("chart"), code=result.get("code"),
                                 sources=result.get("sources", []),
                                 research_sources=result.get("research_sources", []))
                    elif route == "rag":
                        r = orch.rag.query(user_input)
                        push_msg("assistant", f"*[Auto → 📄 RAG]*\n\n{r['answer']}",
                                 agent="🧠 Auto-Route", sources=r["sources"])
                    elif route == "video":
                        r = orch.video_rag.query(user_input)
                        push_msg("assistant", f"*[Auto → 🎬 Video]*\n\n{r['answer']}", agent="🧠 Auto-Route")
                    elif route == "data":
                        r = orch.data_analysis.analyze(user_input)
                        push_msg("assistant", f"*[Auto → 📊 Data]*\n\n{r['answer']}",
                                 agent="🧠 Auto-Route", chart=r.get("chart"))
                    elif route == "code":
                        r = orch.code_gen.generate(user_input)
                        push_msg("assistant", f"*[Auto → 💻 Code]*\n\n{r['explanation']}",
                                 agent="🧠 Auto-Route", code=r["code"])
                    elif route == "research":
                        r = orch.researcher.research(user_input)
                        push_msg("assistant", f"*[Auto → 🔬 Research]*\n\n{r['report']}",
                                 agent="🧠 Auto-Route",
                                 research_sources=r["sources"], queries=r["queries_used"])
                    else:
                        result = orch.chatbot.chat(user_input, ctx)
                        push_msg("assistant", result["answer"], agent="🧠 Auto-Route")

            except Exception as e:
                push_msg("assistant",
                         f"❌ **Error:** {str(e)}\n\nCheck your API key and dependencies.",
                         agent="System")

        st.rerun()


# ═══════════════════════════════════════════════
# TAB 2 — INGEST DATA
# ═══════════════════════════════════════════════
with tab_ingest:
    orch = st.session_state.orchestrator
    ic1, ic2 = st.columns(2)

    with ic1:
        st.markdown("### 📄 Documents")
        st.caption("PDF · DOCX · TXT · MD")
        doc_files = st.file_uploader("Upload docs", accept_multiple_files=True,
                                      type=["pdf","txt","docx","doc","md"], key="du")
        if doc_files and st.button("⬡ Ingest Documents", key="b_docs"):
            paths = []
            for f in doc_files:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix)
                tmp.write(f.read()); tmp.close(); paths.append(tmp.name)
            with st.spinner("Building vector store..."):
                try:
                    msg = orch.rag.ingest(paths)
                    st.session_state.rag_ingested = True
                    st.success(msg)
                    push_msg("assistant",
                             f"✅ **Documents ingested:** {', '.join(f.name for f in doc_files)}\n\n"
                             "Switch to the 💬 Chat tab and ask the 📄 RAG Agent or 🤖 General Chatbot about them.",
                             agent="System")
                except Exception as e:
                    st.error(f"❌ {e}")
        if st.session_state.rag_ingested:
            st.markdown('<span class="badge on">● Documents Ready</span>', unsafe_allow_html=True)

    with ic2:
        st.markdown("### 📊 Data Files")
        st.caption("CSV · Excel · JSON")
        data_file = st.file_uploader("Upload data", type=["csv","xlsx","xls","json"], key="dfu")
        if data_file and st.button("⬡ Load Dataset", key="b_data"):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(data_file.name).suffix)
            tmp.write(data_file.read()); tmp.close()
            try:
                msg = orch.data_analysis.load_data(tmp.name)
                df = orch.data_analysis.df
                st.session_state.data_loaded = True
                st.session_state.data_filename = data_file.name
                st.session_state.data_shape = f"({df.shape[0]:,} × {df.shape[1]})"
                st.session_state.data_columns = df.columns.tolist()
                st.success(msg)
                st.dataframe(df.head(8), use_container_width=True)
                push_msg("assistant",
                         f"✅ **Dataset loaded:** `{data_file.name}` — {df.shape[0]:,} rows × {df.shape[1]} columns\n\n"
                         "Use 📊 Data Analyst, 🤖 General Chatbot, or the Visualize tab.",
                         agent="System")
            except Exception as e:
                st.error(f"❌ {e}")
        if st.session_state.data_loaded:
            st.markdown('<span class="badge on">● Data Ready</span>', unsafe_allow_html=True)
            with st.expander("📋 Data Summary"):
                st.text(orch.data_analysis.get_summary())

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 🎬 Video")
    st.caption("MP4 · AVI · MOV — requires opencv-python")
    vid_file = st.file_uploader("Upload video", type=["mp4","avi","mov","mkv"], key="vfu")
    if vid_file:
        st.video(vid_file)
        if st.button("⬡ Analyze Video", key="b_vid"):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(vid_file.name).suffix)
            tmp.write(vid_file.read()); tmp.close()
            with st.spinner("Gemini Vision analyzing frames..."):
                try:
                    msg = orch.video_rag.ingest(tmp.name)
                    st.session_state.video_ingested = True
                    st.success(msg)
                    push_msg("assistant",
                             f"✅ **Video analyzed:** `{vid_file.name}`\n\n"
                             "Ask the 🎬 Video RAG Agent or 🤖 General Chatbot about the video.",
                             agent="System")
                except Exception as e:
                    st.error(f"❌ {e}")
    if st.session_state.video_ingested:
        st.markdown('<span class="badge on">● Video Ready</span>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# TAB 3 — VISUALIZE
# ═══════════════════════════════════════════════
with tab_viz:
    st.markdown("### 📊 Visualization Studio")

    if not st.session_state.data_loaded:
        st.markdown("""
<div class="empty-state">
  <div style="font-size:2.5rem">📊</div>
  <div style="margin-top:0.8rem">Load a dataset in the Ingest Data tab first</div>
</div>""", unsafe_allow_html=True)
    else:
        orch = st.session_state.orchestrator
        df = orch.data_analysis.df
        cols = df.columns.tolist()
        num_cols = df.select_dtypes(include="number").columns.tolist()

        st.markdown(
            f'<span class="badge on">● {orch.data_analysis.file_name} · {df.shape[0]:,} rows × {df.shape[1]} cols</span>',
            unsafe_allow_html=True)
        st.markdown("")

        # Manual chart builder
        st.markdown("**🎛 Manual Chart Builder**")
        vc1, vc2, vc3, vc4 = st.columns([2, 2, 2, 2])
        chart_type = vc1.selectbox("Type", ["bar","line","scatter","histogram","pie","heatmap","box"])
        x_col = vc2.selectbox("X Axis", ["(auto)"] + cols)
        y_col = vc3.selectbox("Y Axis", ["(auto)"] + num_cols)
        chart_title = vc4.text_input("Title", placeholder="Chart title...")

        if st.button("⬡ Render Chart", use_container_width=True):
            x = None if x_col == "(auto)" else x_col
            y = None if y_col == "(auto)" else y_col
            with st.spinner("Rendering..."):
                try:
                    b64 = orch.data_analysis.custom_chart(chart_type, x, y, chart_title)
                    if b64:
                        st.image(base64.b64decode(b64), use_container_width=True)
                    else:
                        st.error("Chart failed.")
                except Exception as e:
                    st.error(f"❌ {e}")

        st.markdown("<hr>", unsafe_allow_html=True)

        # AI auto chart
        st.markdown("**🤖 AI-Powered Chart**")
        st.caption("Describe the chart — AI picks the best type and columns")
        ai_col1, ai_col2 = st.columns([5, 1])
        ai_prompt = ai_col1.text_input("Describe chart", placeholder="e.g. 'Bar chart of average salary by department'", label_visibility="collapsed")
        ai_btn = ai_col2.button("⬡ Go", use_container_width=True)

        if ai_btn and ai_prompt:
            with st.spinner("Analyzing + rendering..."):
                try:
                    r = orch.data_analysis.analyze(ai_prompt)
                    st.markdown(f"**Analysis:** {r['answer']}")
                    if r.get("chart"):
                        st.image(base64.b64decode(r["chart"]), use_container_width=True)
                except Exception as e:
                    st.error(f"❌ {e}")

        st.markdown("<hr>", unsafe_allow_html=True)

        # Data preview
        pc1, pc2 = st.columns([3, 2])
        with pc1:
            st.markdown("**Data Preview**")
            n = st.slider("Rows to show", 5, 100, 10)
            st.dataframe(df.head(n), use_container_width=True)
        with pc2:
            st.markdown("**Numeric Summary**")
            st.dataframe(df.describe(), use_container_width=True)
