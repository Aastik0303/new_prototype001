"""
backend/__init__.py  — public API
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Module layout
─────────────
  base.py           — ApiKeyPool (multi-key rotation), get_llm(), get_embeddings()
  rag_agent.py      — RAGAgent      + create_rag_agent()
  video_agent.py    — VideoRAGAgent + create_video_rag_agent()
  data_agent.py     — DataAnalysisAgent  + create_data_analysis_agent()
  code_agent.py     — CodeGeneratorAgent + create_code_generator_agent()
  research_agent.py — DeepResearcherAgent + create_deep_researcher_agent()
  chat_agent.py     — GeneralChatbotAgent + create_general_chatbot_agent()
  orchestrator.py   — MultiAgentOrchestrator

Quick start:
    from backend import MultiAgentOrchestrator, key_pool
    key_pool.set_keys(["AIza...", "AIza...", ...])
    orch = MultiAgentOrchestrator()
"""

from .base           import key_pool, get_llm, get_embeddings
from .orchestrator   import MultiAgentOrchestrator
from .rag_agent      import RAGAgent,            create_rag_agent
from .video_agent    import VideoRAGAgent,        create_video_rag_agent
from .data_agent     import DataAnalysisAgent,    create_data_analysis_agent
from .code_agent     import CodeGeneratorAgent,   create_code_generator_agent
from .research_agent import DeepResearcherAgent,  create_deep_researcher_agent
from .chat_agent     import GeneralChatbotAgent,  create_general_chatbot_agent

__all__ = [
    "key_pool",
    "get_llm", "get_embeddings",
    "MultiAgentOrchestrator",
    "RAGAgent",             "create_rag_agent",
    "VideoRAGAgent",        "create_video_rag_agent",
    "DataAnalysisAgent",    "create_data_analysis_agent",
    "CodeGeneratorAgent",   "create_code_generator_agent",
    "DeepResearcherAgent",  "create_deep_researcher_agent",
    "GeneralChatbotAgent",  "create_general_chatbot_agent",
]
