"""
backend/orchestrator.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MultiAgentOrchestrator — owns one instance of each specialist agent and
provides a top-level route() method for intent-based dispatch.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from .base          import get_llm
from .rag_agent      import RAGAgent
from .video_agent    import VideoRAGAgent
from .data_agent     import DataAnalysisAgent
from .code_agent     import CodeGeneratorAgent
from .research_agent import DeepResearcherAgent
from .chat_agent     import GeneralChatbotAgent


class MultiAgentOrchestrator:
    """
    Central coordinator. Owns one instance of every agent.
    route(query) → one of: rag | video | data | code | research | chat
    """

    def __init__(self):
        self.rag            = RAGAgent()
        self.video_rag      = VideoRAGAgent()
        self.data_analysis  = DataAnalysisAgent()
        self.code_gen       = CodeGeneratorAgent()
        self.researcher     = DeepResearcherAgent()
        self.chatbot        = GeneralChatbotAgent()

    def route(self, query: str) -> str:
        """Use Gemini to classify query intent."""
        llm = get_llm(temperature=0.0)
        prompt = (
            f'Classify this query: "{query}"\n\n'
            "Options: rag | video | data | code | research | chat\n"
            "Reply with ONLY one word."
        )
        result = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
        return result if result in {"rag", "video", "data", "code", "research", "chat"} else "chat"
