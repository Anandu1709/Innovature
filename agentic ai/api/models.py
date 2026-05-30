"""
Pydantic Schemas — ChatRequest and ChatResponse matching Section 7.4 contracts.

ChatRequest:  Incoming query + optional session_id.
ChatResponse: Answer, single top image, sources, session tracking.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
import time


class ChatRequest(BaseModel):
    """Incoming chat request from the frontend."""
    query: str
    session_id: Optional[str] = None  # None → server generates UUID


class ChatResponse(BaseModel):
    """
    Outgoing chat response — Section 7.4 contract.

    Mapping from AgentState:
      intent == "clarify" → answer = state["clarification_question"]
      intent != "clarify" → answer = state["final_answer"]
      image_url           → top-scoring image from state["image_results"]
    """
    session_id: str
    intent: str                                        # "text"|"visual"|"both"|"clarify"
    answer: Optional[str] = None                       # Answer OR clarification question
    image_url: Optional[str] = None                    # "/images/arduino/file.png"
    image_caption: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    cache_hit: bool = False                            # True if served from cache
    cached_at: Optional[float] = None                  # Unix timestamp when cached
    coverage_found: bool = True
    offer_global_search: bool = False
    global_search_requested: bool = False
