# -*- coding: utf-8 -*-
"""旅行攻略知识库（轻量 RAG）：BM25 + 可选向量混合检索。"""

from app.core.travel.rag.retriever import TravelKnowledgeRetriever, get_retriever

__all__ = ["TravelKnowledgeRetriever", "get_retriever"]
