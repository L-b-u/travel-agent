"""RAG 检索器测试：索引构建、相关性、城市偏好加权。"""

from __future__ import annotations

from app.core.travel.rag import get_retriever
from app.core.travel.rag.retriever import KNOWLEDGE_DIR, _load_chunks, _tokenize


def test_index_built():
    r = get_retriever()
    assert r.status["chunks"] >= 30
    assert len(r.status["cities"]) == 8
    assert r.status["mode"] in ("bm25", "hybrid")


def test_chunking_by_section():
    chunks = _load_chunks(KNOWLEDGE_DIR)
    assert all("## " not in c.section for c in chunks)  # 切块后章节名不含标记
    assert any(c.city == "杭州" for c in chunks)
    assert len(chunks) == get_retriever().status["chunks"]


def test_relevant_section_ranks_first():
    r = get_retriever()
    hits = r.retrieve("玉龙雪山要提前预约吗，怕高反", city="丽江", k=2)
    assert hits, "应有检索结果"
    assert hits[0]["city"] == "丽江"
    # 玉龙雪山/高反两个专属章节至少命中其一
    top_sections = " ".join(h["section"] for h in hits)
    assert ("玉龙雪山" in top_sections) or ("高反" in top_sections)


def test_city_boost_prefers_target_city():
    r = get_retriever()
    hits = r.retrieve("海鲜市场怎么买", city="三亚", k=3)
    assert hits[0]["city"] == "三亚"


def test_tokenizer_filters_stopwords():
    tokens = _tokenize("我想去玩一下杭州的博物馆")
    assert "我" not in tokens and "想" not in tokens
    assert "杭州" in "".join(tokens) or "博物" in "".join(tokens)
