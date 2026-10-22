"""Unit tests for the retrieval helpers (no network calls)."""
from brand_assistant_rag import (
    build_queries,
    chunk_text,
    clip_text,
    mentions_company,
    root_domain,
    slugify,
)


def test_chunk_text_uses_size_and_overlap():
    text = "x" * 3000
    chunks = chunk_text(text, size=1200, overlap=200)
    assert [len(c) for c in chunks] == [1200, 1200, 1000]  # starts at 0, 1000, 2000


def test_chunk_text_empty_input():
    assert chunk_text("   ") == []


def test_slugify():
    assert slugify("  Nike, Inc.  ") == "nike-inc"
    assert slugify("!!!") == "item"


def test_mentions_company_matches_alias_case_insensitively():
    assert mentions_company("A new campaign from NIKE this fall", "Nike")
    assert not mentions_company("A new campaign from Adidas", "Nike")


def test_clip_text_respects_budget():
    clipped = clip_text("one two three four five six", 12)
    assert clipped.endswith("...") and len(clipped) <= 15


def test_root_domain_strips_www():
    assert root_domain("WWW.Nike.com") == "nike.com"


def test_build_queries_adds_person_queries_only_for_names():
    role_queries = build_queries("Nike", "Director of Brand Marketing")
    name_queries = build_queries("Nike", "Jane Doe")
    assert len(name_queries) > len(role_queries)
    assert all('"Nike"' in q for q in role_queries)
