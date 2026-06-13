"""scan() primitive tests — parity and contract (issue #1797).

Covers ChromaCollection.scan() via the SQL fast path, the base-class
fallback for complex where-filters, and the EmbeddingCollection delegation.
All parity assertions use the same ChromaCollection adapter pattern as
test_palace_stats.py.
"""

from mempalace.backends.base import BaseCollection
from mempalace.backends.chroma import ChromaCollection


def _adapter(collection, palace_path):
    return ChromaCollection(collection, palace_path=palace_path)


# ---------------------------------------------------------------------------
# Parity: SQL fast path == get() (chroma client) ground truth
# ---------------------------------------------------------------------------


def test_scan_parity_full(seeded_collection, collection, palace_path):
    """list(col.scan(include=[documents,metadatas])) matches col.get() ground truth."""
    col = _adapter(collection, palace_path)

    raw = col.get(include=["documents", "metadatas"])
    expected_by_id = {
        did: (doc, meta) for did, doc, meta in zip(raw.ids, raw.documents, raw.metadatas)
    }

    scan_results = list(col.scan(include=["documents", "metadatas"]))
    assert len(scan_results) == len(expected_by_id), "row count mismatch"

    for did, doc, meta in scan_results:
        assert did in expected_by_id, f"unexpected id {did!r}"
        exp_doc, exp_meta = expected_by_id[did]
        assert doc == (exp_doc or ""), f"document mismatch for {did!r}"
        # Only compare keys that are present in metadata (chroma:document excluded)
        for k, v in exp_meta.items():
            assert meta.get(k) == v, f"metadata[{k!r}] mismatch for {did!r}"


def test_scan_metadata_only_excludes_documents(seeded_collection, collection, palace_path):
    """scan(include=['metadatas']) yields every drawer with document == ''."""
    col = _adapter(collection, palace_path)

    results = list(col.scan(include=["metadatas"]))
    total = col.count()
    assert len(results) == total

    for _did, doc, _meta in results:
        assert doc == "", f"expected empty doc for metadata-only scan, got {doc!r}"


def test_scan_documents_match_get(seeded_collection, collection, palace_path):
    """scan(include=['documents','metadatas']) documents match get()."""
    col = _adapter(collection, palace_path)

    raw = col.get(include=["documents", "metadatas"])
    doc_by_id = dict(zip(raw.ids, raw.documents))

    for did, doc, _meta in col.scan(include=["documents", "metadatas"]):
        assert doc == (doc_by_id.get(did) or ""), f"doc mismatch for {did!r}"


def test_scan_where_filter(seeded_collection, collection, palace_path):
    """scan(where={'wing': ...}) returns exactly the drawers get(where=...) returns."""
    col = _adapter(collection, palace_path)

    raw = col.get(where={"wing": "project"}, include=["metadatas"])
    expected_ids = set(raw.ids)

    scan_ids = {did for did, _doc, _meta in col.scan(where={"wing": "project"})}
    assert scan_ids == expected_ids


def test_scan_limit(seeded_collection, collection, palace_path):
    """scan(limit=2) yields exactly 2 rows."""
    col = _adapter(collection, palace_path)

    results = list(col.scan(limit=2))
    assert len(results) == 2


def test_scan_empty_collection(palace_path, collection):
    """scan over an empty collection yields nothing."""
    col = _adapter(collection, palace_path)
    results = list(col.scan())
    assert results == []


def test_scan_complex_where_fallback(seeded_collection, collection, palace_path):
    """scan(where={'wing': {'$in': [...]}}) falls back to base and returns correct results."""
    col = _adapter(collection, palace_path)

    where = {"wing": {"$in": ["project", "notes"]}}
    raw = col.get(where=where, include=["metadatas"])
    expected_ids = set(raw.ids)

    scan_ids = {did for did, _doc, _meta in col.scan(where=where)}
    assert scan_ids == expected_ids
    # Confirm it found something (sanity check)
    assert len(scan_ids) > 0


def test_scan_drawer_with_no_user_metadata(palace_path, collection):
    """A drawer with no user metadata (only chroma:document) still appears in scan."""
    col = _adapter(collection, palace_path)

    # Add a drawer with no user-level metadata keys at all
    collection.add(
        ids=["bare_drawer"],
        documents=["Some content with no wing or room."],
        metadatas=[{"_only_internal": "yes"}],
    )

    ids_seen = {did for did, _doc, _meta in col.scan(include=["metadatas"])}
    assert "bare_drawer" in ids_seen


# ---------------------------------------------------------------------------
# Base-class fallback path (no SQL override)
# ---------------------------------------------------------------------------


def test_base_scan_parity(seeded_collection, collection, palace_path):
    """BaseCollection.scan() (offset-paginated) produces identical id set as SQL path."""
    col = _adapter(collection, palace_path)

    sql_ids = {did for did, _doc, _meta in col.scan()}
    base_ids = {did for did, _doc, _meta in BaseCollection.scan(col)}
    assert sql_ids == base_ids


# ---------------------------------------------------------------------------
# EmbeddingCollection delegation
# ---------------------------------------------------------------------------


def test_embedding_wrapper_delegates_scan(seeded_collection, collection, palace_path):
    """EmbeddingCollection.scan() delegates to the inner ChromaCollection."""
    from mempalace.backends.embedding_wrapper import EmbeddingCollection

    col = _adapter(collection, palace_path)
    wrapped = EmbeddingCollection(col)

    inner_ids = {did for did, _doc, _meta in col.scan()}
    wrapper_ids = {did for did, _doc, _meta in wrapped.scan()}
    assert wrapper_ids == inner_ids
