"""localsearch -- a local, offline semantic + filename file-search backend engine.

Factory config: multilingual-e5-small embeddings + dense-weighted hybrid
(dense + incremental BM25, alpha=0.7) + Everything filename search; no reranker.

    from localsearch import SearchEngine, Config
    eng = SearchEngine("my_index")
    eng.build(["C:/docs"])                    # scan + index (incremental)
    for hit in eng.search_semantic("query"):  # hybrid semantic
        print(hit.path, hit.score, hit.chunk_text[:80])
    print(eng.search_filename("*.pdf"))       # Everything filename search
    eng.start_watching(["C:/docs"])           # live incremental
"""
from .config import Config
from .engine import SearchEngine
from .search import Hit

__all__ = ["SearchEngine", "Config", "Hit"]
__version__ = "0.1.0"
