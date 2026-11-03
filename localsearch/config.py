"""Engine configuration; factory defaults are the brick-1..4 validated config."""
from dataclasses import dataclass, field

from . import resources


@dataclass
class Config:
    model_name: str = field(default_factory=resources.model_path)
    query_prefix: str = "query: "             # multilingual-e5 wants both prefixes
    doc_prefix: str = "passage: "
    device: str = "auto"
    embed_batch_size: int = 128           # chunks gathered across files per embed call
    encode_batch_size: int = 0            # model batch; 0 = auto (CPU 16, GPU 128)

    alpha: float = 0.7
    fusion_fetch: int = 100
    instant_candidates: int = 120

    flush_debounce_s: float = 1.0
    save_interval_s: float = 120.0
    instant_cache_entries: int = 20000

    snippet_sentence_chars: int = 300
    snippet_sents_per_hit: int = 16
    instant_deep_snippet: bool = True
    instant_snippet_max_kb: int = 64

    chunk_max_words: int = 150
    chunk_overlap_sentences: int = 1

    # personalization: local + training-free (learns from what you OPEN, never
    # trains the model). Base relevance still dominates; this only nudges order.
    personalize: bool = True
    pers_lambda: float = 0.25         # personal prior weight vs base relevance
    pers_w_click: float = 0.5         # sub-weights within the personal prior
    pers_w_folder: float = 0.2
    pers_w_interest: float = 0.3
    pers_decay: float = 0.9           # older opens decay -> recency-weighted profile
    pers_top_margin: float = 0.1      # the best match is never displaced by a result
                                      # less relevant than it by more than this share
                                      # of the candidates' score range

    supported_ext: tuple = (".txt", ".md", ".pdf", ".docx", ".xlsx", ".pptx",
                            ".csv", ".html", ".htm", ".rtf")
    ocr: bool = False                 # opt-in image OCR (Windows built-in engine)
    change_detection: str = "mtime_size"
    max_file_mb: float = 50.0

    exclude_dirs: frozenset = frozenset({
        "windows", "program files", "program files (x86)", "programdata",
        "$recycle.bin", "system volume information", "$winreagent", "recovery",
        "appdata", "application data", "node_modules", "site-packages",
        "__pycache__", ".git", ".svn", ".hg", ".venv", "venv", "env", ".env",
        ".cache", ".gradle", ".m2", ".nuget", ".cargo", "build", "dist",
        "obj", "bin", ".idea", ".vs", ".vscode", "temp", "tmp", "cache",
        "msys64", "anaconda3", "miniconda3",
    })
    exclude_hidden: bool = True

    es_exe: str = field(default_factory=lambda: str(resources.es_exe()))
    es_timeout_s: float = 5.0

    watch_debounce_s: float = 0.6

    def resolved_device(self):
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
