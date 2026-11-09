"""ARBench – a local annual-report benchmark for small language models.

Pipeline (re-engineered from HuggingFace YourBench + LightEval):

    ingest -> clean -> chunk -> summarise -> generate questions -> filter
           -> build benchmark (LightEval-style tasks) -> evaluate (Ollama)
           -> judge -> score -> per-model + comparison reports
"""

__version__ = "1.0.0"
