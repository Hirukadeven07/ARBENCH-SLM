# ARBench — Local Annual-Report Benchmark for Small Language Models

Turns **your annual report** into a benchmark and evaluates **Granite 4.1, Gemma 4 and Qwen 3** on it —
fully local with [Ollama](https://ollama.com), with **one command**.

ARBench re-engineers two Hugging Face projects into a single local pipeline:

| From | What was taken | What changed |
|---|---|---|
| [YourBench](https://github.com/huggingface/yourbench) | ingestion → chunking → summarisation → single-hop / multi-hop / multiple-choice question generation → citation-score filtering → dedup → LightEval export schema | Ollama instead of HF Inference/OpenAI; JSON-schema constrained decoding for 2-4B models; structure-aware chunking that keeps tables and page numbers; number-grounding filter for financial figures; generator rotation across models; cross-model verification that removes mislabelled items |
| [LightEval](https://github.com/huggingface/lighteval) | task/prompt-function design, `helm_normalizer`, F1 / exact match / ROUGE, gpqa-style letter extraction, the YourBench LLM-judge task, mean-stderr | Ollama backend; LLM **jury** with cross-judging; new tasks (RAG QA, unanswerable/hallucination probe); finance metrics (numeric match, number hallucination); bootstrap CIs + paired significance tests; efficiency metrics |

---

## Quick start

1. Install **Ollama** (https://ollama.com/download) and **Python 3.11+**.
2. Put the annual report (PDF/DOCX/HTML/MD/TXT) in **`data/annual_report/`**.
3. Run **one command**:

```powershell
.\run.ps1                    # Windows PowerShell   (or run.bat)
```
```bash
./run.sh                     # Linux / macOS
```

The first run creates `.venv`, installs dependencies, starts Ollama if needed, and pulls the models.
Everything after that runs automatically.

| Profile | Use | Approx. time (3 models) |
|---|---|---|
| `--profile smoke` | plumbing check | ~15-30 min on a laptop CPU |
| `--profile quick` | first look (~30 questions) | ~1-3 h on a laptop CPU |
| *(default)* `standard` | ~70 questions, ~200 eval items | ~4-8 h on CPU · ~30 min on an L4 |
| `--profile full` / `--profile colab` | ~200 questions, broadest coverage | GPU recommended |

**Interrupted? Run the same command again with the same `--run-name`.** Every model call is cached in
`outputs/.cache/llm_cache.sqlite`, so finished work is never repeated.

### Google Colab (L4 GPU)
Open `notebooks/ARBench_Colab_L4.ipynb` in Colab, select an **L4** runtime and run all cells. It installs Ollama,
pulls the models, asks for the annual report, and runs `python run_benchmark.py --profile colab`.

---

## What you get (`outputs/<run>/`)

```
benchmark/
  annual_report_benchmark.jsonl   <- THE benchmark: every eval item with task, question, gold answer, context
  questions.jsonl                 <- validated questions with citations + grounding scores
  BENCHMARK_CARD.md               <- how the benchmark was built
  lighteval/                      <- same data in YourBench->LightEval schema + custom_task.py (runs in real LightEval)
reports/
  comparison_report.md            <- leaderboard, significance tests, verdict, charts
  <model>_report.md               <- one detailed report per SLM
  report.html                     <- self-contained dashboard (open in a browser)
  charts/*.png
results/results.json, summary.csv <- all numbers
predictions/<model>.jsonl         <- every answer, metric, and judge verdict (LightEval "details")
01_extracted/  02_chunks.jsonl  03_summaries.json  04_questions_raw.jsonl  run.log
```

---

## The pipeline

```
annual report ─► 1 ingest (PDF→Markdown, tables kept, headers/footers/page numbers removed)
              ─► 2 chunk  (~700-token structure-aware chunks with page + section provenance; TOC/boilerplate dropped)
              ─► 3 summarise (orientation for the question writer only)
              ─► 4 generate  single-hop · multi-hop (2 related chunks) · multiple-choice   [YourBench]
                             written in rotation by the models under test, with verbatim citations
              ─► 5 filter    citation score · number grounding · hygiene · near-duplicates  [YourBench]
              ─► 5b verify   the models that did NOT write an item re-check it against the evidence;
                             items they unanimously reject (mislabelled MCQ, unsupported gold) are dropped
              ─► 6 tasks     grounded QA · RAG QA · MCQ · unanswerable · (closed-book)     [LightEval]
              ─► 7 evaluate  temperature 0, fixed seed, one model in memory at a time
              ─► 8 judge     LLM jury: each answer graded by the *other* models (no self-grading)
              ─► 9 score     means, stderr, bootstrap 95% CIs, paired bootstrap tests, composite
              ─► 10 report   per-model reports + comparison + HTML dashboard
```

### Tasks

| Task | Model sees | Primary metric | Tests |
|---|---|---|---|
| **grounded_qa** | gold evidence chunk(s) | jury judge score | reading comprehension of the report |
| **rag_qa** | top-4 BM25 chunks from the whole report | jury judge score | realistic "chat with the annual report" |
| **multi_choice** | evidence chunk + 4 options | accuracy | precise discrimination between near-miss figures |
| **unanswerable** | a related chunk that lacks the answer | correct abstention rate | hallucination resistance |
| closed_book (off) | nothing | judge score | prior knowledge / guessing |

### Metrics
* **Judge score**: correct = 1, partially correct = 0.5, incorrect = 0, averaged over the jury. Also reported: majority accuracy and inter-judge agreement.
* **Token F1, ROUGE-L, exact match**: LightEval definitions with `helm_normalizer`.
* **Numeric match**: share of the gold answer's figures that the model reproduces. Format- and scale-tolerant: `£1.2bn` = `1,200 million`.
* **Number hallucination**: share of the model's figures that appear nowhere in its context.
* **Over-refusal**: the model said "insufficient context" although the answer was there.
* **Efficiency** (from Ollama timings): generation tokens/s, prompt tokens/s, mean and p95 latency.
* **ARBench score**: a weighted mean of grounded QA 30%, RAG 25%, MCQ 20%, unanswerable 15% and efficiency 10%. The weights are set in `config/default.yaml`.

---

## Useful options

```bash
python run_benchmark.py --profile quick
python run_benchmark.py --models granite4.1:8b gemma4:e4b-it-qat qwen3:8b      # other sizes
python run_benchmark.py --judge qwen3:8b             # one fixed (bigger) judge instead of the cross jury
python run_benchmark.py --no-judge                   # lexical metrics only (fastest)
python run_benchmark.py --generate-only              # just build the benchmark file
python run_benchmark.py --reuse-benchmark outputs/<run>   # evaluate (new) models on an existing benchmark
python run_benchmark.py --run-name myrun             # fixed folder name -> resume after interruption
```

All settings (chunk sizes, number of questions, filters, tasks, weights, judge strategy, models) live in
`config/default.yaml`.

## Default models

| Label | Ollama tag | Size |
|---|---|---|
| Granite 4.1 (3B) | `granite4.1:3b` | 2.1 GB |
| Gemma 4 (E2B) | `gemma4:e2b-it-qat` | 4.3 GB |
| Qwen 3 (4B) | `qwen3:4b-instruct` | 2.5 GB |

These sizes fit a 16 GB-RAM laptop; only one model is loaded at a time. On an L4, try `granite4.1:8b`,
`gemma4:e4b-it-qat` and `qwen3:8b`.

## Tests

```bash
.venv\Scripts\python -m pytest        # unit tests + a full pipeline run with a fake Ollama client
```

## Limitations
* The benchmark questions and the judge verdicts come from small models. Results show **relative** ability on *this* report, so read the example errors in the per-model reports.
* Small benchmarks give wide confidence intervals. The comparison report marks which gaps are statistically significant.
* Scanned PDFs need OCR first (`ocrmypdf`).
