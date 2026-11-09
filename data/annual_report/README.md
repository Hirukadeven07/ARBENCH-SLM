# Put the annual report here

Drop one (or several) annual reports into this folder — **PDF**, DOCX, HTML, MD or TXT — and run:

```
.\run.ps1            # Windows
./run.sh             # Linux / macOS / Colab
```

The report is extracted, cleaned, chunked, turned into a benchmark, and the SLMs are evaluated automatically.
Scanned (image-only) PDFs need OCR first, e.g. `ocrmypdf in.pdf out.pdf`.

(Folders named `data anual report`, `data/anual_report` etc. are also detected automatically.)
