# metr-app

Arabic contract risk and fraud analysis, built on a RAG (Retrieval-Augmented Generation) pipeline grounded in Egyptian law. Upload a contract (PDF, image, or text) and get a summary, the risky or fraudulent clauses, the law articles that apply to each one, and a final verdict.

The name Metr comes from "Maître", a French title used for lawyers in Egypt.

## How it works

```
Contract file
    |
    v
Ingest        PDF text layer, or PaddleOCR for scans and images
    |
    v
Clause split  one clause at a time
    |
    v
RAG           search Egyptian laws for each clause (Qdrant + bge-m3)
    |
    v
LLM           Qwen3-8B reads the contract and the retrieved articles
    |
    v
Result        summary, risks, severity, law reference, verdict
```

The knowledge base behind the RAG is a starting point: the Civil Code, the Real Estate Registration Law, and the Consumer Protection Law. It will keep growing and improving.

Risk types: `unilateral_change`, `unilateral_termination`, `unlimited_liability`, `liability_exclusion`, `unfair_arbitration`, `unfair_deduction`, `unfair_content_removal`, `other`.

## Quick start

```bash
cp src/.env.example src/.env
```

Set the LLM server in `src/.env`:

```
QWEN8_API_URL=<your server url>
QWEN8_API_KEY=<your key>
```

Run:

```bash
docker-compose up --build
```

| Service | Address |
|---|---|
| Web UI | http://localhost:8501 |
| API docs | http://localhost:8000/docs |

To change the LLM server later, edit the two variables and run `docker-compose up -d --force-recreate api`.

## API

| Endpoint | Description |
|---|---|
| `GET /health` | Service status |
| `POST /search` | Search the law articles |
| `POST /summarize` | Summarize a contract |
| `POST /analyze` | Full risk analysis |

## Project structure

```
src/
  main.py        API
  pipeline.py    Connects ingest, RAG, and LLM
  ingest.py      File reading
  config.py      Settings from .env
  schemas.py     Typed models
  benchmark.py   LLM evaluation
  stores/
    ocr/  rag/  llm/
docker/
ml/              Research notebooks
```

## Scalable and maintainable

- **Modular:** OCR, RAG, and LLM are separate packages. Each can be replaced on its own.
- **One config:** models and server URLs come from `.env`, with no code change.
- **Validated output:** every LLM answer is checked against a schema and retried if invalid.
- **Measured:** `benchmark.py` scores the LLM on labeled cases and logs results to MLflow.
- **Containerized:** one command starts everything, and model downloads are cached.

## Research (ML and BERT)

Before the LLM pipeline, we tested ML and BERT models for detecting unfair terms. These are experiments in `ml/` and are not part of the running app.

**Data:** `unfair_tos` from LexGLUE ([Hugging Face](https://huggingface.co/datasets/coastalcph/lex_glue)). Papers: LexGLUE (Chalkidis et al.) and Unfair-ToS (Lippi et al.).

**ML:** 12 classical models compared. The best two:

| Model | Precision | Recall | F1 | Accuracy | Cost |
|---|---|---|---|---|---|
| `lr_tuned` | 0.639 | 0.861 | 0.733 | 0.937 | 272 |
| `svm_word_char` | 0.655 | 0.843 | 0.738 | 0.939 | 282 |

Lower cost is better.

**BERT:** fine-tuned to detect 8 unfair-term categories, with a tuned threshold for each.

| Model | Base |
|---|---|
| `legal-bert` | `nlpaueb/legal-bert-base-uncased` |
| `xlmr-en` | `xlm-roberta-base` |
| `xlmr-en-ar` | `xlm-roberta-base`, with Arabic data translated by `facebook/nllb-200-distilled-600M` |

## Team

Connect with us on LinkedIn:

- [Ahmed Mohamed](https://www.linkedin.com/in/ahmedelsavage)
- [Abdelrahman Mokhtar](https://www.linkedin.com/in/amokhtar51)
- [Toka Abdelaziz](https://www.linkedin.com/in/toka-abdelaziz-80a004306/)
- [Mohamed Tamer](https://www.linkedin.com/in/mohamed-tamer-562921231/)
- Asharf Mohamed
