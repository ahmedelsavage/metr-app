# docker-compose

```bash
docker-compose up --build
```

```bash
http://localhost:8501
```

# Data source

- Source: `coastalcph/lex_glue`, subset `unfair_tos`
- Papers: LexGLUE (Chalkidis et al.) & Unfair-ToS (Lippi et al.)
- [Hugging face Datasets](https://huggingface.co/datasets/coastalcph/lex_glue)

# How to run OCR
```python
python -m src.ocr <image_path>
```
