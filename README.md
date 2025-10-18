## Uniform Segmentation FastAPI Server

Run a FastAPI backend that loads the SegFormer clothes segmentation model and exposes an endpoint to segment uploaded images.

### Setup

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints

- `GET /health` → health check
- `POST /segment` → form-data upload `file` (image). Query params:
  - `output`: `mask` (default) or `overlay`
  - `alpha`: overlay transparency in [0, 1] (used when `output=overlay`)

Returns a PNG image.

### Environment variables

- `SEGFORMER_MODEL` (optional): override model name, default `mattmdjaga/segformer_b2_clothes`.


# uniform
