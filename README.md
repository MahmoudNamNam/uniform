

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