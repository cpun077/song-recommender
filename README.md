# Song Recommender

A content-based music recommendation system. Give it a song (or two), get back the top k most similar tracks ranked by audio features and lyrics.

## How to Run the Application

This project consists of a FastAPI backend and a React (Vite) frontend. You need to run them simultaneously in two separate terminal windows.

### 1. Start the Backend (FastAPI)

Open your first terminal and run the following commands from the project root directory:

```bash
# 1. Activate your virtual environment
source .venv/bin/activate

# 2. Start the server
python3 -m uvicorn api:app --port 8000 --reload
```

The API will be available at http://localhost:8000.

### 2. Start the Frontend (Vite/React)

Open your second terminal and run:

```bash
# 1. Navigate into your frontend folder
cd frontend

# 2. Start the Vite development server
npm run dev
```

Your frontend will usually be accessible at http://localhost:5173 (check the terminal output for the exact URL).

## Evals

Two recommendation models are evaluated:
- **singlestage** — ranks all songs by a weighted fusion of audio and lyrics similarity
- **twostage** — retrieves candidates from audio and lyrics separately, then reranks by weighted similarity scoring

### Query Set Construction

200 songs sampled via EDA: K-Means clustering on audio features, t-SNE for visualization, cluster count selected via silhouette scores and visual inspection. Final sample is proportionally stratified by cluster × release era (pre-2000, 2000s, 2010s, 2020+).

### Running Evals

The query set (`eval-queries.csv`) and recommendation set (`eval-recs.csv`) are already generated in `backend/data/`. To regenerate recommendations:

```bash
cd backend
python3 eval.py
```

**Note:** Manual human relevance scoring of `eval-recs.csv` is not yet complete. Once scored, calculate metrics (NDCG@10, Precision@10, Mean Relevance@10) with:

```python
from eval import evaluate_model
import pandas as pd

eval_recs = pd.read_csv('backend/data/eval-recs.csv')
evaluate_model(eval_recs, 'singlestage')
evaluate_model(eval_recs, 'twostage')
```