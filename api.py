from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from contextlib import asynccontextmanager

from backend.recommender import preprocess, recommend

# Global variables to store the data
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load data ONCE so it's not processed on every request
    print("Loading datasets and precomputing TF-IDF vectors...")
    df = pd.read_csv('./backend/data/top-10k-spotify-songs-2025-07-detailed.csv')
    precomputed = preprocess(df.copy())
    
    ml_models["df"] = df
    ml_models["precomputed"] = precomputed
    print("Precomputation complete. Server ready!")
    yield
    # Clean up (if needed)
    ml_models.clear()

app = FastAPI(title="Song Recommender API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, restrict this to the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/recommend")
async def get_recommendations(song: str, count: int = 5):
    if "df" not in ml_models or "precomputed" not in ml_models:
        raise HTTPException(status_code=503, detail="Model is loading")
        
    df = ml_models["df"]
    precomputed = ml_models["precomputed"]
    
    # Run the recommender
    result = recommend(df, song, n=count, precomputed=precomputed)
    
    if isinstance(result, str):
        # Result is a string when "Song not found in database"
        raise HTTPException(status_code=404, detail=result)
        
    # Convert dataframe result to list of dicts for JSON
    recommendations = result.to_dict(orient="records")
    return {"recommendations": recommendations}
