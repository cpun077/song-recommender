# song-recommender

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