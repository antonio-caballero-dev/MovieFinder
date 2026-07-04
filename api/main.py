from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.search import router
from api.retriever import OnnxDenseRetriever, GENRES

retriever: OnnxDenseRetriever = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    retriever = OnnxDenseRetriever(
        csv_path="data/peliculas_CLEAN.csv",
        embeddings_path="data/embeddings.npy",
        model_dir="model/all-MiniLM-L6-v2",
    )
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4321", "https://tu-portfolio.vercel.app"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/genres")
def genres():
    return GENRES