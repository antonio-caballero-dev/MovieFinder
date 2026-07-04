import time
from fastapi import APIRouter, Query
from api.models import SearchResponse
import api.main as state

router = APIRouter()

@router.get("/search", response_model=SearchResponse)
def search(
    query: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=10),
    genre_id: int | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
):
    t0 = time.time()
    results = state.retriever.search(query, top_k, genre_id, start_year, end_year)
    return SearchResponse(results=results, query_time_ms=(time.time() - t0) * 1000)