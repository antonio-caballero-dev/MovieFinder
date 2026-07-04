from pydantic import BaseModel

class MovieResult(BaseModel):
    id: int
    title: str
    release_date: str
    overview: str
    genres: list[str]
    similarity: float
    poster_url: str | None

class SearchResponse(BaseModel):
    results: list[MovieResult]
    query_time_ms: float