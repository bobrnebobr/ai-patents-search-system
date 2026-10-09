from src.clients.mlserver import MLServerClient
from src.clients.qdrant import PatentSearchClient
from src.schemas import PatentSearchResponse


class PatentSearchService:
    def __init__(self, encoder: MLServerClient, search: PatentSearchClient) -> None:
        self.encoder = encoder
        self.search_client = search

    async def search(self, text: str, limit: int) -> PatentSearchResponse:
        vector = await self.encoder.encode(text)
        patents = await self.search_client.search(vector, limit)
        return PatentSearchResponse(patents=patents)
