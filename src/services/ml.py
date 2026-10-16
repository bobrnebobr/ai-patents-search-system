from src.clients.mlserver import MLServerClient, PatentClassifierClient
from src.clients.qdrant import PatentSearchClient
from src.schemas import PatentSearchResponse


class PatentSearchService:
    def __init__(
        self,
        encoder: MLServerClient,
        search: PatentSearchClient,
        classifier: PatentClassifierClient,
    ) -> None:
        self.encoder = encoder
        self.search_client = search
        self.classifier = classifier

    async def search(self, text: str, limit: int) -> PatentSearchResponse:
        vector = await self.encoder.encode(text)
        patents = await self.search_client.search(vector, limit)
        if not patents:
            return PatentSearchResponse(patents=[])

        patent_vectors = await self.encoder.encode_batch(
            [f"{patent.title}. {patent.abstract}" for patent in patents]
        )
        predictions = await self.classifier.classify(vector, patent_vectors)
        return PatentSearchResponse(
            patents=[
                patent.model_copy(
                    update={
                        "relevance_label": prediction.label,
                        "relevance_score": prediction.score,
                    }
                )
                for patent, prediction in zip(patents, predictions, strict=True)
            ]
        )
