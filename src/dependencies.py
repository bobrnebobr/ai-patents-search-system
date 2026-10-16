from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, asynccontextmanager

import httpx
from litestar import Litestar
from litestar.datastructures import State
from litestar.di import NamedDependency, Provide
from qdrant_client import AsyncQdrantClient

from src.clients.mlserver import MLServerClient, PatentClassifierClient
from src.clients.qdrant import PatentSearchClient
from src.config import Settings
from src.services.ml import PatentSearchService

DependencyMap = dict[str, Provide]


@asynccontextmanager
async def ml_clients_lifespan(app: Litestar) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    async with AsyncExitStack() as stack:
        http = await stack.enter_async_context(
            httpx.AsyncClient(base_url=settings.mlserver_url, timeout=settings.mlserver_timeout)
        )
        qdrant = AsyncQdrantClient(
            url=settings.qdrant_url,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key.get_secret_value() or None,
            timeout=settings.qdrant_timeout + 10,
            check_compatibility=False,
        )
        stack.push_async_callback(qdrant.close)
        app.state.encoder_client = MLServerClient(
            http, settings.encoder_model_name, batch_size=settings.encoder_batch_size
        )
        app.state.classifier_client = PatentClassifierClient(http, settings.classifier_model_name)
        app.state.patent_search_client = PatentSearchClient(
            qdrant, settings.qdrant_collection.get_secret_value(), settings.qdrant_timeout
        )
        yield


def provide_encoder_client(state: State) -> MLServerClient:
    return state.encoder_client


def provide_patent_search_client(state: State) -> PatentSearchClient:
    return state.patent_search_client


def provide_classifier_client(state: State) -> PatentClassifierClient:
    return state.classifier_client


def provide_patent_search_service(
    encoder_client: NamedDependency[MLServerClient],
    patent_search_client: NamedDependency[PatentSearchClient],
    classifier_client: NamedDependency[PatentClassifierClient],
) -> PatentSearchService:
    return PatentSearchService(encoder_client, patent_search_client, classifier_client)


def create_settings_provider(settings: Settings) -> Callable[[], Settings]:
    def provide_settings() -> Settings:
        return settings

    return provide_settings


def create_dependencies(settings: Settings) -> DependencyMap:
    return {
        "encoder_client": Provide(provide_encoder_client, sync_to_thread=False),
        "patent_search_client": Provide(provide_patent_search_client, sync_to_thread=False),
        "classifier_client": Provide(provide_classifier_client, sync_to_thread=False),
        "patent_search_service": Provide(provide_patent_search_service, sync_to_thread=False),
        "settings": Provide(
            create_settings_provider(settings),
            use_cache=True,
            sync_to_thread=False,
        ),
    }
