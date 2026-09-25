from collections.abc import Callable

from litestar.di import Provide

from src.config import Settings

DependencyMap = dict[str, Provide]


def create_settings_provider(settings: Settings) -> Callable[[], Settings]:
    def provide_settings() -> Settings:
        return settings

    return provide_settings


def create_dependencies(settings: Settings) -> DependencyMap:
    return {
        "settings": Provide(
            create_settings_provider(settings),
            use_cache=True,
            sync_to_thread=False,
        ),
    }
