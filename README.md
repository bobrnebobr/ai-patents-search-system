# AI Patent Radar

Асинхронный HTTP-сервис для проекта поиска и анализа патентов. На текущем этапе сервис
предоставляет инфраструктурные endpoints для проверки работоспособности приложения,
версии сборки и состояния PostgreSQL.

## Стек

- Python 3.12
- Litestar и Uvicorn
- PostgreSQL и asyncpg
- Pydantic Settings
- Loguru
- uv
- pytest и pytest-cov
- Ruff
- pre-commit
- Docker и Docker Compose
- GitHub Actions и GitHub Container Registry

## Структура проекта

```text
.
├── .github/workflows/cicd.yml  # CI/CD pipeline
├── src/
│   ├── api/                    # HTTP endpoints
│   ├── services/               # Проверка внешних компонентов
│   ├── app.py                  # Сборка Litestar-приложения
│   ├── config.py               # Настройки из переменных окружения
│   ├── logging_config.py       # Конфигурация логирования
│   └── schemas.py              # Схемы ответов
├── tests/                      # Тесты API
├── Dockerfile
├── Makefile                    # Короткие команды разработки
├── docker-compose.yml
├── pyproject.toml
└── uv.lock
```

API-слой отвечает за HTTP-интерфейс, а обращения к внешним компонентам вынесены в
сервисный слой. Обработчики и PostgreSQL-драйвер работают асинхронно и не блокируют
event loop во время сетевого ожидания.

## Требования

Для локальной разработки нужны:

- Python 3.12;
- [uv](https://docs.astral.sh/uv/);
- Docker с Docker Compose — для PostgreSQL и контейнерного запуска.

## Установка

Установить production- и dev-зависимости из lock-файла:

```bash
uv sync --frozen
```

Основные команды собраны в `Makefile`. Посмотреть полный список:

```bash
make help
```

Например, `make run` запускает приложение, `make test` — тесты, `make lint` — проверки
Ruff, а `make up` — приложение и PostgreSQL через Docker Compose.

Настройки читаются из переменных окружения и необязательного файла `.env`. Доступные
переменные перечислены в `.env.example`:

| Переменная | Назначение | Значение по умолчанию |
|---|---|---|
| `APP_VERSION` | Версия приложения | `0.1.0` |
| `APP_COMMIT_SHA` | SHA коммита сборки | `local` |
| `DATABASE_URL` | Строка подключения к PostgreSQL | локальная PostgreSQL |
| `LOG_LEVEL` | Уровень логирования | `INFO` |

## Запуск через Docker Compose

Запустить приложение вместе с PostgreSQL:

```bash
docker compose up --build
```

После запуска доступны:

- API: <http://localhost:8000>;
- PostgreSQL с хоста: `localhost:5433`;
- PostgreSQL внутри Compose-сети: `postgres:5432`.

Посмотреть логи:

```bash
docker compose logs -f app
```

Остановить сервисы:

```bash
docker compose down
```

Данные PostgreSQL сохраняются в именованном volume `postgres-data`. Чтобы вместе с
контейнерами удалить и локальные данные БД, можно явно выполнить:

```bash
docker compose down --volumes
```

## Локальный запуск

Поднять только PostgreSQL:

```bash
docker compose up -d postgres
```

При таком запуске база доступна на порту `5433`, поэтому перед запуском приложения нужно
задать строку подключения:

```bash
export DATABASE_URL=postgresql://postgres:postgres@localhost:5433/ai_patent_search
uv run uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload --no-access-log
```

## HTTP API

### `GET /healthz`

Быстрая liveness-проверка самого процесса приложения. Endpoint не обращается к базе
данных, поэтому не находится под версионируемым префиксом `/api/v1`.

```bash
curl http://localhost:8000/healthz
```

```json
{
  "status": "ok"
}
```

### `GET /api/v1/version`

Возвращает версию приложения и SHA коммита, из которого собран контейнер:

```bash
curl http://localhost:8000/api/v1/version
```

```json
{
  "version": "0.1.0",
  "commit_sha": "local"
}
```

При сборке в CI/CD вместо `local` передаётся настоящий `${{ github.sha }}`.

### `GET /api/v1/health`

Выполняет end-to-end проверку PostgreSQL и возвращает его версию и время ответа:

```bash
curl http://localhost:8000/api/v1/health
```

Пример успешного ответа:

```json
{
  "status": "ok",
  "dependencies": {
    "postgres": {
      "status": "ok",
      "response_time_ms": 4.2,
      "version": "16.4",
      "error": null
    }
  }
}
```

Если PostgreSQL недоступен, endpoint возвращает HTTP `503` и статус `degraded`.

Все HTTP-ответы содержат заголовок `X-Request-ID`, соответствующий идентификатору
запроса в логах.

## Тесты и покрытие

Запустить тесты:

```bash
uv run pytest
```

Параметры coverage заданы в `pyproject.toml`. Команда выводит непокрытые строки, создаёт
`coverage.xml` и завершается ошибкой, если общее покрытие ниже 85%.

Тестами покрыты:

- liveness endpoint;
- endpoint версии;
- успешная проверка PostgreSQL;
- degraded-сценарий недоступной БД;
- добавление `X-Request-ID`.

Интеграционный тест выполняет полный путь `HTTP → Litestar DI → asyncpg → PostgreSQL`.
Testcontainers автоматически запускает временный PostgreSQL и удаляет контейнер после
теста, поэтому вручную настраивать тестовую БД не нужно. Требуется только запущенный Docker:

```bash
uv run pytest -m integration
```

В GitHub Actions Testcontainers использует Docker runner и выполняет интеграционный тест
автоматически на каждом push.

## Линтер и форматирование

Проверить код:

```bash
uv run ruff check .
uv run ruff format --check .
```

Применить автоматические исправления и форматирование:

```bash
uv run ruff check . --fix
uv run ruff format .
```

Конфигурация Ruff включает pycodestyle, Pyflakes, isort, pyupgrade, проверки именования,
аннотаций типов и дополнительные правила Ruff.

## Pre-commit

Установить Git hook после клонирования репозитория:

```bash
uv run pre-commit install
```

Запустить все hooks вручную:

```bash
uv run pre-commit run --all-files
```

Перед коммитом выполняются Ruff, форматирование, проверки YAML/TOML/JSON, trailing
whitespace, конца файла, размера файлов, merge-конфликтов и поиск секретов через
Gitleaks.

## Docker-образ

Собрать образ локально с метаданными версии:

```bash
docker build \
  --build-arg APP_VERSION=0.1.0 \
  --build-arg APP_COMMIT_SHA=local \
  -t ai-patent-radar:local \
  .
```

Запустить его:

```bash
docker run --rm -p 8000:8000 \
  -e DATABASE_URL=postgresql://postgres:postgres@host.docker.internal:5433/ai_patent_search \
  ai-patent-radar:local
```

Production-образ устанавливает только runtime-зависимости и запускает приложение от
непривилегированного пользователя `appuser`.

## CI/CD

Workflow находится в `.github/workflows/cicd.yml`.

При любом push выполняется job `quality`:

1. установка зависимостей через `uv sync --frozen`;
2. линтинг;
3. проверка форматирования;
4. тесты с контролем покрытия;
5. проверка Docker Compose.

Только после успешного push в `main` запускается job `publish`, которая собирает образ и
отправляет его в GitHub Container Registry:

```text
ghcr.io/bobrnebobr/ai-patents-search-system
```

Один собранный образ публикуется с двумя тегами:

- `edge` — последняя успешная сборка ветки `main`;
- `sha-<commit>` — неизменяемая сборка конкретного коммита.

SHA также записывается внутрь образа и возвращается endpoint `/api/v1/version`.
Тег `sha-*` позволяет однозначно связать образ с исходным кодом и выполнить откат, а
`edge` предоставляет короткое имя последней успешной сборки.

Скачать опубликованный образ:

```bash
docker pull ghcr.io/bobrnebobr/ai-patents-search-system:edge
```

Workflow публикует образ в registry, но не разворачивает его на отдельном сервере.

## Логирование

Приложение пишет логи в стандартный вывод, что позволяет Docker собирать и ротировать
их. Access-log Uvicorn отключён, чтобы не дублировать запись из middleware. Для каждого
HTTP-запроса регистрируются:

- request ID;
- HTTP-метод и путь;
- статус ответа;
- длительность обработки;
- уровень и источник сообщения.

Пример:

```text
2025-09-09T21:33:01.123 | INFO | a1b2c3d4 | src.app | GET /healthz -> 200 (1.2 ms)
```

## Проверка перед отправкой изменений

```bash
uv run pre-commit run --all-files
uv run pytest
docker compose config -q
```

После успешного commit и push GitHub Actions выполняет CI-проверки. Push в `main`
дополнительно собирает и публикует новую версию контейнерного образа в GHCR.
