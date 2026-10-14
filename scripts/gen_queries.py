#!/usr/bin/env python3
"""
Генерация датасета поисковых запросов для патентов через OpenAI-compatible LLM (litellm).

Переменные окружения:
  LLM_BASE_URL        адрес, напр. http://localhost:4000/v1       (обязательно)
  LLM_API_KEY         ключ                                         (обязательно)
  LLM_MODEL           имя модели                                   (обязательно)
  LLM_TEMPERATURE     по умолчанию 0.8
  LLM_MAX_TOKENS      по умолчанию 1024
  LLM_JSON_MODE       1 -> response_format={"type": "json_object"} (по умолчанию 0)
  MAX_TEXT_CHARS      сколько символов text отдавать в LLM (по умолчанию 6000)
  QUERY_LANGUAGE      "auto" (язык патента из source_language) или явно, напр. "English"

Запуск:
    export LLM_BASE_URL=http://localhost:port
    export LLM_API_KEY=sk-...
    export LLM_MODEL=model

  python gen_queries.py patents --input patents.parquet --output queries_patents.jsonl
  python gen_queries.py generic --input patents.parquet --output queries_generic.jsonl

  можно добавить флаги --limit 20 для пробного запуска
  по дефолту QUERY_LANGUAGE="English", нужно будет потом запустить еще с русским

Скрипт возобновляемый: уже обработанные публикации/CPC-классы пропускаются при перезапуске.
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

# ----------------------------------------------------------------------------
# Типы запросов: каждому из 5 слотов задаём свою роль, чтобы не получить
# 5 перефразировок одного и того же.
# ----------------------------------------------------------------------------
QUERY_TYPES = {
    "keywords": "Короткий ключевой запрос из 2-5 слов, как вводит инженер в строку поиска "
                "(без глаголов и связок, только ключевые термины).",
    "natural": "Запрос в виде естественного вопроса или описания задачи/проблемы, которую решает "
               "изобретение, простым языком, без патентного жаргона.",
    "technical": "Точный технический запрос со специфической терминологией: компоненты, материалы, "
                 "параметры, методы из патента. 4-10 слов.",
    "prior_art": "Развёрнутое описание сути изобретения в стиле профессионального патентного поиска "
                 "(prior art search), 15-30 слов, своими словами, не копируя название.",
    "layman": "Расплывчатый запрос неспециалиста: описывает назначение, область применения или "
              "бытовую ситуацию, не называя конкретной технологии. Можно использовать синонимы "
              "и разговорные выражения.",
}

PATENT_SYSTEM = (
    "Ты помогаешь строить обучающий датасет для системы патентного поиска. "
    "По описанию патента ты генерируешь реалистичные поисковые запросы, "
    "которые разные пользователи могли бы ввести, чтобы найти именно этот патент. "
    "Отвечай строго валидным JSON без пояснений и markdown."
)

PATENT_USER_TMPL = """Патент:
Название: {title}
Аннотация: {abstract}
CPC: {cpc}
Текст (фрагмент): {text}

Сгенерируй ровно {n} поисковых запросов на языке: {language}.
Каждый запрос должен иметь свой тип:
{types_desc}

Требования:
- Патент должен быть релевантным результатом для каждого запроса.
- Не копируй название дословно и не используй номера патентов, имена заявителей и CPC-коды.
- Запросы должны заметно отличаться друг от друга по лексике и формулировке.
- Не добавляй кавычки, операторы поиска и пояснения.

Формат ответа:
{{"queries": [{{"type": "<тип>", "query": "<текст запроса>"}}, ...]}}"""

GENERIC_SYSTEM = (
    "Ты эксперт по патентному поиску и знаешь, как пользователи формулируют запросы "
    "в патентных поисковых системах. Отвечай строго валидным JSON без пояснений и markdown."
)

GENERIC_USER_TMPL = """Область (CPC-класс {cpc}). Примеры названий патентов из этой области:
{titles}

Сгенерируй {n} разных запросов на языке: {language}, которые чаще всего вводят пользователи
патентных систем в этой области. Смешай стили: короткие ключевые запросы, вопросы на естественном
языке, технические формулировки, описания задач, поиск аналогов/prior art. Не повторяйся, не
копируй приведённые названия.

Формат ответа:
{{"queries": ["...", "..."]}}"""


# ----------------------------------------------------------------------------
# Утилиты
# ----------------------------------------------------------------------------
def env(name, default=None, required=False):
    val = os.environ.get(name, default)
    if required and not val:
        sys.exit(f"Не задана переменная окружения {name}")
    return val


def load_df(path: str) -> pd.DataFrame:
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".parquet":
        df = pd.read_parquet(p)
    elif suf == ".csv":
        df = pd.read_csv(p)
    elif suf in (".jsonl", ".ndjson"):
        df = pd.read_json(p, lines=True)
    elif suf == ".json":
        df = pd.read_json(p)
    else:
        sys.exit(f"Неизвестный формат файла: {suf}")
    df = df.loc[:, ~df.columns.duplicated()]  # на случай задвоенных колонок
    if "publication_id" in df.columns:
        df = df.drop_duplicates("publication_id")
    return df


def s(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return "; ".join(map(str, v))
    return str(v).strip()


def cpc_subclasses(v) -> list[str]:
    return list(dict.fromkeys(re.findall(r"[A-HY]\d{2}[A-Z]", s(v).upper().replace(" ", ""))))


def parse_json(raw: str):
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("JSON не найден в ответе")
    return json.loads(raw[start:end + 1])


def norm(q: str) -> str:
    return re.sub(r"\s+", " ", q).strip().strip('"«»')


class LLM:
    def __init__(self):
        self.client = AsyncOpenAI(
            base_url=env("LLM_BASE_URL", required=True),
            api_key=env("LLM_API_KEY", required=True),
        )
        self.model = env("LLM_MODEL", required=True)
        self.temperature = float(env("LLM_TEMPERATURE", "0.8"))
        self.max_tokens = int(env("LLM_MAX_TOKENS", "1024"))
        self.json_mode = env("LLM_JSON_MODE", "0") == "1"

    async def chat_json(self, system: str, user: str, retries: int = 4):
        kwargs = {}
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        last_err = None
        for attempt in range(retries):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **kwargs,
                )
                return parse_json(resp.choices[0].message.content or "")
            except Exception as e:  # сеть, rate limit, битый JSON
                last_err = e
                await asyncio.sleep(2 ** attempt + random.random())
        raise RuntimeError(f"LLM не ответила корректно: {last_err}")


def read_done(path: str, key: str) -> set:
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)[key])
                except Exception:
                    pass
    return done


# ----------------------------------------------------------------------------
# Режим 1: запросы по патентам
# ----------------------------------------------------------------------------
def build_patent_prompt(row) -> str:
    max_chars = int(env("MAX_TEXT_CHARS", "6000"))
    lang_cfg = env("QUERY_LANGUAGE", "auto")
    language = (s(row.get("source_language")) or "English") if lang_cfg == "auto" else lang_cfg
    types_desc = "\n".join(f"- {k}: {v}" for k, v in QUERY_TYPES.items())
    cpc = s(row.get("cpc_codes"))
    return PATENT_USER_TMPL.format(
        title=s(row.get("title")),
        abstract=s(row.get("abstract"))[:2000],
        cpc=cpc[:300],
        text=s(row.get("text"))[:max_chars],
        n=len(QUERY_TYPES),
        language=language,
        types_desc=types_desc,
    )


def validate_patent_queries(data: dict, title: str) -> list[dict]:
    items = data.get("queries")
    if not isinstance(items, list):
        raise ValueError("нет поля queries")
    by_type = {}
    for it in items:
        t, q = it.get("type"), norm(str(it.get("query", "")))
        if t in QUERY_TYPES and q and t not in by_type:
            by_type[t] = q
    missing = set(QUERY_TYPES) - set(by_type)
    if missing:
        raise ValueError(f"не хватает типов: {missing}")
    if len({q.lower() for q in by_type.values()}) < len(QUERY_TYPES):
        raise ValueError("есть дубликаты запросов")
    if any(q.lower() == title.lower() for q in by_type.values()):
        raise ValueError("запрос совпадает с названием")
    return [{"query_type": t, "query": by_type[t]} for t in QUERY_TYPES]


async def run_patents(args):
    df = load_df(args.input)
    done = read_done(args.output, "publication_id")
    df = df[~df["publication_id"].astype(str).isin({str(d) for d in done})]
    if args.limit:
        df = df.head(args.limit)
    llm = LLM()
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    out = open(args.output, "a", encoding="utf-8")
    failed = 0

    async def work(row):
        nonlocal failed
        async with sem:
            try:
                data = await llm.chat_json(PATENT_SYSTEM, build_patent_prompt(row))
                qs = validate_patent_queries(data, s(row.get("title")))
            except Exception as e:
                failed += 1
                print(f"[skip] {row.get('publication_id')}: {e}", file=sys.stderr)
                return
        async with lock:
            for q in qs:
                out.write(json.dumps({
                    "publication_id": s(row["publication_id"]),
                    "family_id": s(row.get("family_id")),
                    "query": q["query"],
                    "query_type": q["query_type"],
                    "source": "patent",
                }, ensure_ascii=False) + "\n")
            out.flush()

    rows = [r for _, r in df.iterrows()]
    await tqdm_asyncio.gather(*(work(r) for r in rows), desc="patents")
    out.close()
    print(f"Готово. Пропущено из-за ошибок: {failed}. Перезапустите, чтобы дообработать.")


# ----------------------------------------------------------------------------
# Режим 2: типичные запросы без привязки к патенту (по CPC-областям)
# ----------------------------------------------------------------------------
async def run_generic(args):
    df = load_df(args.input)
    counter = Counter()
    titles_by_cpc: dict[str, list[str]] = {}
    for _, r in df.iterrows():
        for c in cpc_subclasses(r.get("cpc_codes"))[:3]:
            counter[c] += 1
            titles_by_cpc.setdefault(c, []).append(s(r.get("title")))
    top = [c for c, _ in counter.most_common(args.top_cpc)]
    top = [c for c in top if c not in read_done(args.output, "cpc_subclass")]

    llm = LLM()
    lang_cfg = env("QUERY_LANGUAGE", "auto")
    language = "English" if lang_cfg == "auto" else lang_cfg
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    out = open(args.output, "a", encoding="utf-8")

    async def work(cpc):
        titles = random.sample(titles_by_cpc[cpc], min(5, len(titles_by_cpc[cpc])))
        prompt = GENERIC_USER_TMPL.format(
            cpc=cpc, titles="\n".join(f"- {t}" for t in titles),
            n=args.per_cpc, language=language)
        async with sem:
            try:
                data = await llm.chat_json(GENERIC_SYSTEM, prompt)
                qs = list(dict.fromkeys(norm(str(q)) for q in data["queries"] if str(q).strip()))
            except Exception as e:
                print(f"[skip] {cpc}: {e}", file=sys.stderr)
                return
        async with lock:
            for q in qs:
                out.write(json.dumps({"query": q, "cpc_subclass": cpc, "source": "generic"},
                                     ensure_ascii=False) + "\n")
            out.flush()

    await tqdm_asyncio.gather(*(work(c) for c in top), desc="generic")
    out.close()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("patents", "generic"):
        p = sub.add_parser(name)
        p.add_argument("--input", required=True, help="parquet/csv/jsonl/json с патентами")
        p.add_argument("--output", required=True, help="выходной jsonl")
        p.add_argument("--concurrency", type=int, default=8)
        if name == "patents":
            p.add_argument("--limit", type=int, default=0, help="обработать только N патентов (для проб)")
        else:
            p.add_argument("--top-cpc", type=int, default=300, help="сколько самых частых CPC-подклассов взять")
            p.add_argument("--per-cpc", type=int, default=20, help="запросов на один CPC-подкласс")
    args = ap.parse_args()
    asyncio.run(run_patents(args) if args.mode == "patents" else run_generic(args))


if __name__ == "__main__":
    main()