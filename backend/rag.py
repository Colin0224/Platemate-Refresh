from __future__ import annotations

import logging

import chromadb
from google import genai

from .config import get_settings
from .seed_data import DOCUMENTS

logger = logging.getLogger(__name__)

_chroma_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None
_gemini_client: genai.Client | None = None


def _get_collection() -> chromadb.Collection:
    global _chroma_client, _collection
    if _collection is None:
        settings = get_settings()
        _chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        _collection = _chroma_client.get_or_create_collection(
            name="recipes",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        settings = get_settings()
        _gemini_client = genai.Client(api_key=settings.gemini_api_key)
    return _gemini_client


def seed_collection() -> None:
    collection = _get_collection()
    if collection.count() > 0:
        logger.info("ChromaDB already seeded (%d documents).", collection.count())
        return

    ids = [f"doc_{i}" for i in range(len(DOCUMENTS))]
    collection.add(
        documents=[d["text"] for d in DOCUMENTS],
        metadatas=[{"category": d["category"], "food": d["food"]} for d in DOCUMENTS],
        ids=ids,
    )
    logger.info("Seeded ChromaDB with %d recipe/nutrition documents.", len(DOCUMENTS))


def get_ingredients(food_name: str) -> str:
    client = _get_gemini_client()
    settings = get_settings()

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=(
            f"List only the main ingredients for {food_name}. "
            "Respond in a concise, comma-separated list without extra text."
        ),
    )
    return response.text.strip()


def get_healthier_alternatives(food_name: str) -> tuple[str, int]:
    collection = _get_collection()
    client = _get_gemini_client()
    settings = get_settings()

    results = collection.query(
        query_texts=[f"healthy alternative for {food_name}"],
        n_results=5,
    )

    context_docs = results["documents"][0] if results["documents"] else []
    context = "\n\n".join(context_docs)
    sources_used = len(context_docs)

    prompt = (
        "You are a knowledgeable nutritionist and chef. Use the provided context "
        "to give specific, actionable advice about healthier versions of foods. "
        "Include calorie comparisons and a brief recipe when possible. "
        "If the context doesn't cover the exact food, use relevant principles from it.\n\n"
        f"Context from nutrition knowledge base:\n\n{context}\n\n"
        f"What are healthier alternatives or modifications for {food_name}? "
        f"Include a brief healthy recipe suggestion."
    )

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
    )
    return response.text.strip(), sources_used
