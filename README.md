---
title: PlateMate
emoji: 🍽️
colorFrom: green
colorTo: yellow
sdk: streamlit
sdk_version: "1.44.1"
app_file: app.py
pinned: false
short_description: AI food recognition with RAG-powered nutrition advice
---

# PlateMate — AI-Powered Culinary Assistant

PlateMate identifies foods from photos and provides nutritional insights using a **Retrieval-Augmented Generation (RAG)** pipeline. Upload or snap an image of your meal and get the detected food, its ingredients, and healthier alternatives backed by a curated nutrition knowledge base.

## Architecture

```
User (upload / camera)
        │
        ▼
┌────────────────────────────┐
│  HuggingFace Transformers  │  Food image classification (local)
│  food-image-classification │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│  Google Gemini 3 Flash     │  Ingredient generation
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│  ChromaDB vector search    │  Retrieve top-5 nutrition docs
│  + Gemini 3 Flash (RAG)   │  Generate grounded response
└────────────────────────────┘
```

## Tech Stack

| Layer         | Technology                                      |
|---------------|------------------------------------------------ |
| Frontend      | Streamlit                                       |
| ML Inference  | HuggingFace Transformers (image classification) |
| LLM           | Google Gemini 3 Flash                           |
| Vector DB     | ChromaDB (cosine similarity)                    |
| RAG Pipeline  | Retrieve from ChromaDB → augment prompt → Gemini|

## How the RAG Pipeline Works

1. **Indexing** — On startup, 49 curated nutrition and recipe documents are embedded and stored in ChromaDB.
2. **Retrieval** — When a user queries for healthier alternatives, the top-5 most semantically relevant documents are retrieved via cosine similarity.
3. **Generation** — Retrieved context is injected into a prompt sent to Gemini 3 Flash, producing a grounded response.
