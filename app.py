"""Streamlit frontend for the local FastAPI search service."""

from __future__ import annotations

import os

import requests
import streamlit as st


API_URL = os.getenv("SEMANTIC_SEARCH_API", "http://localhost:8000")


def api_get(path: str):
    return requests.get(f"{API_URL}{path}", timeout=10)


def api_post(path: str, **kwargs):
    return requests.post(f"{API_URL}{path}", timeout=120, **kwargs)


st.set_page_config(page_title="Semantic Media Search", layout="wide")
st.title("Semantic Media Search")

with st.sidebar:
    st.header("Upload media")
    uploaded = st.file_uploader("Image or audio", type=["jpg", "jpeg", "png", "wav", "mp3"])
    if uploaded is not None and st.button("Upload and index"):
        response = api_post("/upload", files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)})
        if response.ok:
            st.success(f"Indexed {response.json()['filename']}")
        else:
            st.error(response.json().get("detail", response.text))

    st.header("Index stats")
    try:
        stats = api_get("/stats").json()
        st.metric("Images indexed", stats.get("images_indexed", 0))
        st.metric("Audio indexed", stats.get("audio_indexed", 0))
    except requests.RequestException:
        st.warning("API is unavailable. Start FastAPI with uvicorn first.")

query = st.text_input("Search query", placeholder="a dog running on grass")
media_label = st.radio("Media type", ["Image", "Audio", "Both"], horizontal=True)
media = media_label.lower()
k = st.slider("Results", min_value=1, max_value=10, value=5)

if st.button("Search", type="primary"):
    if not query.strip():
        st.warning("Enter a search query.")
    else:
        response = api_post("/search", json={"query": query, "media": media, "k": k})
        if not response.ok:
            st.error(response.json().get("detail", response.text))
        else:
            payload = response.json()
            if media in {"image", "both"}:
                st.subheader("Image Results")
                image_results = payload.get("image_results") or []
                columns = st.columns(min(3, max(1, len(image_results))))
                for position, result in enumerate(image_results):
                    with columns[position % len(columns)]:
                        path = result["path"]
                        image_url = path if path.startswith("http") else f"{API_URL}{path}"
                        st.image(image_url, caption=result["filename"])
                        st.caption(f"Similarity: {result['score']:.4f}")
            if media in {"audio", "both"}:
                st.subheader("Audio Results")
                for result in payload.get("audio_results") or []:
                    path = result["path"]
                    audio_url = path if path.startswith("http") else f"{API_URL}{path}"
                    st.audio(audio_url)
                    st.caption(f"{result['filename']} — similarity: {result['score']:.4f}")
