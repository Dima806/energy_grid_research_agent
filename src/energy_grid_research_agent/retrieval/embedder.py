from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from energy_grid_research_agent.config import get_settings
from energy_grid_research_agent.network_guard import assert_localhost


def build_vectorstore(corpus_dir: Path | None = None) -> Any:
    from langchain_chroma import Chroma
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_ollama import OllamaEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    settings = get_settings()
    assert_localhost(settings.ollama.base_url)

    corpus_path = Path(corpus_dir or settings.corpus.output_dir)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found at {corpus_path} — run `make generate` first.")

    loader = DirectoryLoader(str(corpus_path), glob="*.txt", loader_cls=TextLoader)
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} documents from {corpus_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.retrieval.chunk_size,
        chunk_overlap=settings.retrieval.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Split into {len(chunks)} chunks")

    embeddings = OllamaEmbeddings(
        base_url=settings.ollama.base_url,
        model=settings.ollama.embed_model,
    )

    vectorstore: Any = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=settings.chroma.persist_dir,
        collection_name=settings.chroma.collection_name,
    )
    logger.success(f"Stored {len(chunks)} chunks in Chroma at {settings.chroma.persist_dir}")
    return vectorstore


def load_vectorstore() -> Any:
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings

    settings = get_settings()
    assert_localhost(settings.ollama.base_url)

    embeddings = OllamaEmbeddings(
        base_url=settings.ollama.base_url,
        model=settings.ollama.embed_model,
    )
    return Chroma(
        persist_directory=settings.chroma.persist_dir,
        embedding_function=embeddings,
        collection_name=settings.chroma.collection_name,
    )


if __name__ == "__main__":
    build_vectorstore()
