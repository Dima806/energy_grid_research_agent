from __future__ import annotations

from functools import lru_cache
from typing import Any

from energy_grid_research_agent.config import get_settings
from energy_grid_research_agent.network_guard import assert_localhost


@lru_cache(maxsize=1)
def get_retrieval_chain() -> Any:
    from langchain_classic.chains import RetrievalQA
    from langchain_ollama import OllamaLLM as Ollama

    from energy_grid_research_agent.retrieval.embedder import load_vectorstore

    settings = get_settings()
    assert_localhost(settings.ollama.base_url)

    vectorstore = load_vectorstore()
    llm = Ollama(base_url=settings.ollama.base_url, model=settings.ollama.llm_model)

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": settings.retrieval.top_k}),
        return_source_documents=True,
    )


def query_chain(query: str) -> dict[str, Any]:
    chain = get_retrieval_chain()
    result: dict[str, Any] = chain.invoke({"query": query})
    return result
