import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os    
import datetime
from typing import Any, cast
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult, RAGQueryResult

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_api",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="RAG: Inngest PDF",
    trigger=inngest.TriggerEvent(event="rag/inngest_pdf")
)
async def rag_inngest_pdf(ctx: inngest.Context):
    async def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = str(ctx.event.data['pdf_path'])
        source_id = str(ctx.event.data.get('source_id', pdf_path))
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    async def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks= chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{source_id}:{i}")) for i in range(len(vecs))]
        payloads = [{'text': chunks[i], 'source': source_id} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(inngested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    inngested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return inngested.model_dump()

@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
) 
async def rag_query_pdf_ai(ctx: inngest.Context):
    async def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k)
        return RAGSearchResult(context=found['context'], sources=found['sources'])
    
    question = str(ctx.event.data['question'])
    top_k_value = ctx.event.data.get('top_k', 5)
    top_k = int(top_k_value) if isinstance(top_k_value, (int, float, str)) else 5

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    context_block = "\n\n".join(f"- {item}" for item in found.context)
    print("\n" + "="*40)
    print(f"WHAT QDRANT FOUND:\n{context_block}")
    print("="*40 + "\n")
    user_content = (
        "Use the following retrieved context to answer the question.\n\n"
        f"Context: {context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer consisely and use only the provided context. Do not include any information that is not in the context."
    )

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY is not set")

    adapter = ai.openai.Adapter(
        auth_key=openai_api_key,
        model="gpt-4o-mini"
    )

    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "You answer questions based on the provided context."},
                {"role": "user", "content": user_content}
            ]
        }
    )

    res_dict = cast(dict[str, Any], res)
    answer = str(res_dict.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
    return {'answer': answer, 'sources': found.sources, 'num_contexts': len(found.context)}
    
app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_inngest_pdf, rag_query_pdf_ai])