from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import numpy as np 
from langchain_core.documents import Document 


# docs = [
#     Document(
#         page_content=(
#             "HNSW builds a multi-layer proximity graph for fast Approximate Nearest Neighbor (ANN) "
#             "search. Upper layers provide long-range jumps while lower layers refine local neighbors."
#         ),
#         metadata={
#             "source": "vector_indexes_guide.md",
#             "topic": "vector_search",
#             "section": "HNSW",
#             "year": 2026,
#         },
#     ),
#     Document(
#         page_content=(
#             "IVF partitions embedding space into clusters using k-means and searches only the top "
#             "nprobe nearest centroids, while PQ compresses vectors into compact sub-vector codes to "
#             "reduce memory usage."
#         ),
#         metadata={
#             "source": "vector_indexes_guide.md",
#             "topic": "vector_search",
#             "section": "IVF-PQ",
#             "year": 2026,
#         },
#     ),
#     Document(
#         page_content=(
#             "Splitting is the mechanical slicing of text by token count, character count, or delimiters "
#             "with optional overlap, whereas chunking is the strategy of preserving semantic coherence "
#             "and context per retrieval unit."
#         ),
#         metadata={
#             "source": "rag_chunking_notes.md",
#             "topic": "rag",
#             "section": "Chunking vs Splitting",
#             "year": 2026,
#         },
#     ),
#     Document(
#         page_content=(
#             "Sentence Transformers encode text into dense semantic vectors—such as 768-dimensional "
#             "embeddings from all-mpnet-base-v2—that can run locally without an API key for public "
#             "models, whereas HF_TOKEN is needed for gated repos or hosted inference."
#         ),
#         metadata={
#             "source": "embeddings_and_auth.md",
#             "topic": "embeddings",
#             "model": "sentence-transformers/all-mpnet-base-v2",
#             "dimensions": 768,
#             "year": 2026,
#         },
#     ),
#     Document(
#         page_content=(
#             "Calling a native SDK directly gives tight control over provider features, token usage, and "
#             "debugging, while frameworks like LangChain or LangGraph help when you need multi-step "
#             "state, tool orchestration, evaluations, and production guardrails."
#         ),
#         metadata={
#             "source": "ai_architecture_tradeoffs.md",
#             "topic": "architecture",
#             "section": "SDK vs Framework",
#             "year": 2026,
#         },
#     ),
# ]

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# vector_store = Chroma.from_documents(
#     documents=docs, 
#     embedding=embedding_model, 
#     persist_directory="./chroma_db"
# )

persist_dir = "/Users/tanmaynegi/codes/GenAILearning/rag_pro/vector_store/chroma_db"
vector_store = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)


# res = vector_store.similarity_search("What is graph?",k = 2)
# for i in res:
#     print()
#     print()
#     print(i.page_content)
#     print()
#     print()


retreiver = vector_store.as_retriever(search_type="mmr",search_kwargs={"k":2,"fetch_k":4})
docs = retreiver.invoke("What is HNSW?")
for i in docs:
    print()
    print()
    print(i.page_content)
    print()
    print()
