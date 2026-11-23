from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

embedding_model = HuggingFaceEmbeddings(model = "sentence-transformers/all-mpnet-base-v2")

vectorstore = Chroma(
    embedding_function=embedding_model,
    persist_directory= "./vector_db"
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":2, #Number of documents to retrieve
        "fetch_k": 10, #Number of documents to fetch
        "lambda_mult": 0.5#Diversity (0-Diverse, 1- Less Diverse)
    }
)

#model
model = init_chat_model(
    "openai/gpt-oss-20b",
    model_provider="groq"
)
#model

#prompt-template
chat_template = ChatPromptTemplate.from_messages([
    ("system",
    """You are a helpful AI assistant, who is an expert in answering questions about 
    the given context.
    
    You are given the following context: {context}
    
    Please answer the following question based on the context provided.
    Use ONLY the provided context to answer the question.
    
    If the answer is not present in the context,
    say "I don't know", without making up any information. Do not make assumptions and answer 
    only what is asked.
    """),
    ("human","""Context:{context}
    Question:{question}
    """)
])
print("RAG created.")
#prompt-templates

print("Press 0 to exit.")
while True: 
    query = input("You: ")
    if query == "0": 
        break 
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = chat_template.invoke({
        "context" : context,
        "question" : query
    })
    response = model.invoke(final_prompt)
    print("AI: ", response.text)