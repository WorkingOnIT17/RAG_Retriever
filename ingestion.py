from dotenv import load_dotenv
load_dotenv()
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


#Loading the Document
data = PyPDFLoader("/Users/tanmaynegi/Downloads/book.pdf")
docs = data.load()
#Loading the Document

#Chunking 
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " "]
)

chunks = splitter.split_documents(docs)
#chunking 

embeddingmodel = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")

vectorstore= Chroma.from_documents(
    documents=chunks,
    embedding=embeddingmodel,
    persist_directory= "./vector_db"
)

print(" vector_db created successfully")
