import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("document_loaders/sample_1.pdf")
docs = data.load()
print(docs[2].page_content)