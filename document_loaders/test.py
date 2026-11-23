import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)

from langchain_community.document_loaders import TextLoader

path = "notes.txt"


data = TextLoader(path)
docs = data.load()

print((docs[0]).page_content)
