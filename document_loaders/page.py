from langchain_community.document_loaders import WebBaseLoader
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

url = "https://en.wikipedia.org/wiki/Generative_pre-trained_transformer"
loader = WebBaseLoader(url)
docs = loader.load()

model = init_chat_model(
    "openai/gpt-oss-20b",
    model_provider="groq"
)

# model = init_chat_model(
#     "gemini-3.5-flash-lite",
#     model_provider="google_genai"
# )
# template = ChatPromptTemplate.from_messages([
#     ("system", "You are supposed to create a summarized version of the webpage that is provided to you."),
#     ("user", "{webpage}")
# ])
# prompt = template.format(webpage = docs)
# response = model.invoke(prompt)
# print(response.text)


response = model.invoke("tell a funny joke")
print(response.text)