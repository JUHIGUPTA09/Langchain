from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddedModel=HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2",dimension=10)
vector=embeddedModel.embed_query("What is capital of delhi")
print(vector)


documents=[
"hello this is juhi"
"Welcome to delhi"
"Jai Hind"
]
vector=embeddedModel.embed_documents(documents)
print(vector)