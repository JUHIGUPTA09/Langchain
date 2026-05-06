from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding_Model=HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

documents=[ 
"Kyoto, Japan, is a must-visit destination known for its stunning classical Buddhist temples, gardens, and traditional wooden houses.",
"The Serengeti National Park in Tanzania offers an unparalleled wildlife experience, especially during the famous Great Migration.",
"Machu Picchu in Peru remains one of the world's most iconic archaeological sites, nestled high in the Andes Mountains.",
"Paris, France, continues to captivate travelers with its world-class art, gastronomy, and the architectural beauty of the Eiffel Tower.",
"Santorini, Greece, is world-renowned for its breathtaking sunsets and the iconic blue-domed buildings overlooking the Aegean Sea."]


query="Tell me something about paris"

print("Starting embedding process...")
doc_embedding = embedding_Model.embed_documents(documents)
print("Document Embeddings generated successfully!")
#print(doc_embedding)

query_embedding=embedding_Model.embed_query(query) 
print("Query Embeddings generated successfully!")
#print(query_embedding)# Print just the first vector to verify
cosine_list=cosine_similarity([query_embedding],doc_embedding)[0]

print(cosine_list)
index,score=sorted(list(enumerate(cosine_list)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("Simlarty score = ",score)