import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()


# 1. Define the Endpoint
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="conversational", 
    max_new_tokens=512 
)

chat_model = ChatHuggingFace(llm=llm)

try:
    result = chat_model.invoke("What is the capital of India?")
    print(result.content)
except Exception as e:
    print(f"Error: {e}")


