import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# Ensure this matches your .env key exactly
#huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# 1. Define the Endpoint
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    # Stick to text-generation here; the Wrapper will handle the 'chat' part
    task="conversational", 
    #huggingfacehub_api_token=huggingfacehub_api_token,
    # Adding this helps with provider-specific quirks
    max_new_tokens=512 
)

# 2. Wrap it in ChatHuggingFace
# This translates your string into the "Chat" format the provider expects
chat_model = ChatHuggingFace(llm=llm)

# 3. Invoke
try:
    result = chat_model.invoke("What is the capital of India?")
    print(result.content)
except Exception as e:
    print(f"Error: {e}")


