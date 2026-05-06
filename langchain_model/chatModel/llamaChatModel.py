from langchain_community.chat_models import ChatOllama

model=ChatOllama(model="llama3")
result=model.invoke("What is capital of India")
print(result)