from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

model=ChatOllama(model="llama3")

template=ChatPromptTemplate.from_messages([
   ("system","You are a helpful assitant"),
    MessagesPlaceholder(variable_name="chat_History"),
   ("human"," {query}")
])

chat_History=[]
with open("chat_History.txt") as f:
                chat_History.extend(f.readlines())
                print(chat_History)

while True:
    query=input("Query : ")
    if query=="exit":
        break
    chain=template|model
    result=chain.invoke(
    {
        "chat_History":chat_History,
        "query":query
    }
)
    print(result.content)






