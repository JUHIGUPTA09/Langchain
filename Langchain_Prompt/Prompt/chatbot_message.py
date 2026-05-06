from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage


model=ChatOllama(model="llama3")

chat_history=[

SystemMessage(content="You are helpful assitant")

]

while True:
    user_input=input("You :")
    chat_history.append(HumanMessage(content=user_input))
    if user_input=="exit":
        break
    else:
        result=model.invoke(chat_history)
        chat_history.append(AIMessage(result.content))
        print("AI : ",result.content)

print(chat_history)
        