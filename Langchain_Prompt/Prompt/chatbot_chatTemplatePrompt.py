from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_core.prompts import ChatPromptTemplate

model=ChatOllama(model="llama3")

template=ChatPromptTemplate.from_messages([
   ("system","You are expert in {domain}"),
   ("human","Explain me the {topic}")
])

while True:
    topic=input("Topic : ")
    if topic=="exit":
        break
    chain=template|model
    result=chain.invoke(
    {
        "domain":"Cricket",
        "topic":topic
    }
)
    print(result.content)






