from langchain_ollama import ChatOllama
from typing import TypedDict

model=ChatOllama(model="llama3.1")

class Employe_Profile(TypedDict):
    name:str
    age:int 
    dept:str 
    skills:list[str]
    experience:int 
    is_manager:bool



prompt="John is a 32-year-old backend engineer in the Platform team with 8 years of experience. He knows Java, Kafka, and Kubernetes. He leads a team of 5."

structured_model=model.with_structured_output(Employe_Profile)

result=structured_model.invoke(prompt)
print(result)
print("is_manager",result['is_manager'])
