from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field

model=ChatOllama(model="llama3")

class Person(BaseModel):
    name:str =Field(description="Name of a Person")
    age:int = Field(description="Age of a Person")
    Occupation:str = Field(description="Occupation of a Person")
    Address:str = Field(description="Address of a Person")

parser=PydanticOutputParser(pydantic_object=Person)


template=PromptTemplate(
    template="GIve me detail of a person working in {country}.\n {format_instruction}",
    input_variables=['country'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser
result=chain.invoke({
    'country':'India'
})

print(result)
