from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.runnables import RunnableParallel,RunnableLambda,RunnableBranch
from pydantic import BaseModel,Field
from typing import Literal


model=ChatOllama(model="llama3")
parser=StrOutputParser()

class QueryAnalyser(BaseModel):
    queryType:Literal['Billing','Technical'] =Field(description="Sentiment of th feedback porvided")

parser2=PydanticOutputParser(pydantic_object=QueryAnalyser)


template=PromptTemplate(
    template="Analysis the type of the query user has asked. \n {Query}.\n {format_instruction}",
    input_variables=['Query'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

template2=PromptTemplate(
   template="Answer the question of the user for the billing issue.Also let him assure his issue related to bill be resolved. \n {Query}",
   input_variables=['Query'],
)

template3=PromptTemplate(
   template="Answer the question of the user for the Technical issue.Also let him assure him customer support exeutive will connect with as soon as possible for the techanical issue. \n {Query}",
   input_variables=['Query'],
)

query_chain=template|model|parser2

condiitonal_chain=RunnableBranch(
     (lambda x: x.queryType == 'Billing' , template2|model|parser),
     (lambda x: x.queryType == 'Technical' , template3|model|parser),
     (lambda x : "could not anlyse the type")
)

chain=query_chain|condiitonal_chain
result=chain.invoke("I was charged double the price of ticket")
print(result)