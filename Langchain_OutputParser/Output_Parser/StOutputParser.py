from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

model=ChatOllama(model="llama3")
parser=StrOutputParser()

template1=PromptTemplate(
    template="Tell me detailed analysis on {topic}",
    input_variables=['topic']
)

template2=PromptTemplate(
    template="Give me summary in 5 lins for the paragraph. \n   {paragraph}",
    input_variables=['paragraph']
)


chain=template1 | model | parser | RunnableLambda (lambda x :  {"paragraph" : x }) |template2 |model |parser
result=chain.invoke(
    {'topic':'Black Hole'}
)
print(result)