from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser


model=ChatOllama(model="llama3")
parser=JsonOutputParser()

template1=PromptTemplate(
    template="GIve me ideal three ideal destiantion to visit in india in June along with price of trip and night also famouse destination.\n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain= template1 | model |parser
result=chain.invoke({})
print(result)