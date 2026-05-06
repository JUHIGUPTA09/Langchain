from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema

model=ChatOllama(model="llama3")

response_schemas=[
    ResponseSchema(name="city1", description="Name of the city, weather, places to visit, and days required"),
    ResponseSchema(name="city2", description="Name of the city, weather, places to visit, and days required"),
    ResponseSchema(name="city3", description="Name of the city, weather, places to visit, and days required"),
]

parser=StructuredOutputParser.from_response_schemas(response_schemas)


template=PromptTemplate(
    template="GIve me ideal three ideal destiantion to visit in india in {country}.\n {format_instruction}",
    input_variables=['country'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser
result=chain.invoke({
    'country':'India'
})

print(result)
