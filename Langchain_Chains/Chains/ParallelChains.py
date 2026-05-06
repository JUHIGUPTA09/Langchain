from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableLambda


model=ChatOllama(model="llama3")
parser=StrOutputParser()

text="""
The Scope of Artificial Intelligence
Artificial Intelligence is no longer a futuristic concept — it is the present. From the moment you unlock your phone with your face to the moment Netflix recommends your next binge, AI is quietly running the show.
The scope of AI today spans virtually every industry. In healthcare, AI models detect cancer from scans faster and more accurately than human radiologists. In finance, algorithms flag fraudulent transactions in milliseconds. In education, personalized learning platforms adapt content to each student's pace and style.
The software industry has been transformed most visibly. AI-powered coding assistants now write, review, and debug code alongside developers. What once took days of boilerplate work is now generated in seconds. This doesn't eliminate developers — it amplifies them.
Beyond white-collar work, AI is reshaping manufacturing through predictive maintenance, agriculture through precision farming, and transportation through autonomous vehicles. Even creative fields like music, art, and writing are being redefined by generative AI tools.
The numbers back this up. The global AI market is projected to exceed $1.8 trillion by 2030, growing at a rate few technologies in history have matched.
But scope isn't just about size — it's about depth. AI is moving from narrow task-specific tools to systems capable of reasoning, planning, and learning across domains. The transition from ANI (Artificial Narrow Intelligence) toward more general capabilities is already underway.
Simply put, there is no industry, no profession, and no corner of daily life that AI will leave untouched.
"""

template=PromptTemplate(
    template="Summzrize the article in as a {mode} ,Also mention I am summaring article in which mode . \n {text}",
    input_variables=['mode']
)

template1=PromptTemplate(
    template="Merge all the three summary and create a final output.{beginnerChain}.\n {medicoreChain}, \n {expertChain}",
    input_variables=['beginnerChain', 'medicoreChain', 'expertChain']
)

beginnerChain= RunnableLambda(lambda x :{'mode':'beginner','text':x['text']})| template|model|parser
medicoreChain= RunnableLambda(lambda x :{'mode':'medicore','text':x['text']})| template|model|parser
expertChain= RunnableLambda(lambda x :{'mode':'expert','text':x['text']})| template|model|parser

ParallelChain=RunnableParallel(
    {
        'beginnerChain':beginnerChain,
        'medicoreChain':medicoreChain,
        'expertChain':expertChain
    }
)

mergeChain= template1 |model|parser 

finalChain=ParallelChain|mergeChain

result=finalChain.invoke(
    {
        'text':text
    }
)
print(result)

