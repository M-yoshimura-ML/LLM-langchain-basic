import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate

load_dotenv()

llm = ChatOpenAI(
    openai_api_key=os.environ.get('OPENAI_API_KEY')
)

graph = Neo4jGraph(
    url=os.environ.get('NEO4J_URL'),
    username=os.environ.get('NEO4J_USERNAME'),
    password=os.environ.get('NEO4J_PASSWORD')
)

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about movies and provide recommendations.
Convert the user's question based on the schema.

Schema: {schema}
Question: {question}
"""

CYPHER_GENERATION_TEMPLATE2 = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about movies and provide recommendations.
Convert the user's question based on the schema.

Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
For movie titles that begin with "The", move "the" to the end, For example "The 39 Steps" becomes "39 Steps, The" or "The Matrix" becomes "Matrix, The".

If no data is returned, do not attempt to answer the question.
Only respond to questions that require you to construct a Cypher statement.
Do not include any explanations or apologies in your responses.

Examples: 

Find movies and genres:
MATCH (m:Movie)-[:IN_GENRE]->(g)
RETURN m.title, g.name

Schema: {schema}
Question: {question}
"""

cypher_generation_prompt = PromptTemplate(
    template=CYPHER_GENERATION_TEMPLATE2,
    input_variables=["schema", "question"],
)

cypher_chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    cypher_prompt=cypher_generation_prompt,
    verbose=True
)

# cypher_chain.invoke({"query": "What is the plot of the movie The Green Mile?"})
# cypher_chain.invoke({"query": "How many movies has Tom Hanks directed?"})
cypher_chain.invoke({"query": "What movies has Tom Hanks directed and what are the genres ?"})
