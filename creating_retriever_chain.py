import os

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector

load_dotenv()

llm = ChatOpenAI(openai_api_key=os.environ.get('OPENAI_API_KEY'))

embedding_provider = OpenAIEmbeddings(openai_api_key=os.environ.get('OPENAI_API_KEY'))

graph = Neo4jGraph(
    url=os.environ.get('NEO4J_URL'),
    username=os.environ.get('NEO4J_USERNAME'),
    password=os.environ.get('NEO4J_PASSWORD')
)

movie_plot_vector = Neo4jVector.from_existing_index(
    embedding_provider,
    graph=graph,
    index_name="moviePlots",
    embedding_node_property="plotEmbedding",
    text_node_property="plot",
)

plot_retriever = RetrievalQA.from_llm(
    llm=llm,
    retriever=movie_plot_vector.as_retriever()
)

response = plot_retriever.invoke(
    {"query": "A movie where a mission to the moon goes wrong"}
)

print(response)
