import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain.schema import Document

load_dotenv()

# A list of Documents
documents = [
    Document(
        page_content="Text to be indexed",
        metadata={"source": "local"}
    )
]

# Service used to create the embeddings
embedding_provider = OpenAIEmbeddings(
    openai_api_key=os.environ.get('OPENAI_API_KEY')
)

graph = Neo4jGraph(
    url=os.environ.get('NEO4J_URL'),
    username=os.environ.get('NEO4J_USERNAME'),
    password=os.environ.get('NEO4J_PASSWORD')
)

new_vector = Neo4jVector.from_documents(
    documents,
    embedding_provider,
    graph=graph,
    index_name="myVectorIndex",
    node_label="Chunk",
    text_node_property="text",
    embedding_node_property="embedding",
    create_id_index=True,
)

