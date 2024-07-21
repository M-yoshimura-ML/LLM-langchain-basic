import os

from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph

load_dotenv()

graph = Neo4jGraph(
    url=os.environ.get('NEO4J_URL'),
    username=os.environ.get('NEO4J_USERNAME'),
    password=os.environ.get('NEO4J_PASSWORD')
)

result = graph.query("""
MATCH (m:Movie {title: 'Toy Story'})
RETURN m.title, m.plot, m.poster
""")

print(result)

# access to schema info
print(graph.schema)
