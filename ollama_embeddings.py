from langchain_community.embeddings import OllamaEmbeddings

embeddings = (
    OllamaEmbeddings(model="gemma:2b")  # by default it uses llama2
)

r1 = embeddings.embed_documents(
    [
        "Alpha is the first letter of Greek alphabet",
        "Beta is the second letter of Greek alphabet",
    ]
)

# print(r1)
# print(r1[0])

query_vector = embeddings.embed_query("what is the second letter of Greek alphabet ?")
# print(query_vector)


embeddings2 = OllamaEmbeddings(model="mxbai-embed-large")
text = "This is a test document."
query_result = embeddings2.embed_query(text)
print(query_result)
