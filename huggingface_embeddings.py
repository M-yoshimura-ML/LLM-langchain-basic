import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

os.environ.get('HF_TOKEN')

"""
Hugging Face sentence-transformer is a Python framework for state-of-the-art sentence, text and image embeddings.
One of the embedding models is used in the HuggingFaceEmbedding class. We have added an alias for 
SentenceTransformerEmbeddings for users who are more familiar with directly using that package.
"""

# this would cause error in local because of sentence-transformer
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "This is a test document."
query_result = embeddings.embed_query(text)
print(query_result)

