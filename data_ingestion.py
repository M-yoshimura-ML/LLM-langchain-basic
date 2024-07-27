import bs4
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader, ArxivLoader, WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


loader = TextLoader('documents/speech.txt')
# print(loader)

text_documents = loader.load()
# print(text_documents)

speech = ""
with open("documents/speech.txt") as f:
    speech = f.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
text = text_splitter.create_documents([speech])
print(text[0])
print(text[1])

loader2 = PyPDFLoader('documents/attention.pdf')
print(loader2)

pdf_documents = loader2.load()
print(type(pdf_documents))
print(type(pdf_documents[0]))
# print(pdf_documents[0])

# Internally WebBaseLoader is using BeautifulSoup
loader3 = WebBaseLoader(web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
                        bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                            class_=("post-title", "post-content", "post-header")
                        ))
                        )
# print(loader3.load())

arxiv_docs = ArxivLoader(query="1605.08386", load_max_docs=2).load()
# print(arxiv_docs)

# attention is all you need
# https://arxiv.org/abs/1706.03762

wiki_docs = WikipediaLoader(query="Hunter X Hunter", load_max_docs=2).load()
# print(wiki_docs)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# create_docs = text_splitter.create_documents(wiki_docs)
split_docs = text_splitter.split_documents(wiki_docs)
print(split_docs[0])
print(split_docs[1])
print(split_docs[2])
