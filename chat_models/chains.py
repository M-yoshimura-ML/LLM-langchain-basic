import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser

load_dotenv()

# template = PromptTemplate.from_template("""
# You are a cockney fruit and vegetable seller.
# Your role is to assist your customer with their fruit and vegetable needs.
# Respond using cockney rhyming slang.
#
# Tell me about the following fruit: {fruit}
# """)
template = PromptTemplate.from_template("""
You are a cockney fruit and vegetable seller.
Your role is to assist your customer with their fruit and vegetable needs.
Respond using cockney rhyming slang.

Output JSON as {{"description": "your response here"}}

Tell me about the following fruit: {fruit}
""")

llm = ChatOpenAI(
    openai_api_key=os.environ.get('OPENAI_API_KEY'),
    model="gpt-4o-mini",
    temperature=0
)

llm_chain = template | llm | StrOutputParser()

response = llm_chain.invoke({"fruit": "apple"})

print(response)

