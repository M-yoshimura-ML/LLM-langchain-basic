import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

memory = ChatMessageHistory()


def get_memory(session_id):
    return memory


chat_llm = ChatOpenAI(
    openai_api_key=os.environ.get('OPENAI_API_KEY'),
    model="gpt-4o-mini",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a surfer dude, having a conversation about the surf conditions on the beach. Respond using surfer slang.",
        ),
        ( "system", "{context}" ),
        ( "human", "{question}" ),
    ]
)

chat_chain = prompt | chat_llm | StrOutputParser()

current_weather = """
    {
        "surf": [
            {"beach": "Fistral", "conditions": "6ft waves and offshore winds"},
            {"beach": "Polzeath", "conditions": "Flat and calm"},
            {"beach": "Watergate Bay", "conditions": "3ft waves and onshore winds"}
        ]
    }"""

chat_with_message_history = RunnableWithMessageHistory(
    chat_chain,
    get_memory,
    input_messages_key="question",
    history_messages_key="chat_history"
)

response = chat_with_message_history.invoke(
    {
        "context": current_weather,
        "question": "Hi, I am at Watergate Bay. What is the surf like?"
    },
    config={"configurable": {"session_id": "none"}}
)
print(response)

response = chat_with_message_history.invoke(
    {
        "context": current_weather,
        "question": "Where I am?"
    },
    config={"configurable": {"session_id": "none"}}
)
print(response)
