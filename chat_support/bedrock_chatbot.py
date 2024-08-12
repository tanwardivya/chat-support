import base64
import random
from io import BytesIO
from typing import List, Tuple, Union, Dict

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.utilities import SerpAPIWrapper
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

from config import config as bedrock_config
from models.models import ChatModel
from bedrock_embedder import load_index, search_index

load_dotenv()

INIT_MESSAGE = {
    "role": "assistant",
    "content": "Hi! I'm your AI Bot on Bedrock to give knowledge on Arthritis. How may I help you?",
}

def set_page_config() -> None:
    """
    Set the Streamlit page configuration.
    """
    st.set_page_config(page_title="🤖 Chat with Arthritis Knowledge Base", layout="wide")
    st.title("🤖 Chat with Agent")

def rag_search(prompt: str) -> str:
    # Perform the search using the search_index function from bedrock_embedder.py
    docs = search_index(prompt)
    # Check if an error message was returned
    if isinstance(docs[0], str):
        return docs[0]
  
    # Format the results
    rag_content = "Here are the RAG search results: \n\n<search>\n\n" + "\n\n".join(doc.page_content for doc in docs) + "\n\n</search>\n\n"
    return rag_content + prompt

def init_runnablewithmessagehistory(system_prompt: str, chat_model: ChatModel) -> RunnableWithMessageHistory:
    """
    Initialize the RunnableWithMessageHistory with the given parameters.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        MessagesPlaceholder(variable_name="query"),
    ])

    chain = prompt | chat_model.llm

    msgs = StreamlitChatMessageHistory()

    # Create chain with history
    conversation = RunnableWithMessageHistory(
        chain,
        lambda session_id: msgs,
        input_messages_key="query",
        history_messages_key="chat_history"
    ) | StrOutputParser()

    # Store LLM generated responses
    if "messages" not in st.session_state:
        st.session_state.messages = [INIT_MESSAGE]

    return conversation

def display_user_message(message_content: Union[str, List[dict]]) -> None:
    """
    Display user message in the chat message.
    """
    if isinstance(message_content, str):
        message_text = message_content
    elif isinstance(message_content, dict):
        message_text = message_content["input"][0]["content"][0]["text"]
    else:
        message_text = message_content[0]["text"]

    message_content_markdown = message_text.split('</context>\n\n', 1)[-1]
    st.markdown(message_content_markdown)

def display_assistant_message(message_content: Union[str, dict]) -> None:
    """
    Display assistant message in the chat message.
    """
    if isinstance(message_content, str):
        st.markdown(message_content)
    elif "response" in message_content:
        st.markdown(message_content["response"])

def generate_response(
    conversation: RunnableWithMessageHistory, input: Union[str, List[dict]]
) -> str:
    """
    Generate a response from the conversation chain with the given input.
    """
    config = {"configurable": {"session_id": "streamlit_chat"}}

    generate_response_stream = conversation.stream(
        {"query": input},
        config=config
    )

    generate_response = st.write_stream(generate_response_stream)

    return generate_response

def display_chat_messages() -> None:
    """
    Display chat messages in the Streamlit app.
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):

            if message["role"] == "user":
                display_user_message(message["content"])

            if message["role"] == "assistant":
                display_assistant_message(message["content"])

def main() -> None:
    """
    Main function to run the Streamlit app.
    """
    set_page_config()
    model_name = "Claude 3 Haiku"
    model_kwargs = bedrock_config["models"][model_name]
    system_prompt = "You're helpful assistant."
    load_index()
    chat_model = ChatModel(model_name, model_kwargs)
    runnable_with_messagehistory = init_runnablewithmessagehistory(system_prompt, chat_model)
     # Display chat messages
    display_chat_messages()

    # User-provided prompt
    prompt = st.chat_input()
    if prompt:
        formatted_prompt = rag_search(prompt)
        st.session_state.messages.append({"role": "user", "content": formatted_prompt})
        with st.chat_message("user"):
            st.markdown(formatted_prompt)
    
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            response = generate_response(
                runnable_with_messagehistory, [{"role": "user",  "content": formatted_prompt}]
            )
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)


if __name__ == "__main__":
    main()