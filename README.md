# Chat Support for Arthritis Knowledge Base with streaming support

## Idea
This is a Chat Support system for an Arthritis knowledge base designed to assist users in understanding symptoms, prevention strategies, and whom to contact. It also provides information on the different types of arthritis. The Chat Support uses Bedrock, LangChain, Claude 3, and Sonnet models to build an AI agent deployed using Retrieval Augmented Generation (RAG) techniques. This approach ensures that the chatbot can provide accurate and contextually relevant information by retrieving and generating responses based on a vast knowledge base.

## Tech Stack

- **Python >= 3.11**
- **Docker**
- **EC2**
- **LangChain**
- **AWS S3 Storage**
- **Streamlit**
- **Amazon Bedrock**

## Steps to create Agent

**Upload File for Creating Vector Store:**

Begin by uploading the necessary file(s) to create a Vector Store, which will be used for efficient information retrieval.

**Store Index File in S3:**

After creating the Vector Store, the index file generated needs to be stored in an S3 bucket for easy access and scalability.

**Load Index File:**

Load the index file from S3 when initializing the agent, ensuring that the agent has access to the pre-processed data.

**Add Context from Index with User Prompt:**

Integrate the context retrieved from the index with the user’s prompt before sending it to the Language Model (LLM) for a more informed and accurate response.

**Add Streamlit Chat History Buffer:**

Implement a Streamlit chat history buffer to preserve the chat history, allowing for continuity in conversations and a better user experience.


## How to Install
To install the Chat Support system, you need to pull the Docker image from Docker Hub. Use the following command:

```bash
aws configure
docker pull divyatanwar/agent:genai-agent-rag
```

## How to Run
```bash
docker run  -v ~/.aws:/root/.aws -p 8501:8501 -it divyatanwar/agent:genai-agent-rag
```

