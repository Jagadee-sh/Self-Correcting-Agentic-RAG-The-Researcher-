# Self-Correcting-Agentic-RAG-The-Researcher-
An advanced Agentic RAG system that uses LangGraph to self-correct. It autonomously retrieves, grades, and rewrites queries to ensure high-quality answers from vector databases.

##Overview
This project implements an advanced Agentic Retrieval-Augmented Generation (RAG) system. Unlike standard RAG pipelines that simply retrieve documents and generate an answer, this system employs a "Reflection" pattern. It autonomously evaluates the quality of retrieved information and decides whether to generate an answer or rewrite the query and search again.

This architecture mimics a human researcher: searching, verifying sources, and refining the search query if the initial results are poor.

## 🏗️ Architecture
The application is built using LangGraph, allowing for cyclic workflows and state management.

###The Flow:

Retrieve: Fetch documents from the Vector Store (ChromaDB).
Grade: An LLM "Judge" evaluates if the documents are relevant to the question.
Decision:
Relevant? -> Generate answer.
Not Relevant? -> Rewrite query and loop back to Retrieve.
Safety Mechanism: A loop counter prevents the agent from getting stuck in an infinite retry cycle.

##🔑 Key Features
Self-Correction: The agent detects irrelevant retrieval and autonomously improves its query.
State Management: Uses TypedDict and LangGraph to maintain conversation history and loop counts.
Grading Logic: Uses an LLM to strictly grade documents on relevance before using them.
Modular Design: Nodes (Retrieve, Grade, Generate) are decoupled, making the system easy to extend.
##🛠️ Tech Stack
Orchestration: LangGraph (LangChain)
LLM: OpenAI GPT-3.5-turbo (or GPT-4)
Vector Store: ChromaDB
Embeddings: OpenAI Embeddings
Language: Python 3.10+

##🚀 Getting Started
1. Prerequisites
Python 3.10 or higher
An OpenAI API Key
2. Installation
Clone the repository:

git clone https://github.com/your-username/agentic-rag-researcher.gitcd agentic-rag-researcher
Install dependencies:

bash

pip install -r requirements.txt
3. Setup Environment Variables
Rename .env.example to .env and add your API key:

text

OPENAI_API_KEY=sk-your_actual_key_here
4. Run the Project
bash

python main.py

##💡 Future Improvements
Web Search Tool: Add a tool (like Tavily) to search the web if local documents are insufficient.
Streaming: Implement token streaming for real-time response generation.
UI: Add a Streamlit interface for easier interaction.
