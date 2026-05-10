import os
from typing import List, TypedDict, Annotated
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==========================================
# 1. SETUP & DUMMY DATA
# ==========================================

# Initialize LLMs
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Create a dummy knowledge base for testing
texts = [
    "The company revenue for 2023 was $5 million.",
    "The main product is the 'AI-Toolbox' which helps developers.",
    "The CEO of the company is Alice Smith.",
    "The company was founded in 2020 in San Francisco.",
    "The 'AI-Toolbox' integrates with LangChain and OpenAI.",
    "The company has 50 employees as of 2023.",
    "The AI-Toolbox supports multiple programming languages including Python, JavaScript, and Java.",
    "Customer satisfaction rate is 95% according to recent surveys.",
    "The company plans to expand to Europe in 2024.",
    "R&D department occupies 30% of the company budget."
]

# Create Document objects
documents = [Document(page_content=text, metadata={"source": f"doc_{i}"}) for i, text in enumerate(texts)]

# Split texts
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# ==========================================
# 2. DEFINE STATE
# ==========================================

class AgentState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    loop_count: int

# ==========================================
# 3. DEFINE THE NODES
# ==========================================

def retrieve(state: AgentState):
    """Node: Retrieve documents from vector store."""
    print("--- RETRIEVING DOCUMENTS ---")
    question = state["question"]
    
    # Retrieve documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    documents = retriever.invoke(question)
    
    return {"documents": documents, "question": question}

def grade_documents(state: AgentState):
    """Node: Grade retrieved documents for relevance."""
    print("--- GRADING DOCUMENTS ---")
    question = state["question"]
    documents = state["documents"]
    
    # Grader LLM
    grader_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Grader prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a grader assessing relevance of a retrieved document to a user question. "
                    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. "
                    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])
    
    grader = prompt | grader_llm | StrOutputParser()
    
    # Filter documents
    filtered_docs = []
    for d in documents:
        score = grader.invoke({"question": question, "document": d.page_content})
        if "yes" in score.lower():
            print("--- GRADE: DOCUMENT RELEVANT ---")
            filtered_docs.append(d)
        else:
            print("--- GRADE: DOCUMENT NOT RELEVANT ---")

    return {"documents": filtered_docs, "question": question}

def rewrite_query(state: AgentState):
    """Node: Rewrite the query to improve retrieval."""
    print("--- REWRITING QUERY ---")
    question = state["question"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a question re-writer. Convert the input question to a better version that is optimized for vectorstore retrieval. \
                    Look at the input and try to reason about the underlying semantic intent / meaning."),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
    ])

    rewriter = prompt | llm | StrOutputParser()
    better_question = rewriter.invoke({"question": question})

    print(f"New Question: {better_question}")
    return {"question": better_question, "loop_count": state.get("loop_count", 0) + 1}

def generate(state: AgentState):
    """Node: Generate answer using the retrieved context."""
    print("--- GENERATING ANSWER ---")
    question = state["question"]
    documents = state["documents"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for question-answering tasks. \
                    Use the following pieces of retrieved context to answer the question. \
                    If you don't know the answer, just say that you don't know. \
                    Cite the source document content in your answer."),
        ("human", "Context: \n\n {documents} \n\n Question: {question}"),
    ])

    rag_chain = prompt | llm | StrOutputParser()
    generation = rag_chain.invoke({"documents": documents, "question": question})

    return {"documents": documents, "question": question, "generation": generation}

# ==========================================
# 4. DEFINE THE CONTROL FLOW (Edges)
# ==========================================

def decide_to_generate(state: AgentState):
    """Decision Function: Do we generate an answer, or rewrite the query?"""
    print("--- DECISION POINT ---")
    filtered_docs = state["documents"]
    loop_count = state.get("loop_count", 0)

    # Safety Mechanism: Prevent infinite loops (max 2 retries)
    if loop_count >= 2:
        print("--- MAX RETRIES REACHED: GENERATING BEST EFFORT ANSWER ---")
        return "generate"

    # If we have relevant docs, generate. If not, rewrite.
    if len(filtered_docs) > 0:
        return "generate"
    else:
        return "rewrite"

# ==========================================
# 5. BUILD THE GRAPH
# ==========================================

workflow = StateGraph(AgentState)

# Add the nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade", grade_documents) # The 'Grader' sits between retrieve and generate
workflow.add_node("generate", generate)
workflow.add_node("rewrite", rewrite_query)

# Set the entry point
workflow.set_entry_point("retrieve")

# Add the edges (The paths between nodes)
workflow.add_edge("retrieve", "grade") # After retrieval, always grade

# Conditional Edge: After grading, decide what to do
workflow.add_conditional_edges(
    "grade",
    decide_to_generate,
    {
        "generate": "generate",
        "rewrite": "rewrite"
    }
)

workflow.add_edge("rewrite", "retrieve") # If we rewrite, we loop back to retrieve
workflow.add_edge("generate", END)       # If we generate, we finish

# Compile the application
app = workflow.compile()

# ==========================================
# 6. RUN THE APPLICATION
# ==========================================

if __name__ == "__main__":
    # Test questions
    questions = [
        "What is the company's revenue?",
        "Who is the CEO?",
        "What products does the company offer?",
        "When was the company founded?",
        "How many employees work there?"
    ]
    
    print("🚀 Self-Correcting Agentic RAG System")
    print("=" * 50)
    
    for question in questions:
        print(f"\n📝 Question: {question}")
        print("-" * 30)
        
        # Initial state
        initial_state = {
            "question": question,
            "documents": [],
            "generation": "",
            "loop_count": 0
        }
        
        # Run the workflow
        result = app.invoke(initial_state)
        
        print(f"\n💡 Answer: {result['generation']}")
        print("=" * 50)
