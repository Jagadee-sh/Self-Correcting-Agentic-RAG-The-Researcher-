import os
from typing import List, TypedDict
from langchain_core.documents import Document

# ==========================================
# 1. SETUP & DUMMY DATA
# ==========================================

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

# ==========================================
# 2. DEFINE STATE
# ==========================================

class AgentState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    loop_count: int

# ==========================================
# 3. DEFINE THE NODES (SIMULATED)
# ==========================================

def retrieve(state: AgentState):
    """Node: Retrieve documents from vector store (simulated)."""
    print("--- RETRIEVING DOCUMENTS ---")
    question = state["question"].lower()
    
    # Simple keyword-based retrieval simulation
    relevant_docs = []
    keywords = {
        "revenue": ["revenue", "million", "2023"],
        "ceo": ["ceo", "alice", "smith"],
        "product": ["product", "ai-toolbox", "developers"],
        "founded": ["founded", "2020", "san francisco"],
        "employees": ["employees", "50", "2023"],
        "languages": ["python", "javascript", "java"],
        "satisfaction": ["satisfaction", "95%", "surveys"],
        "expansion": ["expand", "europe", "2024"],
        "budget": ["budget", "r&d", "30%"]
    }
    
    # Find relevant documents based on keywords
    for doc in documents:
        for key, kw_list in keywords.items():
            if any(kw in question for kw in kw_list):
                if doc not in relevant_docs:
                    relevant_docs.append(doc)
                break
    
    # If no specific keywords found, return all documents
    if not relevant_docs:
        relevant_docs = documents[:3]  # Return first 3 as fallback
    
    return {"documents": relevant_docs, "question": question}

def grade_documents(state: AgentState):
    """Node: Grade retrieved documents for relevance (simulated)."""
    print("--- GRADING DOCUMENTS ---")
    question = state["question"]
    documents = state["documents"]
    
    # Simple grading simulation - keep all documents for demo
    filtered_docs = []
    for d in documents:
        print("--- GRADE: DOCUMENT RELEVANT ---")
        filtered_docs.append(d)

    return {"documents": filtered_docs, "question": question}

def rewrite_query(state: AgentState):
    """Node: Rewrite the query to improve retrieval (simulated)."""
    print("--- REWRITING QUERY ---")
    question = state["question"]
    
    # Simple query rewriting simulation
    question_lower = question.lower()
    if "what" in question_lower:
        better_question = question.replace("What", "Tell me about")
    elif "who" in question_lower:
        better_question = question.replace("Who", "Information about")
    elif "when" in question_lower:
        better_question = question.replace("When", "Date of")
    else:
        better_question = question
    
    print(f"New Question: {better_question}")
    return {"question": better_question, "loop_count": state.get("loop_count", 0) + 1}

def generate(state: AgentState):
    """Node: Generate answer using the retrieved context (simulated)."""
    print("--- GENERATING ANSWER ---")
    question = state["question"]
    documents = state["documents"]
    
    # Simple answer generation based on document content
    question_lower = question.lower()
    
    if "revenue" in question_lower:
        answer = "Based on the retrieved documents, the company's revenue for 2023 was $5 million."
    elif "ceo" in question_lower:
        answer = "According to the documents, the CEO of the company is Alice Smith."
    elif "product" in question_lower:
        answer = "The main product is the 'AI-Toolbox' which helps developers and integrates with LangChain and OpenAI."
    elif "founded" in question_lower:
        answer = "The company was founded in 2020 in San Francisco."
    elif "employees" in question_lower:
        answer = "According to the information, the company has 50 employees as of 2023."
    elif "languages" in question_lower or "programming" in question_lower:
        answer = "The AI-Toolbox supports multiple programming languages including Python, JavaScript, and Java."
    elif "satisfaction" in question_lower:
        answer = "Customer satisfaction rate is 95% according to recent surveys."
    elif "expansion" in question_lower:
        answer = "The company plans to expand to Europe in 2024."
    elif "budget" in question_lower or "r&d" in question_lower:
        answer = "R&D department occupies 30% of the company budget."
    else:
        answer = f"Based on the retrieved documents, here's what I found: {documents[0].page_content if documents else 'No relevant information found.'}"

    return {"documents": documents, "question": question, "generation": answer}

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
# 5. BUILD THE GRAPH (SIMULATED WORKFLOW)
# ==========================================

def run_agentic_rag(question: str):
    """Simulate the agentic RAG workflow."""
    print(f"\n🚀 Self-Correcting Agentic RAG System")
    print("=" * 50)
    print(f"📝 Question: {question}")
    print("-" * 30)
    
    # Initial state
    state = {
        "question": question,
        "documents": [],
        "generation": "",
        "loop_count": 0
    }
    
    # Simulate the workflow
    while True:
        # Step 1: Retrieve
        state.update(retrieve(state))
        
        # Step 2: Grade
        state.update(grade_documents(state))
        
        # Step 3: Decide
        decision = decide_to_generate(state)
        
        if decision == "generate":
            # Step 4: Generate
            state.update(generate(state))
            break
        else:
            # Step 4: Rewrite and loop
            state.update(rewrite_query(state))
    
    print(f"\n💡 Answer: {state['generation']}")
    print("=" * 50)
    
    return state

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
        "How many employees work there?",
        "What programming languages are supported?",
        "What is the customer satisfaction rate?",
        "What are the company's expansion plans?"
    ]
    
    for question in questions:
        run_agentic_rag(question)
