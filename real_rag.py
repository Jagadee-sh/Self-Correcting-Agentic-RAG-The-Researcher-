import os
import json
import numpy as np
from typing import List, TypedDict, Dict, Any
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ==========================================
# 1. SETUP & DATA
# ==========================================

# Create a comprehensive knowledge base
texts = [
    "The company revenue for 2023 was $5 million, showing a 25% increase from 2022.",
    "The main product is the 'AI-Toolbox' which helps developers build AI applications faster.",
    "The CEO of the company is Alice Smith, who has 15 years of experience in tech industry.",
    "The company was founded in 2020 in San Francisco by a team of AI researchers.",
    "The 'AI-Toolbox' integrates with LangChain and OpenAI for seamless AI development.",
    "The company has 50 employees as of 2023, with plans to hire 20 more in 2024.",
    "The AI-Toolbox supports multiple programming languages including Python, JavaScript, Java, and TypeScript.",
    "Customer satisfaction rate is 95% according to recent surveys conducted in Q4 2023.",
    "The company plans to expand to Europe in 2024, starting with offices in London and Berlin.",
    "R&D department occupies 30% of the company budget, focusing on next-gen AI models.",
    "The company has raised $10 million in Series A funding led by TechVentures Capital.",
    "Product pricing starts at $99/month for the basic plan and goes up to $999/month for enterprise.",
    "The AI-Toolbox has over 10,000 active users and 500 enterprise customers.",
    "The company's mission is to democratize AI development for businesses of all sizes.",
    "Technical support is available 24/7 with average response time under 2 hours."
]

# Create Document objects
documents = [Document(page_content=text, metadata={"source": f"doc_{i}", "index": i}) for i, text in enumerate(texts)]

# ==========================================
# 2. SIMPLE EMBEDDINGS (TF-IDF)
# ==========================================

class SimpleEmbeddings:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.embeddings = None
        self.documents = None
    
    def fit(self, documents):
        self.documents = documents
        texts = [doc.page_content for doc in documents]
        self.embeddings = self.vectorizer.fit_transform(texts)
        return self.embeddings
    
    def embed_query(self, query):
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted")
        return self.vectorizer.transform([query])
    
    def similarity_search(self, query, k=3):
        query_embedding = self.embed_query(query)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top k most similar documents
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Threshold for relevance
                results.append(self.documents[idx])
        
        return results

# ==========================================
# 3. DEFINE STATE
# ==========================================

class AgentState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    loop_count: int
    query_history: List[str]

# ==========================================
# 4. SIMPLE LLM SIMULATOR
# ==========================================

class SimpleLLM:
    def __init__(self):
        self.patterns = {
            "revenue": [
                r"revenue|income|earnings|financial|money",
                "Based on the documents, the company's revenue for 2023 was $5 million, showing a 25% increase from 2022."
            ],
            "ceo": [
                r"ceo|leader|chief|executive|alice",
                "According to the information, the CEO of the company is Alice Smith, who has 15 years of experience in tech industry."
            ],
            "product": [
                r"product|tool|service|ai-toolbox|software",
                "The main product is the 'AI-Toolbox' which helps developers build AI applications faster and integrates with LangChain and OpenAI."
            ],
            "founded": [
                r"founded|started|established|begin|create",
                "The company was founded in 2020 in San Francisco by a team of AI researchers."
            ],
            "employees": [
                r"employees|staff|workers|team|people",
                "The company has 50 employees as of 2023, with plans to hire 20 more in 2024."
            ],
            "languages": [
                r"languages|programming|python|javascript|java|typescript",
                "The AI-Toolbox supports multiple programming languages including Python, JavaScript, Java, and TypeScript."
            ],
            "satisfaction": [
                r"satisfaction|customer|feedback|rating|review",
                "Customer satisfaction rate is 95% according to recent surveys conducted in Q4 2023."
            ],
            "expansion": [
                r"expand|expansion|growth|europe|london|berlin",
                "The company plans to expand to Europe in 2024, starting with offices in London and Berlin."
            ],
            "funding": [
                r"funding|investment|capital|money|raise",
                "The company has raised $10 million in Series A funding led by TechVentures Capital."
            ],
            "pricing": [
                r"price|cost|pricing|plan|subscription|fee",
                "Product pricing starts at $99/month for the basic plan and goes up to $999/month for enterprise."
            ],
            "users": [
                r"users|customers|clients|adoption|active",
                "The AI-Toolbox has over 10,000 active users and 500 enterprise customers."
            ],
            "mission": [
                r"mission|vision|purpose|goal|why",
                "The company's mission is to democratize AI development for businesses of all sizes."
            ],
            "support": [
                r"support|help|assistance|service|24/7",
                "Technical support is available 24/7 with average response time under 2 hours."
            ],
            "rnd": [
                r"r&d|research|development|budget|innovation",
                "R&D department occupies 30% of the company budget, focusing on next-gen AI models."
            ]
        }
    
    def generate_response(self, question: str, context: str) -> str:
        question_lower = question.lower()
        
        # Find matching pattern
        for key, (pattern, response) in self.patterns.items():
            if re.search(pattern, question_lower):
                # Enhance response with context if available
                if context and len(context) > 50:
                    return f"{response} Additional context: {context[:200]}..."
                return response
        
        # Default response
        if context:
            return f"Based on the available information: {context[:300]}..."
        return "I don't have specific information about that topic. Could you try asking about revenue, CEO, products, employees, or other company details?"
    
    def grade_document(self, question: str, document: str) -> str:
        """Grade document relevance"""
        question_lower = question.lower()
        doc_lower = document.lower()
        
        # Check for keyword overlap
        question_words = set(re.findall(r'\w+', question_lower))
        doc_words = set(re.findall(r'\w+', doc_lower))
        
        overlap = len(question_words.intersection(doc_words))
        
        if overlap >= 2:
            return "yes"
        elif overlap >= 1:
            return "maybe"
        else:
            return "no"
    
    def rewrite_query(self, question: str) -> str:
        """Rewrite query for better retrieval"""
        question_lower = question.lower()
        
        # Simple query rewriting rules
        rewrites = {
            r"what is": "tell me about",
            r"who is": "information about",
            r"when was": "date of",
            r"how many": "number of",
            r"tell me about": "details regarding",
            r"information about": "facts about"
        }
        
        new_question = question
        for pattern, replacement in rewrites.items():
            new_question = re.sub(pattern, replacement, new_question, flags=re.IGNORECASE)
        
        return new_question

# ==========================================
# 5. DEFINE THE NODES
# ==========================================

def retrieve(state: AgentState, embeddings: SimpleEmbeddings):
    """Node: Retrieve documents from vector store."""
    print("--- RETRIEVING DOCUMENTS ---")
    question = state["question"]
    
    # Retrieve relevant documents
    relevant_docs = embeddings.similarity_search(question, k=5)
    
    if not relevant_docs:
        # Fallback to keyword search
        relevant_docs = documents[:3]
    
    print(f"Retrieved {len(relevant_docs)} documents")
    return {"documents": relevant_docs, "question": question}

def grade_documents(state: AgentState, llm: SimpleLLM):
    """Node: Grade retrieved documents for relevance."""
    print("--- GRADING DOCUMENTS ---")
    question = state["question"]
    documents = state["documents"]
    
    filtered_docs = []
    for d in documents:
        score = llm.grade_document(question, d.page_content)
        if score == "yes":
            print("--- GRADE: DOCUMENT RELEVANT ---")
            filtered_docs.append(d)
        elif score == "maybe":
            print("--- GRADE: DOCUMENT SOMEWHAT RELEVANT ---")
            filtered_docs.append(d)
        else:
            print("--- GRADE: DOCUMENT NOT RELEVANT ---")

    return {"documents": filtered_docs, "question": question}

def rewrite_query(state: AgentState, llm: SimpleLLM):
    """Node: Rewrite the query to improve retrieval."""
    print("--- REWRITING QUERY ---")
    question = state["question"]
    
    better_question = llm.rewrite_query(question)
    
    print(f"New Question: {better_question}")
    
    # Add to query history
    query_history = state.get("query_history", [])
    query_history.append(question)
    
    return {"question": better_question, "loop_count": state.get("loop_count", 0) + 1, "query_history": query_history}

def generate(state: AgentState, llm: SimpleLLM):
    """Node: Generate answer using the retrieved context."""
    print("--- GENERATING ANSWER ---")
    question = state["question"]
    documents = state["documents"]
    
    # Combine context from relevant documents
    context = " ".join([doc.page_content for doc in documents[:3]])
    
    # Generate response
    generation = llm.generate_response(question, context)
    
    return {"documents": documents, "question": question, "generation": generation}

# ==========================================
# 6. DEFINE THE CONTROL FLOW
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
# 7. BUILD THE GRAPH WORKFLOW
# ==========================================

def run_agentic_rag(question: str, embeddings: SimpleEmbeddings, llm: SimpleLLM):
    """Run the agentic RAG workflow."""
    print(f"\n🚀 Self-Correcting Agentic RAG System")
    print("=" * 60)
    print(f"📝 Question: {question}")
    print("-" * 40)
    
    # Initial state
    state = {
        "question": question,
        "documents": [],
        "generation": "",
        "loop_count": 0,
        "query_history": []
    }
    
    # Workflow loop
    while True:
        # Step 1: Retrieve
        state.update(retrieve(state, embeddings))
        
        # Step 2: Grade
        state.update(grade_documents(state, llm))
        
        # Step 3: Decide
        decision = decide_to_generate(state)
        
        if decision == "generate":
            # Step 4: Generate
            state.update(generate(state, llm))
            break
        else:
            # Step 4: Rewrite and loop
            state.update(rewrite_query(state, llm))
    
    print(f"\n💡 Answer: {state['generation']}")
    print(f"🔄 Query Rewrites: {state['loop_count']}")
    if state['query_history']:
        print(f"📜 Query History: {' → '.join(state['query_history'])} → {state['question']}")
    print("=" * 60)
    
    return state

# ==========================================
# 8. MAIN APPLICATION
# ==========================================

if __name__ == "__main__":
    print("🔧 Initializing Agentic RAG System...")
    
    # Initialize embeddings
    embeddings = SimpleEmbeddings()
    embeddings.fit(documents)
    
    # Initialize LLM
    llm = SimpleLLM()
    
    # Test questions
    questions = [
        "What is the company's revenue?",
        "Who is the CEO?",
        "What products does the company offer?",
        "When was the company founded?",
        "How many employees work there?",
        "What programming languages are supported?",
        "What is the customer satisfaction rate?",
        "What are the company's expansion plans?",
        "How much funding has the company raised?",
        "What is the pricing structure?",
        "How many users do they have?",
        "What is the company's mission?"
    ]
    
    print(f"📚 Knowledge Base: {len(documents)} documents loaded")
    print(f"🤖 LLM Simulator: Ready with {len(llm.patterns)} response patterns")
    print(f"🔍 Embeddings: TF-IDF vectorizer trained")
    
    # Run tests
    for question in questions:
        run_agentic_rag(question, embeddings, llm)
