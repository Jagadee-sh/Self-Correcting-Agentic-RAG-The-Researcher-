import streamlit as st
import os
import json
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import re
from langchain_core.documents import Document
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# ==========================================
# CONFIGURATION
# ==========================================

@dataclass
class Config:
    """Application configuration"""
    max_retries: int = 2
    max_docs_retrieved: int = 5
    similarity_threshold: float = 0.3
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "llama3-8b-8192"
    app_title: str = "🚀 Professional Agentic RAG System"
    app_description: str = "Self-correcting AI system with real-time document retrieval and intelligent query rewriting"

# ==========================================
# EMBEDDINGS MANAGER
# ==========================================

class EmbeddingsManager:
    """Handle embeddings with multiple providers"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.index = None
        self.documents = []
        self.embeddings = None
        
    def initialize_model(self):
        """Initialize sentence transformer model"""
        try:
            self.model = SentenceTransformer(self.config.embedding_model)
            logger.info(f"Loaded embedding model: {self.config.embedding_model}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def create_embeddings(self, documents: List[Document]) -> bool:
        """Create embeddings for documents"""
        if not self.model:
            if not self.initialize_model():
                return False
        
        try:
            self.documents = documents
            texts = [doc.page_content for doc in documents]
            
            # Create embeddings
            self.embeddings = self.model.encode(texts)
            
            # Create FAISS index
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(self.embeddings.astype('float32'))
            
            logger.info(f"Created embeddings for {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            return False
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents"""
        if not self.model or not self.index:
            return []
        
        try:
            query_embedding = self.model.encode([query])
            query_embedding = query_embedding.astype('float32')
            
            # Search
            distances, indices = self.index.search(query_embedding, min(k, len(self.documents)))
            
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.documents) and dist < (1 - self.config.similarity_threshold):
                    doc = self.documents[idx]
                    doc.metadata['similarity_score'] = float(1 - dist)
                    doc.metadata['rank'] = i + 1
                    results.append(doc)
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

# ==========================================
# LLM MANAGER
# ==========================================

class LLMManager:
    """Handle LLM interactions with multiple providers"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = None
        self.initialize_client()
    
    def initialize_client(self):
        """Initialize Groq client"""
        try:
            api_key = os.getenv("GROQ_API_KEY")
            if api_key and api_key != "your_groq_api_key_here":
                self.client = Groq(api_key=api_key)
                logger.info("Groq client initialized")
                return True
            else:
                logger.warning("Groq API key not configured")
                return False
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            return False
    
    def generate_response(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate response using Groq"""
        if not self.client:
            return "LLM client not available. Please configure API keys."
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"Error generating response: {str(e)}"
    
    def grade_document(self, question: str, document: str) -> str:
        """Grade document relevance"""
        prompt = f"""
        You are a document relevance grader. Rate the following document's relevance to the question.
        
        Question: {question}
        Document: {document}
        
        Respond with only: "relevant", "somewhat_relevant", or "not_relevant"
        """
        
        response = self.generate_response(prompt, max_tokens=10).lower().strip()
        
        if "relevant" in response and "somewhat" not in response:
            return "relevant"
        elif "relevant" in response:
            return "somewhat_relevant"
        else:
            return "not_relevant"
    
    def rewrite_query(self, question: str) -> str:
        """Rewrite query for better retrieval"""
        prompt = f"""
        Rewrite the following question to be more effective for document retrieval.
        Make it clearer, more specific, and optimize for keyword search.
        
        Original question: {question}
        
        Rewritten question:
        """
        
        return self.generate_response(prompt, max_tokens=100).strip()
    
    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer based on context"""
        prompt = f"""
        You are a helpful AI assistant. Answer the question based on the provided context.
        If the context doesn't contain the answer, say "I don't have enough information to answer this question."
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """
        
        return self.generate_response(prompt, max_tokens=500)

# ==========================================
# AGENTIC RAG SYSTEM
# ==========================================

class AgenticRAGSystem:
    """Main agentic RAG system"""
    
    def __init__(self, config: Config):
        self.config = config
        self.embeddings_manager = EmbeddingsManager(config)
        self.llm_manager = LLMManager(config)
        self.conversation_history = []
        
    def initialize(self, documents: List[Document]) -> bool:
        """Initialize the system with documents"""
        return self.embeddings_manager.create_embeddings(documents)
    
    def process_query(self, question: str) -> Dict[str, Any]:
        """Process a query through the agentic workflow"""
        start_time = time.time()
        
        # Initialize state
        state = {
            "question": question,
            "original_question": question,
            "documents": [],
            "filtered_documents": [],
            "answer": "",
            "loop_count": 0,
            "query_history": [],
            "grades": [],
            "reasoning": []
        }
        
        # Agentic workflow
        while state["loop_count"] <= self.config.max_retries:
            # Retrieve documents
            state["documents"] = self.embeddings_manager.search(
                question, k=self.config.max_docs_retrieved
            )
            state["reasoning"].append(f"Retrieved {len(state['documents'])} documents")
            
            if not state["documents"]:
                state["reasoning"].append("No documents found, generating answer without context")
                state["answer"] = "I couldn't find any relevant documents to answer your question."
                break
            
            # Grade documents
            state["filtered_documents"] = []
            for doc in state["documents"]:
                grade = self.llm_manager.grade_document(question, doc.page_content)
                state["grades"].append(grade)
                
                if grade in ["relevant", "somewhat_relevant"]:
                    state["filtered_documents"].append(doc)
                    state["reasoning"].append(f"Document graded as {grade}")
                else:
                    state["reasoning"].append("Document graded as not relevant")
            
            # Decision point
            if state["filtered_documents"] or state["loop_count"] >= self.config.max_retries:
                # Generate answer
                context = "\n\n".join([doc.page_content for doc in state["filtered_documents"][:3]])
                state["answer"] = self.llm_manager.generate_answer(question, context)
                state["reasoning"].append("Generated answer based on filtered documents")
                break
            else:
                # Rewrite query
                old_question = question
                question = self.llm_manager.rewrite_query(question)
                state["query_history"].append(old_question)
                state["reasoning"].append(f"Rewrote query: '{old_question}' → '{question}'")
                state["loop_count"] += 1
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Add to conversation history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "question": state["original_question"],
            "answer": state["answer"],
            "processing_time": processing_time,
            "loop_count": state["loop_count"],
            "documents_found": len(state["documents"]),
            "relevant_documents": len(state["filtered_documents"])
        })
        
        return {
            "question": state["original_question"],
            "answer": state["answer"],
            "processing_time": processing_time,
            "loop_count": state["loop_count"],
            "documents_found": len(state["documents"]),
            "relevant_documents": len(state["filtered_documents"]),
            "query_history": state["query_history"],
            "reasoning": state["reasoning"],
            "grades": state["grades"]
        }

# ==========================================
# SAMPLE DATA
# ==========================================

def get_sample_documents() -> List[Document]:
    """Get sample documents for demonstration"""
    texts = [
        """The company reported revenue of $5 million for fiscal year 2023, representing a 25% increase from the previous year's $4 million. 
        This growth was primarily driven by the success of their flagship AI-Toolbox product and expansion into new markets.""",
        
        """Alice Smith serves as the Chief Executive Officer of the company. With over 15 years of experience in the technology sector, 
        she previously held leadership positions at major tech companies including Google and Microsoft before joining in 2021.""",
        
        """The AI-Toolbox is the company's main product offering. It's a comprehensive development platform that helps developers build, 
        deploy, and manage AI applications. The platform integrates seamlessly with popular frameworks like LangChain, OpenAI, and Hugging Face.""",
        
        """Founded in 2020 in San Francisco, the company started as a small team of AI researchers from Stanford and MIT. 
        The initial funding round raised $2 million, which helped establish the core technology and hire the first engineering team.""",
        
        """As of 2023, the company employs 50 people across engineering, sales, and operations. Plans are in place to expand the team 
        by 20 additional hires in 2024, focusing on engineering and customer support roles to support growing demand.""",
        
        """The AI-Toolbox supports multiple programming languages including Python, JavaScript, Java, TypeScript, and Go. 
        This multi-language support allows developers to use their preferred language while building AI applications.""",
        
        """Customer satisfaction surveys conducted in Q4 2023 showed a 95% satisfaction rate among users. The survey included responses 
        from over 1,000 customers and highlighted the platform's ease of use and comprehensive feature set.""",
        
        """The company plans to expand internationally in 2024, with initial focus on European markets. New offices are scheduled to open 
        in London and Berlin in Q2 2024, followed by Paris in Q4 2024. This expansion is supported by the recent Series A funding.""",
        
        """R&D investment represents 30% of the company's budget allocation. The research team focuses on developing next-generation 
        AI models, improving embedding techniques, and enhancing the platform's natural language processing capabilities.""",
        
        """The company successfully closed a $10 million Series A funding round led by TechVentures Capital, with participation from 
        several other institutional investors. The funds will be used for product development, team expansion, and market entry into Europe.""",
        
        """Product pricing follows a tiered structure: Basic plan at $99/month for individual developers, Professional plan at $299/month 
        for small teams, and Enterprise plan starting at $999/month for large organizations with custom requirements.""",
        
        """The AI-Toolbox platform has over 10,000 active monthly users and 500 enterprise customers. Notable clients include Fortune 500 
        companies in finance, healthcare, and technology sectors who use the platform for various AI applications.""",
        
        """The company's mission is to democratize AI development by making advanced AI tools accessible to businesses of all sizes. 
        They believe that AI should empower innovation rather than replace human creativity and problem-solving capabilities.""",
        
        """Technical support is available 24/7 through multiple channels including live chat, email, and phone. The average response time 
        for support tickets is under 2 hours, with a 98% customer satisfaction rate for support interactions.""",
        
        """The platform includes advanced features such as automated model training, data preprocessing tools, and deployment pipelines. 
        These features help reduce development time by up to 60% compared to traditional AI development approaches."""
    ]
    
    return [Document(page_content=text, metadata={"source": f"company_doc_{i}", "index": i}) for i, text in enumerate(texts)]

# ==========================================
# STREAMLIT APP
# ==========================================

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Professional Agentic RAG",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize config
    config = Config()
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = AgenticRAGSystem(config)
        st.session_state.initialized = False
        st.session_state.conversation_history = []
    
    # Header
    st.title(config.app_title)
    st.markdown(f"*{config.app_description}*")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Status
        st.subheader("API Status")
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key and groq_key != "your_groq_api_key_here":
            st.success("✅ Groq API Configured")
        else:
            st.error("❌ Groq API Key Missing")
            st.info("Get free key from: https://console.groq.com/")
        
        # Initialize system
        if not st.session_state.initialized:
            if st.button("🚀 Initialize System", type="primary"):
                with st.spinner("Loading documents and creating embeddings..."):
                    documents = get_sample_documents()
                    if st.session_state.rag_system.initialize(documents):
                        st.session_state.initialized = True
                        st.success("System initialized successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to initialize system")
        
        # System info
        if st.session_state.initialized:
            st.subheader("System Information")
            st.info(f"📚 Documents: {len(get_sample_documents())}")
            st.info(f"🤖 Model: {config.llm_model}")
            st.info(f"🔍 Embeddings: {config.embedding_model}")
    
    # Main content
    if not st.session_state.initialized:
        st.warning("Please initialize the system from the sidebar to begin.")
        return
    
    # Query interface
    st.header("💬 Ask a Question")
    
    # Example questions
    example_questions = [
        "What is the company's revenue?",
        "Who is the CEO?",
        "What products does the company offer?",
        "When was the company founded?",
        "How many employees work there?",
        "What programming languages are supported?",
        "What is the customer satisfaction rate?",
        "What are the company's expansion plans?"
    ]
    
    # Question input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        question = st.text_input(
            "Enter your question:",
            placeholder="e.g., What is the company's revenue?",
            key="question_input"
        )
    
    with col2:
        st.write("")
        if st.button("🔍 Search", type="primary"):
            if question:
                with st.spinner("Processing your question..."):
                    result = st.session_state.rag_system.process_query(question)
                    st.session_state.current_result = result
                    st.session_state.conversation_history.append(result)
    
    # Example questions buttons
    st.subheader("📝 Example Questions")
    cols = st.columns(2)
    for i, q in enumerate(example_questions[:8]):
        with cols[i % 2]:
            if st.button(q, key=f"example_{i}", use_container_width=True):
                st.session_state.question_input = q
    
    # Display results
    if 'current_result' in st.session_state:
        result = st.session_state.current_result
        
        st.divider()
        st.header("📊 Results")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Processing Time", f"{result['processing_time']:.2f}s")
        with col2:
            st.metric("Documents Found", result['documents_found'])
        with col3:
            st.metric("Relevant Docs", result['relevant_documents'])
        with col4:
            st.metric("Query Rewrites", result['loop_count'])
        
        # Answer
        st.subheader("💡 Answer")
        st.write(result['answer'])
        
        # Reasoning
        with st.expander("🧠 Reasoning Process"):
            for i, step in enumerate(result['reasoning'], 1):
                st.write(f"{i}. {step}")
        
        # Query history
        if result['query_history']:
            st.subheader("🔄 Query Evolution")
            evolution = [result['question']] + result['query_history']
            for i, q in enumerate(evolution):
                if i == 0:
                    st.write(f"🔍 **Original**: {q}")
                else:
                    st.write(f"✏️ **Rewrite {i}**: {q}")
    
    # Conversation history
    if st.session_state.conversation_history:
        st.divider()
        st.header("📜 Conversation History")
        
        for i, conv in enumerate(reversed(st.session_state.conversation_history[-5:]), 1):
            with st.expander(f"Q{i}: {conv['question'][:50]}..."):
                st.write(f"**Question**: {conv['question']}")
                st.write(f"**Answer**: {conv['answer']}")
                st.write(f"**Processing Time**: {conv['processing_time']:.2f}s")
                st.write(f"**Documents**: {conv['relevant_documents']}/{conv['documents_found']} relevant")

if __name__ == "__main__":
    main()
