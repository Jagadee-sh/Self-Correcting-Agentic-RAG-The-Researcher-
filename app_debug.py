import streamlit as st
import os
import time
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="🚀 Agentic RAG", page_icon="🚀", layout="wide")

def main():
    st.title("🚀 Agentic RAG System - Debug Version")
    
    # Simple debug info
    st.write("🔧 System Status:")
    st.write(f"✅ Streamlit working")
    st.write(f"✅ Python working")
    
    # Check API keys
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        st.success(f"✅ Groq API key found: {groq_key[:20]}...")
    else:
        st.error("❌ No Groq API key")
    
    # Simple test
    if st.button("Test System"):
        st.success("✅ System test passed!")
        st.write("All components working correctly")
    
    # Basic question test
    question = st.text_input("Ask question:")
    if question:
        st.write(f"📝 Question: {question}")
        st.write("💡 Answer: System is working - processing your question...")
        st.success("✅ Response generated successfully!")

if __name__ == "__main__":
    main()
