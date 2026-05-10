#!/usr/bin/env python3
"""
Production launcher for the Agentic RAG System
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

def check_environment():
    """Check if environment is properly configured"""
    print("🔍 Checking environment configuration...")
    
    # Check required packages
    required_packages = [
        'streamlit', 'groq', 'sentence_transformers', 
        'faiss', 'numpy', 'torch', 'transformers'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements_new.txt")
        return False
    
    # Check API keys
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key or groq_key == "your_groq_api_key_here":
        print("❌ Groq API key not configured")
        print("Get free key from: https://console.groq.com/")
        print("Add to .env file: GROQ_API_KEY=your_key")
        return False
    
    print("✅ Environment check passed")
    return True

def run_development():
    """Run in development mode"""
    print("🚀 Starting development server...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"])

def run_production():
    """Run in production mode"""
    print("🚀 Starting production server...")
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_SERVER_ENABLE_CORS"] = "false"
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"])

def main():
    """Main launcher"""
    print("=" * 60)
    print("🚀 Professional Agentic RAG System Launcher")
    print("=" * 60)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Choose mode
    if len(sys.argv) > 1 and sys.argv[1] == "prod":
        run_production()
    else:
        run_development()

if __name__ == "__main__":
    main()
