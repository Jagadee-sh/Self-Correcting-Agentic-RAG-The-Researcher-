# 🚀 Professional Agentic RAG System - Deployment Guide

## 📋 Overview

This is a production-ready, self-correcting Agentic RAG system with real API integrations and professional web interface.

## 🛠️ Features

- **Real API Integration**: Groq (free LLM), Sentence Transformers (embeddings)
- **Professional Web UI**: Streamlit-based interface with real-time metrics
- **Self-Correcting Workflow**: Automatic query rewriting and document grading
- **Production Ready**: Docker containerization, health checks, logging
- **Advanced Features**: Conversation history, reasoning visualization, performance metrics

## 🔧 Setup Instructions

### 1. Get Free API Keys

#### Groq API (Required - Free)
1. Visit https://console.groq.com/
2. Sign up for free account
3. Get your API key from dashboard
4. Add to `.env` file: `GROQ_API_KEY=your_key_here`

#### Optional APIs
- **Hugging Face**: https://huggingface.co/settings/tokens (for additional embeddings)
- **Cohere**: https://dashboard.cohere.com/api-keys (for better embeddings)

### 2. Configure Environment

Update your `.env` file:
```bash
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
```

### 3. Install Dependencies

```bash
pip install -r requirements_new.txt
```

## 🚀 Deployment Options

### Option 1: Local Development

```bash
streamlit run app.py
```

### Option 2: Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or manual Docker
docker build -t agentic-rag .
docker run -p 8501:8501 --env-file .env agentic-rag
```

### Option 3: Cloud Deployment

#### Heroku
```bash
# Install Heroku CLI
heroku create your-app-name
heroku config:set GROQ_API_KEY=your_key
git push heroku main
```

#### Railway
```bash
# Install Railway CLI
railway login
railway init
railway up
```

#### AWS (ECS/Fargate)
```bash
# Build and push to ECR
aws ecr create-repository --repository-name agentic-rag
docker build -t agentic-rag .
docker tag agentic-rag:latest your-account.dkr.ecr.region.amazonaws.com/agentic-rag:latest
docker push your-account.dkr.ecr.region.amazonaws.com/agentic-rag:latest
```

## 📊 System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Interface │    │  Agentic RAG     │    │   API Services  │
│   (Streamlit)   │◄──►│   System Core    │◄──►│  (Groq, etc.)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │  Vector Store    │
                       │  (FAISS)         │
                       └──────────────────┘
```

## 🔍 Features in Detail

### 1. Self-Correcting Workflow
- **Retrieve**: Find relevant documents using semantic search
- **Grade**: Evaluate document relevance with LLM
- **Decide**: Generate answer or rewrite query
- **Rewrite**: Improve query for better results
- **Loop**: Maximum 2 retries to prevent infinite loops

### 2. Professional Web Interface
- **Real-time Metrics**: Processing time, document counts, query rewrites
- **Reasoning Visualization**: See exactly how the system thinks
- **Conversation History**: Track all interactions
- **Example Questions**: Quick-start buttons for common queries

### 3. Production Features
- **Health Checks**: Automatic monitoring
- **Error Handling**: Graceful failure management
- **Logging**: Comprehensive system logging
- **Security**: Non-root Docker user, environment variables

## 📈 Performance Metrics

The system tracks:
- **Processing Time**: How long each query takes
- **Document Retrieval**: Number of documents found vs relevant
- **Query Rewrites**: How many times queries were improved
- **Success Rate**: Percentage of queries answered successfully

## 🔧 Customization

### Adding Your Own Documents

1. Create a function to load your documents:
```python
def load_your_documents() -> List[Document]:
    # Load your PDFs, text files, etc.
    return [Document(page_content=text, metadata=metadata)]
```

2. Replace `get_sample_documents()` in `app.py`

### Customizing the LLM

Change the model in `Config` class:
```python
llm_model: str = "mixtral-8x7b-32768"  # Other Groq models
```

### Adjusting Search Parameters

```python
max_retries: int = 2              # Max query rewrites
max_docs_retrieved: int = 5       # Documents to retrieve
similarity_threshold: float = 0.3 # Minimum similarity score
```

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure Groq API key is correctly set in `.env`
   - Check if key has sufficient quota

2. **Slow Performance**
   - Reduce `max_docs_retrieved` in config
   - Use smaller embedding model

3. **Memory Issues**
   - Reduce document count
   - Use smaller embedding model

4. **Docker Issues**
   - Ensure all dependencies in `requirements_new.txt`
   - Check environment variables

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔒 Security Considerations

- API keys stored in environment variables
- Non-root Docker user
- Input validation and sanitization
- Rate limiting (implement in production)

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in the application
3. Verify API key configuration
4. Test with example questions first

## 🚀 Next Steps

- Add document upload functionality
- Implement user authentication
- Add more embedding providers
- Create API endpoints for integration
- Add monitoring and alerting
