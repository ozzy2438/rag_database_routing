# 📠 AI News Generator with RAG & Database Integration

A Streamlit application that generates comprehensive blog posts about any topic using AI agents, with database integration for history tracking and RAG capabilities.

## Features

- **AI Content Generation**: Uses CrewAI and Cohere's Command R7B to generate well-researched blog posts
- **Database Integration**: PostgreSQL with pgvector for storing and retrieving research history
- **Search History**: Advanced filtering and search capabilities for past queries
- **Data Visualization**: Interactive charts and graphs for data analysis
- **RAG Capabilities**: Retrieval Augmented Generation for enhanced content creation

## Requirements

- Python 3.11
- PostgreSQL 14+
- pgvector extension

## Installation

1. **Create Virtual Environment**:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # Unix/macOS
   # or
   .\venv\Scripts\activate  # Windows
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up PostgreSQL**:
   ```sql
   CREATE DATABASE ai_research_db;
   \c ai_research_db
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

4. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your credentials:
   # - Database credentials
   # - Cohere API key
   # - Serper API key
   ```

5. **Run the Application**:
   ```bash
   streamlit run news_agent.py
   ```

## Project Structure

```
rag_database_routing/
├── news_agent.py          # Main application with AI content generation
├── database.py           # Database operations and connections
├── main.py              # RAG and data visualization
├── requirements.txt     # Project dependencies
├── .env                # Configuration (private)
├── .env.example        # Example configuration
└── .gitignore         # Git ignore rules
```

## Technologies Used

- **CrewAI**: For orchestrating AI agents
- **Cohere**: Command R7B model for content generation
- **PostgreSQL**: Database with pgvector extension
- **Streamlit**: User interface
- **Plotly**: Data visualization
- **Pandas**: Data processing

## How It Works

1. **Content Generation**
   - User enters a topic
   - AI agents research and generate content
   - Results are stored in database

2. **History Tracking**
   - All queries and results are saved
   - Advanced filtering and search
   - Date-based organization

3. **Data Analysis**
   - Interactive visualizations
   - Trend analysis
   - Statistical insights

## License

MIT
