import os
from dotenv import load_dotenv
import streamlit as st
import cohere
from crewai import Agent, Task, Crew
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from datetime import datetime
import json
from database import *
from duckduckgo_search import DDGS

# Load environment variables
load_dotenv()

# Streamlit page config
st.set_page_config(page_title="AI Content Generator", page_icon="📝", layout="wide")

st.markdown("""
<style>
/* Ana tema renkleri */
:root {
    --primary: #FF4B4B;
    --secondary: #7E44FF;
    --background: #0E1117;
    --text: #FFFFFF;
}

/* Sidebar stilleri */
.sidebar .sidebar-content {
    background: linear-gradient(180deg, var(--background) 0%, #1E1E2E 100%);
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* Buton stilleri */
.stButton > button {
    background: linear-gradient(45deg, var(--primary), var(--secondary));
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}

/* Input alanları */
.stTextInput input, .stTextArea textarea {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    color: var(--text);
}

/* Daktilo efekti geliştirmeleri */
.typing-effect {
    font-family: 'Courier New', monospace;
    line-height: 1.6;
    padding: 2rem;
    background: rgba(255,255,255,0.02);
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.05);
    margin: 1rem 0;
}

@keyframes typing {
    from { width: 0; opacity: 0; }
    to { width: 100%; opacity: 1; }
}

/* History butonları */
.history-button {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    padding: 0.75rem;
    margin: 0.5rem 0;
    transition: all 0.3s ease;
}

.history-button:hover {
    background: rgba(255,255,255,0.05);
    transform: translateX(5px);
}

/* Başlık stilleri */
h1, h2, h3 {
    background: linear-gradient(45deg, var(--primary), var(--secondary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
}

/* Spinner stil */
.stSpinner > div {
    border-color: var(--primary) transparent transparent transparent;
}

/* View butonları için özel stil */
.stButton>button {
    background: rgba(126, 68, 255, 0.2);
    border: 1px solid rgba(126, 68, 255, 0.3);
    color: #7E44FF;
    font-size: 0.8em;
    padding: 2px 8px;
    border-radius: 4px;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    background: rgba(126, 68, 255, 0.3);
    border-color: #7E44FF;
    transform: translateY(-1px);
}

/* Sidebar içeriği için stil */
.sidebar .sidebar-content {
    background: linear-gradient(180deg, #0E1117 0%, #1E1E2E 100%);
}

/* Scrollbar stilleri */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
}

::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.2);
}
</style>
""", unsafe_allow_html=True)

def search_web(query: str) -> str:
    """Search the web for information"""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        return "\n".join([f"- {r['body']}" for r in results])
    except Exception as e:
        return f"Search error: {str(e)}"

def generate_with_cohere(topic, temperature=0.7):
    """Generate content using Cohere directly"""
    try:
        co = cohere.Client(os.getenv('COHERE_API_KEY'))
        response = co.generate(
            prompt=f"""Write a comprehensive article about {topic}.
            The article should:
            - Be well-structured with clear sections
            - Include relevant information and insights
            - Be written in markdown format
            - Be engaging and informative
            
            Article:""",
            max_tokens=2000,
            temperature=temperature,
            model='command'  # or 'command-light', 'command-medium', 'command-xlarge'
        )
        return response.generations[0].text
    except Exception as e:
        st.error(f"Cohere error: {str(e)}")
        return None

def generate_with_crew(topic):
    """Generate content using CrewAI"""
    try:
        # Disable telemetry
        os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
        os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
        
        # Initialize OpenAI
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7
        )
        
        # Researcher Agent
        researcher = Agent(
            role='Researcher',
            goal=f'Research about {topic}',
            backstory='Expert researcher with vast knowledge',
            tools=[Tool(
                name='Web Search',
                func=search_web,
                description='Search the web for information'
            )],
            llm=llm
        )

        # Writer Agent
        writer = Agent(
            role='Writer',
            goal='Write engaging content',
            backstory='Professional content writer',
            llm=llm
        )

        # Tasks
        research = Task(
            description=f"Research about {topic}",
            expected_output="Comprehensive research findings",
            agent=researcher
        )

        write = Task(
            description="Write a markdown article using the research",
            expected_output="Well-structured article in markdown format",
            agent=writer
        )

        # Create Crew
        crew = Crew(
            agents=[researcher, writer],
            tasks=[research, write]
        )

        result = crew.kickoff()
        return str(result)
        
    except Exception as e:
        st.error(f"CrewAI error: {str(e)}")
        st.error("Falling back to Cohere...")
        return generate_with_cohere(topic)

def save_new_content(query_text, content_text):
    """Save new content to database"""
    try:
        query_id = save_query_to_db(query_text)
        content = {
            'final_content': str(content_text),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_results_to_db(query_id, content)
        return True
    except Exception as e:
        st.error(f"Save error: {str(e)}")
        return False

def main():
    st.title("📝 AI Content Generator")
    
    # Sidebar tasarımı
    with st.sidebar:
        st.markdown("""
        <style>
        .sidebar-content {
            padding: 20px;
            border-radius: 10px;
            background: #1E1E1E;
        }
        </style>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown("### ✨ New Content")
            
            topic = st.text_area(
                "What would you like to write about?",
                placeholder="Enter your topic here...",
                height=100
            )
            
            col1, col2 = st.columns(2)
            with col1:
                ai_model = st.radio(
                    "AI Model",
                    ["🚀 Cohere", "🤖 CrewAI"],
                    format_func=lambda x: x.split()[1]
                )
            
            with col2:
                if "Cohere" in ai_model:
                    temperature = st.slider("Creativity", 0.0, 1.0, 0.7)
            
            if st.button("Generate ✨", type="primary", use_container_width=True):
                if topic:
                    with st.spinner('Creating your content...'):
                        try:
                            if "Cohere" in ai_model:
                                result = generate_with_cohere(topic, temperature)
                            else:
                                result = generate_with_crew(topic)
                                
                            if result and save_new_content(topic, result):
                                st.success("Content generated!")
                                st.session_state.new_content = result
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.warning("Please enter a topic")

        st.markdown("---")
        
        # History bölümü
        st.markdown("### 📚 History")
        search = st.text_input("🔍", placeholder="Search history...")
        
        start_date = st.date_input("From", key="start_date")
        end_date = st.date_input("To", key="end_date")
        
        sort_order = st.selectbox(
            "Sort by",
            ["Newest First", "Oldest First"],
            label_visibility="collapsed"
        )

    # Ana içerik alanı
    content_placeholder = st.empty()
    
    with content_placeholder.container():
        if 'new_content' in st.session_state:
            # Daktilo efekti için JavaScript
            st.markdown("""
            <style>
            @keyframes typing {
                from { width: 0 }
                to { width: 100% }
            }
            .typing-effect {
                overflow: hidden;
                white-space: pre-wrap;
                animation: typing 2s steps(40, end);
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown(f'<div class="typing-effect">{st.session_state.new_content}</div>', 
                       unsafe_allow_html=True)
            
        elif 'selected_query' in st.session_state:
            query = st.session_state.selected_query
            st.header(query['query'])
            st.markdown(f"*{query['created_at'].strftime('%Y-%m-%d %H:%M:%S')}*")
            st.markdown("---")
            
            content = query['content']
            if isinstance(content, dict):
                st.markdown(f'<div class="typing-effect">{content.get("final_content", "No content available")}</div>', 
                          unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="typing-effect">{str(content)}</div>', 
                          unsafe_allow_html=True)
        else:
            st.info("👈 Generate new content or select from history")

    # History listesi
    history = get_filtered_history(search, start_date, end_date, sort_order)
    
    for query_id, query, created_at, content, title in history:
        with st.sidebar:
            col1, col2 = st.columns([8, 2])
            
            with col1:
                # Sol tarafta başlık ve tarih
                st.markdown(f"""
                    <div style='
                        padding: 0.5rem;
                        cursor: pointer;
                    '>
                        <div style='
                            font-size: 0.9em; 
                            color: #E0E0E0;
                            margin-bottom: 4px;
                        '>{query[:50]}...</div>
                        <div style='
                            font-size: 0.7em; 
                            color: #808080;
                        '>{created_at.strftime('%Y-%m-%d %H:%M')}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Sağ tarafta view butonu
                if st.button("View", key=f"view_{query_id}"):
                    try:
                        if content:
                            content_data = json.loads(content) if isinstance(content, str) else content
                            st.session_state.selected_query = {
                                'id': query_id,
                                'query': query,
                                'created_at': created_at,
                                'content': content_data
                            }
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            # Ayırıcı çizgi
            st.markdown("<hr style='margin: 5px 0; opacity: 0.2;'>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
