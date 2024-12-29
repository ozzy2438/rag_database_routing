import os

import gc
import tempfile
import uuid
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.readers.docling import DoclingReader
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core import Document

import streamlit as st

# Torch uyarılarını gizle
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('transformers').setLevel(logging.ERROR)

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None

@st.cache_resource
def load_llm():
    try:
        llm = Ollama(model="llama3.2", request_timeout=120.0)
        return llm
    except Exception as e:
        st.error("Model not found. Please check your Ollama installation.")
        st.error(f"Error: {str(e)}")
        st.stop()

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()


def display_file(file, file_type):
    """Display Excel or CSV file preview with visualizations"""
    st.markdown(f"### 📑 {file_type} Preview")
    if file_type == "Excel":
        df = pd.read_excel(file)
    else:  # CSV
        df = pd.read_csv(file)
    
    # Veri önizlemesi
    with st.expander("Show Data Preview", expanded=True):
        st.dataframe(df)
    
    # Veri özeti
    with st.expander("Show Data Summary", expanded=True):
        st.markdown("#### 📊 Data Statistics")
        st.write(df.describe())
        
        st.markdown("#### 📋 Column Info")
        col_info = pd.DataFrame({
            'Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum()
        })
        st.write(col_info)
    
    # Görselleştirmeler
    create_visualizations(df)


def load_document(file_path, file_type):
    """Load document based on file type"""
    if file_type == "Excel":
        reader = DoclingReader()
        loader = SimpleDirectoryReader(
            input_dir=os.path.dirname(file_path),
            file_extractor={".xlsx": reader, ".xls": reader},
        )
        return loader.load_data()
    else:  # CSV için özelleştirilmiş işleme
        df = pd.read_csv(file_path)
        # Veriyi daha anlaşılır bir formata çevirelim
        summary = []
        
        # Genel istatistikler
        summary.append(f"Total Records: {len(df)}")
        summary.append(f"Columns: {', '.join(df.columns)}")
        
        # Her sayısal sütun için istatistikler
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            stats = f"\n{col} Statistics:"
            stats += f"\n- Average: {df[col].mean():.2f}"
            stats += f"\n- Minimum: {df[col].min()}"
            stats += f"\n- Maximum: {df[col].max()}"
            summary.append(stats)
        
        # Ham veriyi ekle
        summary.append("\nRaw Data Sample (first 10 rows):")
        summary.append(df.head(10).to_string())
        
        text_content = "\n".join(summary)
        return [Document(text=text_content)]


def create_visualizations(df):
    """Create interactive visualizations for the data"""
    st.markdown("### 📊 Data Visualizations")
    
    # Sayısal ve kategorik sütunları belirle
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Görselleştirme seçenekleri
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Bar Chart", "Line Chart", "Scatter Plot", "Box Plot", "Histogram"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_axis = st.selectbox("Select X-axis", df.columns)
    
    with col2:
        if viz_type in ["Scatter Plot"]:
            y_axis = st.selectbox("Select Y-axis", numeric_cols)
        else:
            y_axis = st.selectbox("Select Y-axis (optional)", ["None"] + list(numeric_cols))
    
    # Renk kodlaması için kategori seçimi
    color_col = st.selectbox("Select Color Category (optional)", ["None"] + list(categorical_cols))
    color_col = None if color_col == "None" else color_col
    
    # Görselleştirmeyi oluştur
    fig = None
    
    try:
        if viz_type == "Bar Chart":
            if y_axis == "None":
                fig = px.bar(df, x=x_axis, color=color_col)
            else:
                fig = px.bar(df, x=x_axis, y=y_axis, color=color_col)
                
        elif viz_type == "Line Chart":
            if y_axis == "None":
                fig = px.line(df, x=x_axis, color=color_col)
            else:
                fig = px.line(df, x=x_axis, y=y_axis, color=color_col)
                
        elif viz_type == "Scatter Plot":
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color_col)
            
        elif viz_type == "Box Plot":
            if y_axis == "None":
                fig = px.box(df, x=x_axis, color=color_col)
            else:
                fig = px.box(df, x=x_axis, y=y_axis, color=color_col)
                
        elif viz_type == "Histogram":
            fig = px.histogram(df, x=x_axis, color=color_col)
        
        # Grafik düzenlemeleri
        fig.update_layout(
            title=f"{viz_type} of {y_axis if y_axis != 'None' else x_axis}",
            xaxis_title=x_axis,
            yaxis_title=y_axis if y_axis != 'None' else "Count",
            template="plotly_dark"
        )
        
        # Grafiği göster
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")


with st.sidebar:
    st.header("Add your documents!")
    
    uploaded_file = st.file_uploader("Choose your `.xlsx` or `.csv` file", type=["xlsx", "xls", "csv"])

    if uploaded_file:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                file_type = "Excel" if uploaded_file.name.endswith((".xlsx", ".xls")) else "CSV"
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("Indexing your document...")

                if file_key not in st.session_state.get('file_cache', {}):
                    if os.path.exists(temp_dir):
                        docs = load_document(file_path, file_type)
                        
                        # Daha basit bir prompt template kullanalım
                        qa_prompt_tmpl_str = (
                            "Below is data from a {file_type} file.\n"
                            "---------------------\n"
                            "{context_str}\n"
                            "---------------------\n"
                            "Question: {query_str}\n"
                            "Please provide a clear and concise answer based on the data above.\n"
                            "If you need to calculate something, show your work.\n"
                            "Answer: "
                        )
                        qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

                        # Embedding modelini değiştirelim
                        embed_model = HuggingFaceEmbedding(
                            model_name="sentence-transformers/all-MiniLM-L6-v2",  # Daha hafif bir model
                            trust_remote_code=True
                        )
                        
                        Settings.embed_model = embed_model
                        Settings.llm = load_llm()
                        
                        index = VectorStoreIndex.from_documents(
                            documents=docs,
                            show_progress=True
                        )
                        
                        query_engine = index.as_query_engine(
                            streaming=True,
                            similarity_top_k=3  # Top 3 en alakalı sonucu al
                        )
                        
                        query_engine.update_prompts(
                            {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
                        )
                        
                        st.session_state.file_cache[file_key] = query_engine
                else:
                    query_engine = st.session_state.file_cache[file_key]

                # Inform the user that the file is processed and Display the file
                st.success("Ready to Chat!")
                display_file(uploaded_file, file_type)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()     

col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"RAG over Excel using Dockling 🐥 &  Llama-3.2")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept user input
if prompt := st.chat_input("What's up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Simulate stream of response with milliseconds delay
        streaming_response = query_engine.query(prompt)
        
        for chunk in streaming_response.response_gen:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        # full_response = query_engine.query(prompt)

        message_placeholder.markdown(full_response)
        # st.session_state.context = ctx

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})