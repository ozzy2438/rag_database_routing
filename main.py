import os

import gc
import tempfile
import uuid
import pandas as pd

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
    """Display Excel or CSV file preview"""
    st.markdown(f"### {file_type} Preview")
    if file_type == "Excel":
        df = pd.read_excel(file)
    else:  # CSV
        df = pd.read_csv(file)
    st.dataframe(df)


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