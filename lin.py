import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain_community.llms.ollama import Ollama
import re
import time

# Configure the page
st.set_page_config(
    page_title="LinkedIn AI Analyzer",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple dark theme CSS
st.markdown("""
<style>
    /* Dark background for entire app */
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    
    /* Sidebar dark background */
    .css-1d391kg {
        background-color: #262730;
    }
    
    /* Main content area */
    .main .block-container {
        background-color: #0e1117;
        color: white;
    }
    
    /* All text white */
    body {
        color: white !important;
    }
    
    /* Chat messages - transparent with white text */
    .user-message {
        background-color: transparent;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 4px solid #0077b5;
        color: white;
    }
    
    .assistant-message {
        background-color: transparent;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 4px solid #00a0dc;
        color: white;
    }
    
    /* Remove all white backgrounds from Streamlit components */
    .stTextInput>div>div>input {
        background-color: #262730;
        color: white;
        border: 1px solid #555;
        border-radius: 4px;
        padding: 10px;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #0077b5;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: 500;
        width: 100%;
    }
    
    /* Metric containers - dark */
    [data-testid="metric-container"] {
        background-color: #262730;
        color: white;
        border: 1px solid #555;
        padding: 10px;
        border-radius: 5px;
    }
    
    /* Expander - dark */
    .streamlit-expanderHeader {
        background-color: #262730;
        color: white;
    }
    
    .streamlit-expanderContent {
        background-color: #262730;
        color: white;
    }
    
    /* Text area - dark */
    .stTextArea textarea {
        background-color: #262730;
        color: white;
    }
    
    /* Select boxes - dark */
    .stSelectbox>div>div>select {
        background-color: #262730;
        color: white;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #0077b5;
    }
    
    /* Success/Error messages - dark */
    .stSuccess {
        background-color: #1a3a1a;
        color: #90ee90;
    }
    
    .stError {
        background-color: #3a1a1a;
        color: #ff6b6b;
    }
    
    .stWarning {
        background-color: #3a3a1a;
        color: #ffff90;
    }
    
    .stInfo {
        background-color: #1a3a3a;
        color: #90ffff;
    }
    
    /* Header */
    .main-header {
        background: #0077b5;
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    /* Make all text white */
    h1, h2, h3, h4, h5, h6, p, div, span {
        color: white !important;
    }
    
    /* Chat input */
    .stChatInput>div>div>textarea {
        background-color: #262730;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def check_ollama_running():
    """Check if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_available_models():
    """Get list of available Ollama models"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return [model['name'] for model in models]
    except:
        return ["llama2", "mistral", "gemma"]

def extract_linkedin_data(url, data_type):
    """Extract data from LinkedIn with robust error handling"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            return f"❌ Failed to access page (Status: {response.status_code}). The page may be private or require login."
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove scripts and styles
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get all text and clean it
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Extract meaningful content (longer text blocks)
        paragraphs = text.split('.')
        meaningful_content = [p.strip() for p in paragraphs if len(p.strip()) > 50]
        
        if not meaningful_content:
            return "❌ No meaningful content found. This might be a private page or require LinkedIn login."
        
        # Structure the output based on data type
        if data_type == "profile":
            result = "👤 LINKEDIN PROFILE EXTRACTED DATA\n\n"
            result += f"🔗 Profile URL: {url}\n"
            result += "="*60 + "\n\n"
        elif data_type == "company":
            result = "🏢 LINKEDIN COMPANY EXTRACTED DATA\n\n"
            result += f"🔗 Company URL: {url}\n"
            result += "="*60 + "\n\n"
        else:  # post
            result = "📝 LINKEDIN POST EXTRACTED DATA\n\n"
            result += f"🔗 Post URL: {url}\n"
            result += "="*60 + "\n\n"
        
        # Add the most meaningful content blocks
        result += "📋 EXTRACTED CONTENT:\n\n"
        for i, content in enumerate(meaningful_content[:10], 1):  # Limit to 10 blocks
            result += f"{i}. {content}\n\n"
        
        result += "="*60 + "\n"
        result += f"✅ Successfully extracted {len(meaningful_content)} content blocks\n"
        result += f"📊 Total characters: {len(text)}\n"
        
        return result
        
    except Exception as e:
        return f"❌ Error extracting data: {str(e)}"

def get_text_chunks(text):
    """Split text into chunks for processing"""
    if not text.strip():
        return []
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)

def get_vectorstore(text_chunks):
    """Create vector store from text chunks"""
    if not text_chunks:
        return None
    documents = [Document(page_content=chunk) for chunk in text_chunks]
    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

def get_conversation_chain(vectorstore, model_name="llama2"):
    """Initialize conversation chain with Ollama"""
    if vectorstore is None:
        return None
    
    try:
        llm = Ollama(
            model=model_name,
            base_url="http://localhost:11434",
            temperature=0.7,
            top_p=0.9,
            num_predict=500
        )
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=True,
            output_key="answer"
        )
        
        return chain
        
    except Exception as e:
        st.error(f"❌ Error initializing AI model: {e}")
        return None

def display_message(role, content):
    """Display a chat message with simple formatting"""
    if role == "user":
        st.markdown(f"""
        <div class="user-message">
            <strong>👤 You:</strong><br>{content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-message">
            <strong>🤖 Assistant:</strong><br>{content}
        </div>
        """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>💼 LinkedIn AI Analyzer</h1>
        <p>Extract and analyze LinkedIn data with AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "extracted_data" not in st.session_state:
        st.session_state.extracted_data = ""
    if "data_type" not in st.session_state:
        st.session_state.data_type = "profile"
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Ollama status
        ollama_status = check_ollama_running()
        if ollama_status:
            st.success("✅ Ollama is running")
        else:
            st.error("❌ Ollama not detected")
            st.info("Please start Ollama: `ollama serve`")
        
        # Model selection
        available_models = get_available_models()
        model_name = st.selectbox(
            "🤖 Select AI Model",
            available_models if available_models else ["llama2"],
            help="Choose the Ollama model to use for analysis"
        )
        
        st.markdown("---")
        st.markdown("### 🔗 Data Source")
        
        # Data type selection
        data_type = st.selectbox(
            "📊 Content Type",
            ["profile", "company", "post"],
            help="Select the type of LinkedIn content to analyze"
        )
        
        # URL input
        url_placeholder = {
            "profile": "https://www.linkedin.com/in/username/",
            "company": "https://www.linkedin.com/company/companyname/",
            "post": "https://www.linkedin.com/posts/username_postid/"
        }
        
        linkedin_url = st.text_input(
            "🌐 LinkedIn URL",
            placeholder=url_placeholder[data_type],
            help="Enter a public LinkedIn URL"
        )
        
        # Extract button
        if st.button("🚀 Extract & Analyze", type="primary"):
            if not ollama_status:
                st.error("Ollama is not running. Please start it first.")
            elif not linkedin_url.strip():
                st.warning("Please enter a LinkedIn URL")
            else:
                with st.spinner("🔄 Extracting data from LinkedIn..."):
                    progress_bar = st.progress(0)
                    
                    # Step 1: Extract data
                    extracted_data = extract_linkedin_data(linkedin_url, data_type)
                    progress_bar.progress(33)
                    
                    if extracted_data and not extracted_data.startswith("❌"):
                        # Step 2: Process chunks
                        chunks = get_text_chunks(extracted_data)
                        progress_bar.progress(66)
                        
                        if chunks:
                            # Step 3: Create vector store and conversation chain
                            vectorstore = get_vectorstore(chunks)
                            conversation = get_conversation_chain(vectorstore, model_name)
                            progress_bar.progress(100)
                            
                            if conversation:
                                st.session_state.conversation = conversation
                                st.session_state.processed = True
                                st.session_state.extracted_data = extracted_data
                                st.session_state.data_type = data_type
                                st.session_state.chat_history = []
                                
                                st.success(f"✅ Success! Ready to analyze {len(chunks)} content chunks.")
                                
                                # Add welcome message
                                welcome_msg = f"I've analyzed this LinkedIn {data_type}. What would you like to know about it?"
                                st.session_state.chat_history.append({
                                    "role": "assistant", 
                                    "content": welcome_msg
                                })
                            else:
                                st.error("❌ Failed to initialize AI conversation")
                        else:
                            st.error("❌ No meaningful content could be extracted")
                    else:
                        st.error(extracted_data)
                    
                    progress_bar.empty()
        
        st.markdown("---")
        st.markdown("### 💡 Quick Actions")
        
        if st.session_state.processed:
            # Quick questions based on data type
            quick_questions = {
                "profile": [
                    "Summarize this person's background",
                    "What are their key skills and experience?",
                    "Tell me about their career journey",
                    "What industries have they worked in?"
                ],
                "company": [
                    "What does this company do?",
                    "Summarize the company's overview",
                    "What industry is this company in?",
                    "What are the company's main focus areas?"
                ],
                "post": [
                    "What is this post about?",
                    "Summarize the key points of this post",
                    "What insights can we gain from this content?",
                    "What is the main message of this post?"
                ]
            }
            
            st.markdown("**Quick Questions:**")
            for question in quick_questions[st.session_state.data_type]:
                if st.button(question, key=f"quick_{question}"):
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": question
                    })
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": ""
                    })
                    st.rerun()
            
            if st.button("🗑️ Clear Chat", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat interface
        st.markdown("### 💬 Analysis Chat")
        
        # Display chat history
        for i, chat in enumerate(st.session_state.chat_history):
            if chat["role"] == "user":
                display_message("user", chat["content"])
            elif chat["role"] == "assistant":
                if chat["content"]:  # Only display if there's content
                    display_message("assistant", chat["content"])
                else:
                    # Process pending assistant messages
                    with st.spinner("🤔 Analyzing..."):
                        try:
                            if st.session_state.conversation:
                                response = st.session_state.conversation.invoke({
                                    "question": st.session_state.chat_history[i-1]["content"]
                                })
                                answer = response.get("answer", "I couldn't generate a response for this question.")
                                st.session_state.chat_history[i]["content"] = answer
                                st.rerun()
                        except Exception as e:
                            st.session_state.chat_history[i]["content"] = f"❌ Error: {str(e)}"
                            st.rerun()
        
        # Chat input
        if st.session_state.processed:
            user_input = st.chat_input("Ask about the LinkedIn data...")
            
            if user_input:
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_input
                })
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": ""
                })
                st.rerun()
        
        else:
            # Welcome message
            st.info("""
            **👋 Welcome to LinkedIn AI Analyzer!**
            
            To get started:
            1. Ensure Ollama is running on your system
            2. Select the type of LinkedIn content you want to analyze
            3. Enter a public LinkedIn URL in the sidebar
            4. Click 'Extract & Analyze' to begin
            
            **Supported URLs:**
            - Profiles: `https://www.linkedin.com/in/username/`
            - Companies: `https://www.linkedin.com/company/companyname/`  
            - Posts: `https://www.linkedin.com/posts/username_postid/`
            """)
    
    with col2:
        if st.session_state.processed:
            st.markdown("### 📊 Data Overview")
            
            data = st.session_state.extracted_data
            chunks = get_text_chunks(data)
            
            # Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Content Type", st.session_state.data_type.title())
                st.metric("Text Chunks", len(chunks))
            
            with col2:
                st.metric("Characters", f"{len(data):,}")
                st.metric("Words", f"{len(data.split()):,}")
            
            # Data preview
            with st.expander("📋 View Extracted Data", expanded=False):
                st.text_area(
                    "Raw Data",
                    data,
                    height=200,
                    label_visibility="collapsed"
                )

if __name__ == "__main__":
    main()