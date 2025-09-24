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
import time
from urllib.parse import urlparse, urljoin
import pandas as pd

def check_ollama_running():
    """Check if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            st.sidebar.success("✅ Ollama is running")
            return True
    except:
        st.sidebar.error("❌ Ollama is not running. Please run: `ollama serve`")
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

def extract_linkedin_profile(profile_url):
    """Extract data from LinkedIn public profile"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        response = requests.get(profile_url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            return f"Failed to access profile. Status: {response.status_code}"
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract profile information
        profile_data = {}
        
        # Name
        name_element = soup.find('h1', class_=['top-card-layout__title', 'text-heading-xlarge'])
        if not name_element:
            name_element = soup.find('h1')
        profile_data['name'] = name_element.get_text(strip=True) if name_element else "Not found"
        
        # Headline
        headline_element = soup.find('h2', class_=['top-card-layout__headline', 'text-body-medium'])
        if not headline_element:
            headline_element = soup.find('h2')
        profile_data['headline'] = headline_element.get_text(strip=True) if headline_element else "Not found"
        
        # About section
        about_element = soup.find('section', class_=['summary', 'about'])
        if not about_element:
            about_element = soup.find('div', class_=['core-section-container__content', 'break-words'])
        profile_data['about'] = about_element.get_text(strip=True) if about_element else "Not found"
        
        # Experience
        experience_section = soup.find('section', class_=['experience', 'experience-section'])
        experiences = []
        if experience_section:
            exp_items = experience_section.find_all('li', class_=['experience-item', 'experience-list__item'])
            for exp in exp_items[:5]:
                try:
                    title_elem = exp.find(['h3', 'h4'])
                    company_elem = exp.find(['h4', 'h5'])
                    duration_elem = exp.find('span', class_=['date-range', 'experience-item__duration'])
                    
                    experience = {
                        'title': title_elem.get_text(strip=True) if title_elem else "Not specified",
                        'company': company_elem.get_text(strip=True) if company_elem else "Not specified",
                        'duration': duration_elem.get_text(strip=True) if duration_elem else "Not specified"
                    }
                    experiences.append(experience)
                except:
                    continue
        profile_data['experiences'] = experiences
        
        # Education
        education_section = soup.find('section', class_=['education', 'education-section'])
        educations = []
        if education_section:
            edu_items = education_section.find_all('li', class_=['education__item', 'education-list__item'])
            for edu in edu_items[:3]:
                try:
                    school_elem = edu.find(['h3', 'h4'])
                    degree_elem = edu.find(['h4', 'h5'])
                    duration_elem = edu.find('span', class_=['date-range', 'education__item--duration'])
                    
                    education = {
                        'school': school_elem.get_text(strip=True) if school_elem else "Not specified",
                        'degree': degree_elem.get_text(strip=True) if degree_elem else "Not specified",
                        'duration': duration_elem.get_text(strip=True) if duration_elem else "Not specified"
                    }
                    educations.append(education)
                except:
                    continue
        profile_data['educations'] = educations
        
        # Format the data
        result = f"LINKEDIN PROFILE ANALYSIS\n\n"
        result += f"Profile URL: {profile_url}\n"
        result += f"Name: {profile_data['name']}\n"
        result += f"Headline: {profile_data['headline']}\n"
        result += "="*60 + "\n\n"
        
        result += "ABOUT:\n"
        result += f"{profile_data['about']}\n\n"
        
        result += "EXPERIENCE:\n"
        for i, exp in enumerate(profile_data['experiences'], 1):
            result += f"{i}. {exp['title']} at {exp['company']} ({exp['duration']})\n"
        result += "\n"
        
        result += "EDUCATION:\n"
        for i, edu in enumerate(profile_data['educations'], 1):
            result += f"{i}. {edu['degree']} at {edu['school']} ({edu['duration']})\n"
        
        return result
        
    except Exception as e:
        return f"Error extracting profile: {str(e)}"

def get_linkedin_data(url, data_type):
    """Main function to extract LinkedIn data based on URL type"""
    return extract_linkedin_profile(url)

def get_text_chunks(text):
    if not text.strip():
        return []
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def get_vectorstore(text_chunks):
    if not text_chunks:
        return None
    documents = [Document(page_content=chunk) for chunk in text_chunks]
    vectorstore = FAISS.from_documents(documents, SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2'))
    return vectorstore

def get_conversation_chain(vectorstore, model_name="llama2"):
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
        st.error(f"Error initializing Ollama: {e}")
        return None

def display_chat_message(role, content, avatar):
    """Display a chat message with beautiful formatting"""
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)

def main():
    st.set_page_config(page_title="LinkedIn Data Extractor", page_icon="💼", layout="wide")
    st.title("💼 LinkedIn Public Data Extractor")
    st.caption("Extract and analyze public LinkedIn profiles")
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #f0f2f6;
        border-left: 4px solid #4CAF50;
    }
    .chat-message.assistant {
        background-color: #e8f4fd;
        border-left: 4px solid #2196F3;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 1.2em;
    }
    .chat-message.user .avatar {
        background-color: #4CAF50;
        color: white;
    }
    .chat-message.assistant .avatar {
        background-color: #2196F3;
        color: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    ollama_running = check_ollama_running()
    
    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed" not in st.session_state:
        st.session_state.processed = False

    with st.sidebar:
        st.header("⚙️ Configuration")
        
        available_models = get_available_models()
        if available_models:
            model_name = st.selectbox("Select Ollama Model", available_models)
        else:
            model_name = st.text_input("Ollama Model Name", "llama2")
        
        st.header("🔗 LinkedIn URL")
        linkedin_url = st.text_input(
            "LinkedIn Public Profile URL",
            placeholder="https://www.linkedin.com/in/username/",
            help="Enter a public LinkedIn profile URL"
        )
        
        if st.button("🚀 Extract LinkedIn Data", type="primary", use_container_width=True):
            if not ollama_running:
                st.error("Please start Ollama first: `ollama serve`")
            elif not linkedin_url.strip():
                st.warning("Please enter a LinkedIn URL")
            else:
                with st.spinner("Extracting LinkedIn data..."):
                    extracted_data = get_linkedin_data(linkedin_url, "profile")
                    
                    if extracted_data and not extracted_data.startswith("Error") and not extracted_data.startswith("Failed"):
                        chunks = get_text_chunks(extracted_data)
                        if chunks:
                            vectorstore = get_vectorstore(chunks)
                            st.session_state.vectorstore = vectorstore
                            st.session_state.conversation = get_conversation_chain(vectorstore, model_name)
                            
                            if st.session_state.conversation:
                                st.session_state.processed = True
                                st.session_state.extracted_data = extracted_data
                                st.success(f"✅ Success! Extracted {len(chunks)} data chunks.")
                            else:
                                st.error("❌ Failed to initialize AI model.")
                        else:
                            st.error("❌ No data extracted.")
                    else:
                        st.error(f"❌ Extraction failed: {extracted_data}")

        # Quick action buttons
        if st.session_state.processed:
            st.header("💡 Quick Questions")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 Summary", use_container_width=True):
                    st.session_state.chat_history.append({
                        "question": "Can you provide a comprehensive summary of this profile?",
                        "answer": ""
                    })
            with col2:
                if st.button("💼 Experience", use_container_width=True):
                    st.session_state.chat_history.append({
                        "question": "What is their professional experience background?",
                        "answer": ""
                    })
            
            col3, col4 = st.columns(2)
            with col3:
                if st.button("🎓 Education", use_container_width=True):
                    st.session_state.chat_history.append({
                        "question": "Tell me about their educational background",
                        "answer": ""
                    })
            with col4:
                if st.button("🌟 Skills", use_container_width=True):
                    st.session_state.chat_history.append({
                        "question": "What skills and expertise does this person have?",
                        "answer": ""
                    })

    # Main content area
    if st.session_state.processed:
        # Data overview cards
        st.header("📈 Data Overview")
        data = st.session_state.extracted_data
        chunks = get_text_chunks(data)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Data Length", f"{len(data):,} chars")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Text Chunks", len(chunks))
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Words", f"{len(data.split()):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Lines", data.count('\n') + 1)
            st.markdown('</div>', unsafe_allow_html=True)

        # Chat interface
        st.header("💬 Analyze LinkedIn Data")
        
        # Display chat history
        for i, chat in enumerate(st.session_state.chat_history):
            if chat["question"]:
                display_chat_message("user", f"**{chat['question']}**", "👤")
                
                if chat["answer"]:
                    display_chat_message("assistant", chat["answer"], "🤖")
                else:
                    # Process unanswered questions
                    with st.chat_message("assistant", avatar="🤖"):
                        with st.spinner("Analyzing..."):
                            try:
                                response = st.session_state.conversation.invoke({"question": chat["question"]})
                                answer = response.get("answer", "I couldn't generate a response for this question.")
                                st.session_state.chat_history[i]["answer"] = answer
                                st.markdown(answer)
                            except Exception as e:
                                error_msg = f"Error processing question: {e}"
                                st.session_state.chat_history[i]["answer"] = error_msg
                                st.error(error_msg)
                    st.rerun()
        
        # Chat input
        user_question = st.chat_input("Ask about the LinkedIn profile...")
        
        if user_question:
            # Add user question to chat history
            st.session_state.chat_history.append({"question": user_question, "answer": ""})
            st.rerun()

        # Clear chat button
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()

        # Data preview
        with st.expander("📋 View Extracted Data", expanded=False):
            st.text_area("Extracted Profile Data", data, height=300, label_visibility="collapsed")

    else:
        # Welcome state
        st.markdown("""
        <div style='text-align: center; padding: 4rem 2rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 1rem; margin: 2rem 0;'>
            <h2 style='color: #2c3e50; margin-bottom: 1rem;'>🔍 Analyze LinkedIn Profiles</h2>
            <p style='color: #34495e; font-size: 1.2rem; margin-bottom: 2rem;'>
                Enter a LinkedIn profile URL to extract and analyze professional data using AI
            </p>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>👆</div>
            <p style='color: #7f8c8d;'>Get started by entering a LinkedIn URL in the sidebar</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Example URLs
        with st.expander("🌐 Example LinkedIn Profile URLs"):
            st.write("""
            **Public Profile Examples:**
            - `https://www.linkedin.com/in/satyanadella/` (Microsoft CEO)
            - `https://www.linkedin.com/in/tim-cook/` (Apple CEO)
            - `https://www.linkedin.com/in/sundarpichai/` (Google CEO)
            
            **Note:** Only public profiles are accessible. Make sure the profile has public visibility settings.
            """)

if __name__ == "__main__":
    main()