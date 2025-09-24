import streamlit as st
import os
import pickle
import json
from io import BytesIO
import tempfile
from typing import List, Optional

import google.generativeai as genai
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import Document, BaseRetriever, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

# Page configuration
st.set_page_config(
    page_title="Google Drive AI Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'retrieval_chain' not in st.session_state:
    st.session_state.retrieval_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'credentials' not in st.session_state:
    st.session_state.credentials = None

# Configuration
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# OAuth configuration for Streamlit deployment
OAUTH_CONFIG = {
    "web": {
        "client_id": st.secrets.get("GOOGLE_CLIENT_ID", ""),
        "client_secret": st.secrets.get("GOOGLE_CLIENT_SECRET", ""),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": [st.secrets.get("REDIRECT_URI", "http://localhost:8501")]
    }
}

class SimpleTextRetriever(BaseRetriever):
    """Simple text-based retriever for when embeddings fail"""
    
    def __init__(self, documents: List[Document]):
        super().__init__()
        self._documents = documents
        self._texts = [doc.page_content.lower() for doc in documents]
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_lower = query.lower()
        relevant_docs = []
        
        # Simple keyword matching
        for i, text in enumerate(self._texts):
            if any(word in text for word in query_lower.split()):
                relevant_docs.append(self._documents[i])
        
        # If no matches, return all documents
        if not relevant_docs:
            relevant_docs = self._documents
            
        return relevant_docs[:5]

def create_simple_text_store(documents: List[Document]):
    """Create simple text store without embeddings"""
    class SimpleTextStore:
        def __init__(self, documents):
            self._documents = documents
        
        def as_retriever(self):
            return SimpleTextRetriever(self._documents)
    
    return SimpleTextStore(documents)

def get_auth_url():
    """Generate Google OAuth authorization URL"""
    try:
        flow = Flow.from_client_config(
            OAUTH_CONFIG,
            scopes=SCOPES,
            redirect_uri=OAUTH_CONFIG["web"]["redirect_uris"][0]
        )
        
        auth_url, _ = flow.authorization_url(prompt='consent')
        return flow, auth_url
    except Exception as e:
        st.error(f"Error setting up OAuth: {e}")
        return None, None

def handle_oauth_callback(authorization_code: str, flow):
    """Handle OAuth callback and get credentials"""
    try:
        flow.fetch_token(code=authorization_code)
        return flow.credentials
    except Exception as e:
        st.error(f"Error getting credentials: {e}")
        return None

def list_drive_files(service, folder_id: Optional[str] = None, search_term: Optional[str] = None):
    """List files from Google Drive"""
    try:
        supported_types = [
            'application/vnd.google-apps.document',
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/vnd.google-apps.spreadsheet',
            'text/plain',
            'application/msword',
            'application/vnd.google-apps.shortcut'
        ]
        
        if search_term:
            query = f"name contains '{search_term}' and trashed=false and (" + " or ".join([f"mimeType='{mime}'" for mime in supported_types]) + ")"
        elif folder_id:
            query = f"'{folder_id}' in parents and trashed=false and (" + " or ".join([f"mimeType='{mime}'" for mime in supported_types]) + ")"
        else:
            query = "trashed=false and (" + " or ".join([f"mimeType='{mime}'" for mime in supported_types]) + ")"
        
        results = service.files().list(
            q=query,
            fields="files(id, name, mimeType, shortcutDetails)",
            pageSize=50
        ).execute()
        
        return results.get('files', [])
    except Exception as e:
        st.error(f"Error listing files: {e}")
        return []

def process_file(service, file_info: dict) -> Optional[Document]:
    """Process a single file and return Document"""
    file_id = file_info['id']
    file_name = file_info['name']
    mime_type = file_info['mimeType']
    
    try:
        content = ""
        
        if mime_type == 'application/vnd.google-apps.document':
            # Google Docs
            request = service.files().export_media(fileId=file_id, mimeType='text/plain')
            file_content = request.execute()
            content = file_content.decode('utf-8')
            
        elif mime_type == 'application/vnd.google-apps.spreadsheet':
            # Google Sheets
            request = service.files().export_media(fileId=file_id, mimeType='text/csv')
            file_content = request.execute()
            content = file_content.decode('utf-8')
            
        elif mime_type == 'application/vnd.google-apps.shortcut':
            # Handle shortcuts
            shortcut_details = file_info.get('shortcutDetails', {})
            target_id = shortcut_details.get('targetId')
            target_mime_type = shortcut_details.get('targetMimeType')
            
            if not target_id:
                return None
                
            if target_mime_type == 'application/vnd.google-apps.document':
                request = service.files().export_media(fileId=target_id, mimeType='text/plain')
                file_content = request.execute()
                content = file_content.decode('utf-8')
            elif target_mime_type == 'application/vnd.google-apps.spreadsheet':
                request = service.files().export_media(fileId=target_id, mimeType='text/csv')
                file_content = request.execute()
                content = file_content.decode('utf-8')
            else:
                return None
                
        elif mime_type == 'application/pdf':
            # Basic PDF support - would need PyPDF2 for full support
            try:
                import PyPDF2
                request = service.files().get_media(fileId=file_id)
                file_content = request.execute()
                pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
                content = ""
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n"
            except ImportError:
                st.warning("PyPDF2 not installed - skipping PDF files")
                return None
            except Exception:
                return None
                
        else:
            return None
        
        if content.strip():
            return Document(
                page_content=content.strip(),
                metadata={
                    'source': file_name,
                    'file_id': file_id,
                    'mime_type': mime_type
                }
            )
    except Exception as e:
        st.warning(f"Error processing {file_name}: {e}")
        return None

def create_vector_store(documents: List[Document], gemini_api_key: str):
    """Create vector store with fallback to simple text search"""
    if not documents:
        return None
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    
    # Try embeddings first, fall back to simple search
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=gemini_api_key
        )
        vector_store = FAISS.from_documents(chunks, embeddings)
        st.success(f"Created vector store with embeddings for {len(chunks)} chunks")
        return vector_store
    except Exception as e:
        st.warning(f"Embeddings failed: {str(e)[:100]}...")
        st.info("Using simple text search instead")
        return create_simple_text_store(documents)

def main():
    st.title("📚 Google Drive AI Chat")
    st.markdown("Chat with your Google Drive documents using AI")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input
        gemini_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Enter your Google Gemini API key"
        )
        
        if not gemini_api_key:
            st.warning("Please enter your Gemini API key to continue")
            st.stop()
        
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        
        st.header("Google Drive Authentication")
        
        # OAuth flow
        if st.session_state.credentials is None:
            if st.button("🔐 Connect to Google Drive"):
                flow, auth_url = get_auth_url()
                if auth_url:
                    st.markdown(f"[Click here to authorize Google Drive access]({auth_url})")
                    st.session_state.flow = flow
            
            # Handle OAuth callback
            auth_code = st.text_input(
                "Authorization Code",
                help="Paste the authorization code from Google here"
            )
            
            if auth_code and hasattr(st.session_state, 'flow'):
                credentials = handle_oauth_callback(auth_code, st.session_state.flow)
                if credentials:
                    st.session_state.credentials = credentials
                    st.success("✅ Successfully connected to Google Drive!")
                    st.rerun()
        else:
            st.success("✅ Connected to Google Drive")
            if st.button("🔓 Disconnect"):
                st.session_state.credentials = None
                st.session_state.documents = []
                st.session_state.vector_store = None
                st.session_state.retrieval_chain = None
                st.rerun()
    
    # Main content
    if st.session_state.credentials is None:
        st.info("👈 Please connect to Google Drive using the sidebar")
        return
    
    # Create Google Drive service
    try:
        service = build('drive', 'v3', credentials=st.session_state.credentials)
    except Exception as e:
        st.error(f"Error creating Drive service: {e}")
        return
    
    # Document loading interface
    st.header("📂 Load Documents")
    
    col1, col2 = st.columns(2)
    
    with col1:
        search_term = st.text_input("🔍 Search documents", placeholder="Enter keywords to search...")
    
    with col2:
        folder_id = st.text_input("📁 Folder ID (optional)", placeholder="Google Drive folder ID")
    
    if st.button("🔄 Load Documents"):
        with st.spinner("Loading documents from Google Drive..."):
            files = list_drive_files(service, folder_id, search_term)
            
            if not files:
                st.warning("No matching documents found")
                return
            
            st.info(f"Found {len(files)} files. Processing...")
            
            documents = []
            progress_bar = st.progress(0)
            
            for i, file_info in enumerate(files):
                doc = process_file(service, file_info)
                if doc:
                    documents.append(doc)
                progress_bar.progress((i + 1) / len(files))
            
            if documents:
                st.session_state.documents = documents
                
                # Create vector store
                st.session_state.vector_store = create_vector_store(documents, gemini_api_key)
                
                # Create retrieval chain
                if st.session_state.vector_store:
                    model = ChatGoogleGenerativeAI(
                        model="gemini-1.5-pro-latest",
                        temperature=0.7,
                        google_api_key=gemini_api_key
                    )
                    
                    st.session_state.retrieval_chain = ConversationalRetrievalChain.from_llm(
                        llm=model,
                        retriever=st.session_state.vector_store.as_retriever(),
                        return_source_documents=True
                    )
                
                st.success(f"✅ Loaded {len(documents)} documents successfully!")
            else:
                st.error("No documents could be processed")
    
    # Document viewer
    if st.session_state.documents:
        st.header("📄 Loaded Documents")
        
        with st.expander(f"View {len(st.session_state.documents)} loaded documents"):
            for i, doc in enumerate(st.session_state.documents, 1):
                st.subheader(f"{i}. {doc.metadata.get('source', 'Unknown')}")
                st.caption(f"Type: {doc.metadata.get('mime_type', 'Unknown')} | Length: {len(doc.page_content)} characters")
                
                # Show preview
                preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                st.text_area(
                    f"Preview of document {i}",
                    value=preview,
                    height=150,
                    disabled=True,
                    key=f"preview_{i}"
                )
    
    # Chat interface
    if st.session_state.retrieval_chain:
        st.header("💬 Chat with Your Documents")
        
        # Display chat history
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                st.write(answer)
        
        # Chat input
        user_question = st.chat_input("Ask a question about your documents...")
        
        if user_question:
            # Add user message to chat
            with st.chat_message("user"):
                st.write(user_question)
            
            # Generate response
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Thinking..."):
                        result = st.session_state.retrieval_chain({
                            "question": user_question,
                            "chat_history": st.session_state.chat_history
                        })
                        answer = result['answer']
                        st.write(answer)
                        
                        # Show sources if available
                        if 'source_documents' in result and result['source_documents']:
                            with st.expander("📚 Sources"):
                                for doc in result['source_documents']:
                                    st.caption(f"From: {doc.metadata.get('source', 'Unknown')}")
                                    st.text(doc.page_content[:200] + "...")
                        
                        # Add to history
                        st.session_state.chat_history.append((user_question, answer))
                
                except Exception as e:
                    st.error(f"Error generating response: {str(e)[:200]}...")
                    
                    # Fallback: show relevant document snippets
                    try:
                        st.info("Showing relevant document content instead:")
                        if hasattr(st.session_state.retrieval_chain.retriever, '_documents'):
                            relevant_docs = st.session_state.retrieval_chain.retriever._get_relevant_documents(user_question)
                            
                            for doc in relevant_docs[:2]:
                                st.caption(f"From: {doc.metadata.get('source', 'Unknown')}")
                                
                                # Find relevant snippet
                                content = doc.page_content.lower()
                                query_words = user_question.lower().split()
                                
                                best_snippet = ""
                                for word in query_words:
                                    if word in content:
                                        word_pos = content.find(word)
                                        start = max(0, word_pos - 200)
                                        end = min(len(content), word_pos + 200)
                                        snippet = doc.page_content[start:end]
                                        if len(snippet) > len(best_snippet):
                                            best_snippet = snippet
                                
                                if best_snippet:
                                    st.text_area("Relevant content", best_snippet, height=100)
                                else:
                                    preview = doc.page_content[:300]
                                    st.text_area("Document preview", preview, height=100)
                    except Exception as fallback_error:
                        st.error(f"Fallback also failed: {fallback_error}")

if __name__ == "__main__":
    main()