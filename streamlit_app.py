import streamlit as st
import json
from io import BytesIO
from typing import List, Optional

import google.generativeai as genai
from google.oauth2 import service_account
from googleapiclient.discovery import build

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import Document, BaseRetriever
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

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

class SimpleTextRetriever(BaseRetriever):
    """Simple text-based retriever for when embeddings fail"""
    
    def __init__(self, documents: List[Document]):
        super().__init__()
        self._documents = documents
        self._texts = [doc.page_content.lower() for doc in documents]
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_lower = query.lower()
        relevant_docs = []
        
        for i, text in enumerate(self._texts):
            if any(word in text for word in query_lower.split()):
                relevant_docs.append(self._documents[i])
        
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

def get_drive_service():
    """Create Google Drive service using service account credentials"""
    try:
        # Get service account info from Streamlit secrets
        service_account_info = st.secrets["gcp_service_account"]
        
        # Create credentials
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=SCOPES
        )
        
        # Build and return service
        return build('drive', 'v3', credentials=credentials)
    except Exception as e:
        st.error(f"Error creating Drive service: {e}")
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
            request = service.files().export_media(fileId=file_id, mimeType='text/plain')
            file_content = request.execute()
            content = file_content.decode('utf-8')
            
        elif mime_type == 'application/vnd.google-apps.spreadsheet':
            request = service.files().export_media(fileId=file_id, mimeType='text/csv')
            file_content = request.execute()
            content = file_content.decode('utf-8')
            
        elif mime_type == 'application/vnd.google-apps.shortcut':
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
            try:
                import PyPDF2
                request = service.files().get_media(fileId=file_id)
                file_content = request.execute()
                pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
                content = ""
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n"
            except:
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
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=gemini_api_key
        )
        vector_store = FAISS.from_documents(chunks, embeddings)
        st.success(f"✅ Created vector store with embeddings for {len(chunks)} chunks")
        return vector_store
    except Exception as e:
        st.warning(f"⚠️ Embeddings failed: {str(e)[:100]}...")
        st.info("📝 Using simple text search instead")
        return create_simple_text_store(documents)

def main():
    st.title("📚 Google Drive AI Chat")
    st.markdown("Chat with your Google Drive documents using AI")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        gemini_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Enter your Google Gemini API key",
            value=st.session_state.get('gemini_api_key', '')
        )
        
        if gemini_api_key:
            st.session_state.gemini_api_key = gemini_api_key
            genai.configure(api_key=gemini_api_key)
        else:
            st.warning("⚠️ Please enter your Gemini API key")
            st.info("""
            Get your API key from:
            [Google AI Studio](https://aistudio.google.com/)
            """)
            st.stop()
        
        st.divider()
        
        # Check for service account credentials
        if "gcp_service_account" not in st.secrets:
            st.error("❌ Service account not configured")
            st.info("""
            **Setup Instructions:**
            
            1. Create a service account in Google Cloud Console
            2. Download the JSON key file
            3. Share your Google Drive files with the service account email
            4. Add the JSON content to Streamlit secrets as `gcp_service_account`
            
            [Learn more](https://docs.streamlit.io/knowledge-base/tutorials/databases/gcs)
            """)
            st.stop()
        else:
            st.success("✅ Service account configured")
            service_email = st.secrets["gcp_service_account"].get("client_email", "")
            if service_email:
                st.caption(f"📧 {service_email}")
                st.info(f"""
                **Important:** Share your Google Drive folders/files with:
                
                `{service_email}`
                """)
    
    # Create Drive service
    service = get_drive_service()
    if not service:
        st.error("Failed to connect to Google Drive")
        st.stop()
    
    # Document loading interface
    st.header("📂 Load Documents")
    
    col1, col2 = st.columns(2)
    
    with col1:
        search_term = st.text_input("🔍 Search documents", placeholder="Enter keywords...")
    
    with col2:
        folder_id = st.text_input("📁 Folder ID (optional)", placeholder="Google Drive folder ID")
    
    if st.button("🔄 Load Documents", use_container_width=True):
        with st.spinner("Loading documents from Google Drive..."):
            files = list_drive_files(service, folder_id, search_term)
            
            if not files:
                st.warning("No matching documents found")
                st.info("Make sure the service account has access to your files")
            else:
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
                    
                    st.session_state.vector_store = create_vector_store(documents, gemini_api_key)
                    
                    if st.session_state.vector_store:
                        # Store the API key for later use instead of creating model now
                        st.session_state.gemini_configured = True
                    
                    st.success(f"✅ Loaded {len(documents)} documents successfully!")
                else:
                    st.error("No documents could be processed")
    
    # Document viewer
    if st.session_state.documents:
        st.header("📄 Loaded Documents")
        
        with st.expander(f"📋 View {len(st.session_state.documents)} loaded documents"):
            for i, doc in enumerate(st.session_state.documents, 1):
                st.subheader(f"{i}. {doc.metadata.get('source', 'Unknown')}")
                st.caption(f"Type: {doc.metadata.get('mime_type', 'Unknown')} | Length: {len(doc.page_content)} characters")
                
                preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                st.text_area(
                    f"Preview",
                    value=preview,
                    height=150,
                    disabled=True,
                    key=f"preview_{i}"
                )
    
    # Chat interface
    if st.session_state.vector_store and st.session_state.get('gemini_configured'):
        st.header("💬 Chat with Your Documents")
        
        for question, answer in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                st.write(answer)
        
        user_question = st.chat_input("Ask a question about your documents...")
        
        if user_question:
            with st.chat_message("user"):
                st.write(user_question)
            
            with st.chat_message("assistant"):
                try:
                    # Get relevant documents
                    if hasattr(st.session_state.vector_store, 'as_retriever'):
                        retriever = st.session_state.vector_store.as_retriever()
                        if hasattr(retriever, '_get_relevant_documents'):
                            relevant_docs = retriever._get_relevant_documents(user_question)
                        else:
                            relevant_docs = retriever.get_relevant_documents(user_question)
                    else:
                        relevant_docs = st.session_state.documents
                    
                    # Create context from documents
                    context = "\n\n".join([doc.page_content[:1000] for doc in relevant_docs[:3]])
                    
                    # Use native Gemini API directly
                    with st.spinner("Thinking..."):
                        model = genai.GenerativeModel('gemini-pro')
                        prompt = f"""Based on the following document content, please answer this question: {user_question}

Document content:
{context}

Please provide a helpful and accurate answer based only on the information provided."""
                        
                        response = model.generate_content(prompt)
                        answer = response.text
                        
                        st.write(answer)
                        
                        # Show sources
                        with st.expander("📚 Sources"):
                            for doc in relevant_docs[:3]:
                                st.caption(f"From: {doc.metadata.get('source', 'Unknown')}")
                                st.text(doc.page_content[:200] + "...")
                        
                        st.session_state.chat_history.append((user_question, answer))
                
                except Exception as e:
                    st.error(f"⚠️ Error: {str(e)[:200]}")
                    
                    # Fallback to showing document snippets
                    try:
                        st.info("📝 Showing relevant document content instead:")
                        if hasattr(st.session_state.vector_store, 'as_retriever'):
                            retriever = st.session_state.vector_store.as_retriever()
                            if hasattr(retriever, '_get_relevant_documents'):
                                relevant_docs = retriever._get_relevant_documents(user_question)
                            else:
                                relevant_docs = st.session_state.documents
                        else:
                            relevant_docs = st.session_state.documents
                        
                        for doc in relevant_docs[:2]:
                            st.caption(f"From: {doc.metadata.get('source', 'Unknown')}")
                            
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
                                st.text_area("Relevant content", best_snippet, height=100, key=f"snippet_{doc.metadata.get('file_id', 'unknown')}")
                            else:
                                preview = doc.page_content[:300]
                                st.text_area("Document preview", preview, height=100, key=f"preview_{doc.metadata.get('file_id', 'unknown')}")
                    except Exception as fallback_error:
                        st.error(f"Fallback also failed: {fallback_error}")

if __name__ == "__main__":
    main()