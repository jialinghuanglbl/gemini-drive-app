import streamlit as st
import json
from io import BytesIO
from typing import List, Optional

import google.generativeai as genai
from google.oauth2 import service_account
from googleapiclient.discovery import build

from langchain.schema import Document, BaseRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

def find_relevant_excerpts(content: str, query: str, answer: str = "", num_excerpts: int = 3, excerpt_length: int = 400) -> List[tuple[str, int]]:
    """
    Find the most relevant excerpts from content based on query and answer.
    Returns list of (excerpt, relevance_score) tuples.
    """
    content_lower = content.lower()
    query_words = [w.strip('?.,!;:') for w in query.lower().split() if len(w) > 3]
    
    # Also extract key terms from the answer if provided
    answer_words = []
    if answer:
        answer_words = [w.strip('?.,!;:') for w in answer.lower().split() if len(w) > 4]
        # Filter out common words
        common_words = {'this', 'that', 'these', 'those', 'there', 'where', 'when', 'what', 
                       'which', 'while', 'with', 'about', 'would', 'could', 'should', 'their',
                       'document', 'section', 'states', 'mentions', 'based', 'provided'}
        answer_words = [w for w in answer_words if w not in common_words][:10]  # Top 10 answer keywords
    
    all_search_words = query_words + answer_words
    
    if not all_search_words:
        return [(content[:excerpt_length] + "..." if len(content) > excerpt_length else content, 0)]
    
    # Split content into chunks with overlap
    chunk_size = excerpt_length
    chunks = []
    for i in range(0, len(content), chunk_size // 3):  # 66% overlap for better coverage
        chunk_text = content[i:i + chunk_size]
        if chunk_text.strip():
            chunks.append((chunk_text, i))
    
    # Score each chunk
    scored_chunks = []
    for chunk_text, start_pos in chunks:
        chunk_lower = chunk_text.lower()
        score = 0
        
        # Higher weight for query words (what the user asked)
        for word in query_words:
            count = chunk_lower.count(word)
            score += count * 3
        
        # Medium weight for answer keywords (what the AI said)
        for word in answer_words:
            count = chunk_lower.count(word)
            score += count * 2
        
        # Bonus for having both query AND answer terms
        has_query_term = any(word in chunk_lower for word in query_words)
        has_answer_term = any(word in chunk_lower for word in answer_words)
        if has_query_term and has_answer_term:
            score += 10
        
        # Count unique words found
        unique_query_words = sum(1 for word in query_words if word in chunk_lower)
        unique_answer_words = sum(1 for word in answer_words if word in chunk_lower)
        score += unique_query_words * 4
        score += unique_answer_words * 2
        
        # Bonus for word density (multiple relevant words in small area)
        positions = []
        for word in all_search_words:
            pos = 0
            while pos < len(chunk_lower):
                pos = chunk_lower.find(word, pos)
                if pos == -1:
                    break
                positions.append(pos)
                pos += 1
        
        if len(positions) > 2:
            # Calculate how clustered the words are
            positions.sort()
            gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
            avg_gap = sum(gaps) / len(gaps) if gaps else chunk_size
            if avg_gap < chunk_size / 3:  # Words are clustered
                score += 8
        
        # Boost for exact phrase matches
        query_phrases = []
        if len(query.split()) >= 2:
            words = query.lower().split()
            for i in range(len(words) - 1):
                phrase = ' '.join(words[i:i+2])
                if phrase in chunk_lower:
                    score += 15
        
        if score > 0:
            scored_chunks.append((chunk_text, start_pos, score))
    
    # Sort by score
    scored_chunks.sort(key=lambda x: x[2], reverse=True)
    
    if not scored_chunks:
        return [(content[:excerpt_length] + "..." if len(content) > excerpt_length else content, 0)]
    
    # Format results
    results = []
    seen_positions = set()
    
    for chunk_text, start_pos, score in scored_chunks[:num_excerpts * 2]:  # Get more candidates
        # Avoid duplicate/overlapping excerpts
        if any(abs(start_pos - seen) < chunk_size // 2 for seen in seen_positions):
            continue
        
        seen_positions.add(start_pos)
        
        formatted = chunk_text.strip()
        if start_pos > 0:
            formatted = "..." + formatted
        if start_pos + len(chunk_text) < len(content):
            formatted = formatted + "..."
        
        results.append((formatted, score))
        
        if len(results) >= num_excerpts:
            break
    
    return results if results else [(content[:excerpt_length] + "...", 0)]

def extract_drive_id_from_url(url: str) -> tuple[Optional[str], str]:
    """
    Extract Google Drive file/folder ID from URL
    Returns: (id, type) where type is 'file' or 'folder'
    """
    import re
    
    # Pattern for folder URLs: /folders/FOLDER_ID or ?id=FOLDER_ID
    folder_patterns = [
        r'/folders/([a-zA-Z0-9_-]+)',
        r'[?&]id=([a-zA-Z0-9_-]+)'
    ]
    
    # Pattern for file URLs: /d/FILE_ID or ?id=FILE_ID
    file_patterns = [
        r'/d/([a-zA-Z0-9_-]+)',
        r'/file/d/([a-zA-Z0-9_-]+)',
        r'[?&]id=([a-zA-Z0-9_-]+)'
    ]
    
    # Check for folder
    if '/folders/' in url or 'folder' in url.lower():
        for pattern in folder_patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1), 'folder'
    
    # Check for file
    for pattern in file_patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1), 'file'
    
    return None, 'unknown'

def list_drive_folders(service, parent_id: Optional[str] = 'root', search_term: Optional[str] = None):
    """List folders from Google Drive"""
    try:
        if search_term:
            query = f"mimeType='application/vnd.google-apps.folder' and trashed=false and name contains '{search_term}'"
        else:
            query = f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        
        results = service.files().list(
            q=query,
            fields="files(id, name)",
            pageSize=50,
            orderBy="name"
        ).execute()
        
        return results.get('files', [])
    except Exception as e:
        st.error(f"Error listing folders: {e}")
        return []

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
    """Create simple text store (no embeddings to avoid quota issues)"""
    if not documents:
        return None
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    
    # Skip embeddings entirely, go straight to simple text search
    st.info(f"Using simple text search for {len(documents)} documents")
    return create_simple_text_store(documents)

def main():
    st.title("📚 Google Drive AI Chat")
    st.markdown("Chat with your Google Drive documents using AI")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Check for API key in secrets first
        default_api_key = st.secrets.get("GEMINI_API_KEY", "")
        
        if default_api_key:
            st.success("✅ Gemini API key loaded from secrets")
            gemini_api_key = default_api_key
            # Show option to override
            if st.checkbox("Use different API key"):
                gemini_api_key = st.text_input(
                    "Gemini API Key",
                    type="password",
                    help="Enter a different Google Gemini API key"
                )
        else:
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
    
    # Option selector
    load_option = st.radio(
        "Choose loading method:",
        ["Paste URL", "Search files", "Browse by folder", "Search folders"],
        horizontal=True
    )
    
    if load_option == "Paste URL":
        drive_url = st.text_input(
            "🔗 Google Drive URL", 
            placeholder="Paste link to file or folder (e.g., https://drive.google.com/drive/folders/...)",
            help="Paste any Google Drive file or folder URL"
        )
        
        if drive_url:
            file_id, file_type = extract_drive_id_from_url(drive_url)
            
            if file_id:
                if file_type == 'folder':
                    st.success(f"✅ Detected folder ID: `{file_id}`")
                    folder_id = file_id
                    search_term = None
                    folder_search = None
                else:
                    st.success(f"✅ Detected file ID: `{file_id}`")
                    # Store as a list for processing
                    folder_id = None
                    search_term = None
                    folder_search = None
                    st.session_state.url_file_id = file_id
            else:
                st.error("❌ Could not extract ID from URL. Make sure you're pasting a valid Google Drive link.")
                folder_id = None
                search_term = None
                folder_search = None
        else:
            folder_id = None
            search_term = None
            folder_search = None
            
    elif load_option == "Search files":
        search_term = st.text_input("🔍 Search for files", placeholder="Enter keywords to find files...")
        folder_id = None
        folder_search = None
        
    elif load_option == "Browse by folder":
        folder_id = st.text_input("📁 Folder ID", placeholder="Paste Google Drive folder ID")
        search_term = None
        folder_search = None
        
    else:  # Search folders
        folder_search = st.text_input("🔍 Search for folders", placeholder="Enter folder name keywords...")
        search_term = None
        folder_id = None
        
        if folder_search and st.button("🔍 Find Folders", use_container_width=True):
            folders = list_drive_folders(service, search_term=folder_search)
            
            if folders:
                st.success(f"Found {len(folders)} matching folders:")
                
                # Create a selectbox for folders
                folder_options = {f"{folder['name']}": folder['id'] for folder in folders}
                selected_folder_name = st.selectbox(
                    "Select a folder to load documents from:",
                    options=list(folder_options.keys())
                )
                
                if selected_folder_name:
                    folder_id = folder_options[selected_folder_name]
                    st.info(f"Selected folder ID: `{folder_id}`")
            else:
                st.warning("No folders found matching your search")
    
    if st.button("🔄 Load Documents", use_container_width=True):
        with st.spinner("Loading documents from Google Drive..."):
            # Handle URL-based single file
            if hasattr(st.session_state, 'url_file_id') and st.session_state.url_file_id:
                try:
                    file_info = service.files().get(
                        fileId=st.session_state.url_file_id,
                        fields="id, name, mimeType, shortcutDetails"
                    ).execute()
                    
                    files = [file_info]
                    st.info(f"Loading file: {file_info['name']}")
                    del st.session_state.url_file_id  # Clear after use
                except Exception as e:
                    st.error(f"Error loading file from URL: {e}")
                    files = []
            else:
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
        
        # Debug: Show available models
        with st.expander("🔍 Debug: Check Available Models"):
            try:
                available_models = []
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        available_models.append(m.name)
                
                st.write("Available models for generateContent:")
                for model_name in available_models:
                    st.code(model_name)
                
                if available_models:
                    st.info(f"Try using one of these model names: {available_models[0]}")
            except Exception as e:
                st.error(f"Could not list models: {e}")
        
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
                    
                    # Create context from documents - use full content
                    context = "\n\n".join([doc.page_content for doc in relevant_docs[:3]])
                    
                    # Use native Gemini API directly - try multiple model names
                    with st.spinner("Thinking..."):
                        # Use the models that actually exist for your API key
                        model_names = [
                            'models/gemini-2.0-flash',
                            'models/gemini-2.5-flash', 
                            'models/gemini-flash-latest'
                        ]
                        
                        answer = None
                        last_error = None
                        
                        for model_name in model_names:
                            try:
                                model = genai.GenerativeModel(model_name)
                                prompt = f"""Based on the following document content, please answer this question: {user_question}

Document content:
{context}

Please provide a helpful and accurate answer based only on the information provided."""
                                
                                response = model.generate_content(prompt)
                                answer = response.text
                                break  # Success, exit loop
                            except Exception as model_error:
                                last_error = model_error
                                continue
                        
                        if answer:
                            st.write(answer)
                            
                            # Show sources with intelligent excerpt extraction
                            with st.expander("📚 Sources", expanded=True):
                                for idx, doc in enumerate(relevant_docs[:3], 1):
                                    st.subheader(f"Source {idx}: {doc.metadata.get('source', 'Unknown')}")
                                    
                                    content = doc.page_content
                                    
                                    # Find multiple relevant excerpts using both query AND answer
                                    excerpts = find_relevant_excerpts(
                                        content, 
                                        user_question, 
                                        answer=answer,  # Pass the AI's answer to find matching excerpts
                                        num_excerpts=2, 
                                        excerpt_length=600
                                    )
                                    
                                    for excerpt_idx, (excerpt, relevance_score) in enumerate(excerpts, 1):
                                        if len(excerpts) > 1:
                                            st.markdown(f"**Relevant section {excerpt_idx}** (relevance score: {relevance_score})")
                                        else:
                                            st.markdown(f"**Most relevant section** (relevance score: {relevance_score})")
                                        
                                        # Highlight search terms in the excerpt
                                        highlighted_excerpt = excerpt
                                        
                                        st.text_area(
                                            f"Excerpt {excerpt_idx}",
                                            value=highlighted_excerpt,
                                            height=150,
                                            disabled=True,
                                            key=f"source_{idx}_excerpt_{excerpt_idx}_{doc.metadata.get('file_id', 'unknown')}",
                                            label_visibility="collapsed"
                                        )
                                    
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        st.caption(f"📄 Full document: {len(content):,} characters")
                                    with col2:
                                        # Option to show full document
                                        if st.button(f"View full doc", key=f"view_full_{idx}_{doc.metadata.get('file_id', 'unknown')}", use_container_width=True):
                                            st.session_state[f'show_full_{idx}'] = not st.session_state.get(f'show_full_{idx}', False)
                                    
                                    if st.session_state.get(f'show_full_{idx}', False):
                                        st.text_area(
                                            "Full document content",
                                            value=content,
                                            height=300,
                                            disabled=True,
                                            key=f"full_content_{idx}_{doc.metadata.get('file_id', 'unknown')}"
                                        )
                            
                            st.session_state.chat_history.append((user_question, answer))
                        else:
                            raise last_error if last_error else Exception("All model attempts failed")
                
                except Exception as e:
                    st.error(f"⚠️ Error: {str(e)[:200]}")
                    
                    # Fallback to showing document snippets
                    st.info("📝 Showing relevant document content instead:")
                    try:
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