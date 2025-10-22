import streamlit as st
import json
from io import BytesIO
from typing import List, Optional
import requests

from google.oauth2 import service_account
from googleapiclient.discovery import build

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class RecursiveCharacterTextSplitter:
    """Minimal replacement for langchain text splitter used in this app.

    Provides split_documents(documents) -> List[Document], splitting page_content
    into chunks with overlap.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []
        chunks = []
        step = self.chunk_size - self.chunk_overlap if self.chunk_size > self.chunk_overlap else self.chunk_size
        for i in range(0, len(text), step):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        out = []
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for idx, chunk in enumerate(chunks):
                meta = dict(doc.metadata) if getattr(doc, 'metadata', None) else {}
                # annotate chunk index to metadata
                meta.update({
                    'chunk_index': idx,
                    'source': meta.get('source', None)
                })
                out.append(Document(page_content=chunk, metadata=meta))
        return out

# Page configuration
st.set_page_config(
    page_title="Google Drive AI Chat - CBORG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

class SimpleTextRetriever(BaseRetriever):
    """Simple text-based retriever"""
    
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

def query_cborg(prompt: str, api_key: str, model: str = "llama-3.1-70b-instruct") -> str:
    """Query CBORG API with a prompt"""
    api_url = "https://api.cborg.lbl.gov/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    import streamlit as st
    try:
        # st.info(f"CBORG API payload: {payload}")
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        # st.info(f"CBORG API response status: {response.status_code}")
        # if response.status_code != 200:
        #     st.error(f"CBORG API error {response.status_code}: {response.text}")
        response.raise_for_status()
        result = response.json()
        # st.info(f"CBORG API raw response: {result}")
        return result['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        # st.error(f"CBORG API exception: {str(e)}")
        raise Exception(f"CBORG API error: {str(e)}")

def get_drive_service():
    """Create Google Drive service using service account credentials"""
    try:
        service_account_info = st.secrets["gcp_service_account"]
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=SCOPES
        )
        return build('drive', 'v3', credentials=credentials)
    except Exception as e:
        st.error(f"Error creating Drive service: {e}")
        return None

def find_relevant_excerpts(content: str, query: str, answer: str = "", num_excerpts: int = 3, excerpt_length: int = 400) -> List[tuple[str, int]]:
    """Find the most relevant excerpts from content based on query and answer"""
    content_lower = content.lower()
    query_words = [w.strip('?.,!;:') for w in query.lower().split() if len(w) > 3]
    
    answer_words = []
    if answer:
        answer_words = [w.strip('?.,!;:') for w in answer.lower().split() if len(w) > 4]
        common_words = {'this', 'that', 'these', 'those', 'there', 'where', 'when', 'what', 
                       'which', 'while', 'with', 'about', 'would', 'could', 'should', 'their',
                       'document', 'section', 'states', 'mentions', 'based', 'provided'}
        answer_words = [w for w in answer_words if w not in common_words][:10]
    
    all_search_words = query_words + answer_words
    
    if not all_search_words:
        return [(content[:excerpt_length] + "..." if len(content) > excerpt_length else content, 0)]
    
    chunk_size = excerpt_length
    chunks = []
    for i in range(0, len(content), chunk_size // 3):
        chunk_text = content[i:i + chunk_size]
        if chunk_text.strip():
            chunks.append((chunk_text, i))
    
    scored_chunks = []
    for chunk_text, start_pos in chunks:
        chunk_lower = chunk_text.lower()
        score = 0
        
        for word in query_words:
            count = chunk_lower.count(word)
            score += count * 3
        
        for word in answer_words:
            count = chunk_lower.count(word)
            score += count * 2
        
        has_query_term = any(word in chunk_lower for word in query_words)
        has_answer_term = any(word in chunk_lower for word in answer_words)
        if has_query_term and has_answer_term:
            score += 10
        
        unique_query_words = sum(1 for word in query_words if word in chunk_lower)
        unique_answer_words = sum(1 for word in answer_words if word in chunk_lower)
        score += unique_query_words * 4
        score += unique_answer_words * 2
        
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
            positions.sort()
            gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
            avg_gap = sum(gaps) / len(gaps) if gaps else chunk_size
            if avg_gap < chunk_size / 3:
                score += 8
        
        if len(query.split()) >= 2:
            words = query.lower().split()
            for i in range(len(words) - 1):
                phrase = ' '.join(words[i:i+2])
                if phrase in chunk_lower:
                    score += 15
        
        if score > 0:
            scored_chunks.append((chunk_text, start_pos, score))
    
    scored_chunks.sort(key=lambda x: x[2], reverse=True)
    
    if not scored_chunks:
        return [(content[:excerpt_length] + "..." if len(content) > excerpt_length else content, 0)]
    
    results = []
    seen_positions = set()
    
    for chunk_text, start_pos, score in scored_chunks[:num_excerpts * 2]:
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
    """Extract Google Drive file/folder ID from URL"""
    import re
    
    folder_patterns = [
        r'/folders/([a-zA-Z0-9_-]+)',
        r'[?&]id=([a-zA-Z0-9_-]+)'
    ]
    
    file_patterns = [
        r'/d/([a-zA-Z0-9_-]+)',
        r'/file/d/([a-zA-Z0-9_-]+)',
        r'[?&]id=([a-zA-Z0-9_-]+)'
    ]
    
    if '/folders/' in url or 'folder' in url.lower():
        for pattern in folder_patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1), 'folder'
    
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

def list_folder_contents(service, folder_id: str = 'root'):
    """List both folders and files in a given folder"""
    try:
        # Get folders
        folder_query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        folder_results = service.files().list(
            q=folder_query,
            fields="files(id, name, mimeType)",
            orderBy="name",
            pageSize=100
        ).execute()
        folders = folder_results.get('files', [])
        
        # Get files
        supported_types = [
            'application/vnd.google-apps.document',
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/vnd.google-apps.spreadsheet',
            'text/plain',
            'application/msword',
            'application/vnd.google-apps.shortcut'
        ]
        
        file_query = f"'{folder_id}' in parents and trashed=false and (" + " or ".join([f"mimeType='{mime}'" for mime in supported_types]) + ")"
        file_results = service.files().list(
            q=file_query,
            fields="files(id, name, mimeType, shortcutDetails)",
            orderBy="name",
            pageSize=100
        ).execute()
        files = file_results.get('files', [])
        
        return folders, files
    except Exception as e:
        st.error(f"Error listing contents: {e}")
        return [], []

def get_file_icon(mime_type: str) -> str:
    """Return emoji icon based on file type"""
    icon_map = {
        'application/vnd.google-apps.folder': '📁',
        'application/vnd.google-apps.document': '📄',
        'application/vnd.google-apps.spreadsheet': '📊',
        'application/pdf': '📕',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '📝',
        'text/plain': '📃',
        'application/msword': '📝',
        'application/vnd.google-apps.shortcut': '🔗'
    }
    return icon_map.get(mime_type, '📄')

def interactive_browser(service, mode: str = 'files'):
    """Interactive file/folder browser with visual selection"""
    
    # Initialize browser state
    if 'browser_path' not in st.session_state:
        st.session_state.browser_path = []
        st.session_state.browser_current_folder = 'root'
        st.session_state.browser_current_name = 'My Drive'
        st.session_state.selected_items = []
        st.session_state.selected_folder = None
        st.session_state.open_folder_browser = False
    
    st.subheader(f"📂 Current Location: {' > '.join([st.session_state.browser_current_name] if not st.session_state.browser_path else [st.session_state.browser_current_name] + st.session_state.browser_path)}")
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("⬆️ Go Up", disabled=st.session_state.browser_current_folder == 'root'):
            if st.session_state.browser_path:
                st.session_state.browser_path.pop()
            # Note: Going up would require tracking parent IDs, simplified here
            st.session_state.browser_current_folder = 'root'
            st.session_state.browser_current_name = 'My Drive'
            st.rerun()
    
    with col2:
        if st.button("🏠 Home"):
            st.session_state.browser_path = []
            st.session_state.browser_current_folder = 'root'
            st.session_state.browser_current_name = 'My Drive'
            st.session_state.selected_items = []
            st.rerun()
    
    # Get current folder contents
    folders, files = list_folder_contents(service, st.session_state.browser_current_folder)
    
    st.write(f"**Folders:** {len(folders)} | **Files:** {len(files)}")
    
    # Display folders first
    if folders:
        st.markdown("### 📁 Folders")
        
        # Create grid layout for folders
        cols_per_row = 3
        for i in range(0, len(folders), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, folder in enumerate(folders[i:i+cols_per_row]):
                with cols[j]:
                    if st.button(
                        f"📁\n{folder['name'][:30]}{'...' if len(folder['name']) > 30 else ''}",
                        key=f"folder_{folder['id']}",
                        use_container_width=True
                    ):
                        # Navigate into folder
                        st.session_state.browser_path.append(folder['name'])
                        st.session_state.browser_current_folder = folder['id']
                        st.session_state.browser_current_name = folder['name']
                        st.session_state.selected_items = []
                        st.rerun()
    
    # Display files
    if files:
        st.markdown("### 📄 Files")
        
        # File selection
        st.info("Select files to load (click to toggle selection):")
        
        cols_per_row = 3
        for i in range(0, len(files), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, file in enumerate(files[i:i+cols_per_row]):
                with cols[j]:
                    icon = get_file_icon(file['mimeType'])
                    file_id = file['id']
                    is_selected = file_id in st.session_state.selected_items
                    
                    # Create button with visual selection indicator
                    button_label = f"{'✅ ' if is_selected else ''}{icon}\n{file['name'][:30]}{'...' if len(file['name']) > 30 else ''}"
                    
                    if st.button(
                        button_label,
                        key=f"file_{file_id}",
                        use_container_width=True,
                        type="primary" if is_selected else "secondary"
                    ):
                        # Toggle selection
                        if file_id in st.session_state.selected_items:
                            st.session_state.selected_items.remove(file_id)
                        else:
                            st.session_state.selected_items.append(file_id)
                        st.rerun()
    
    # Selection summary and load button
    if st.session_state.selected_items:
        st.success(f"✅ {len(st.session_state.selected_items)} file(s) selected")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if mode == 'files':
                if st.button("🔄 Load Selected Files", use_container_width=True, type="primary"):
                    return st.session_state.selected_items
            else:
                if st.button("📁 Select this folder", use_container_width=True, type="primary"):
                    # Set the selected folder in session state and exit
                    st.session_state.selected_folder = st.session_state.browser_current_folder
                    st.session_state.open_folder_browser = False
                    st.rerun()
        with col2:
            if st.button("❌ Clear Selection", use_container_width=True):
                st.session_state.selected_items = []
                st.rerun()
    else:
        if mode == 'files':
            st.info("👆 Click on files above to select them for loading")
        else:
            st.info("Navigate to the folder you want and click 'Select this folder'.")
    
    return None

    # (Removed duplicate module-level load_option/elif block; main() contains the proper UI flow)
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
        return None

def create_vector_store(documents: List[Document], cborg_api_key: str):
    """Create simple text store"""
    if not documents:
        return None
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    
    st.info(f"Using simple text search for {len(documents)} documents")
    return create_simple_text_store(documents)

def main():
    def get_cborg_models(api_key):
        api_url = "https://api.cborg.lbl.gov/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            response = requests.get(api_url, headers=headers, timeout=30)
            response.raise_for_status()
            models = response.json().get("data", [])
            return [m["id"] for m in models]
        except Exception as e:
            st.warning(f"Could not fetch CBORG models: {e}")
            return []
    st.title("📚 Google Drive AI Chat - CBORG")
    st.markdown("Chat with your Google Drive documents using CBORG models")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Get CBORG API key from secrets
        if "CBORG_API_KEY" not in st.secrets:
            st.error("❌ CBORG API key not found in secrets")
            st.info("Please add CBORG_API_KEY to your Streamlit secrets")
            st.stop()
        
        cborg_api_key = st.secrets["CBORG_API_KEY"]
        st.success("✅ CBORG API key loaded")
        st.session_state.cborg_api_key = cborg_api_key
        
        # Model selection (fetch from API)
        st.subheader("Model Selection")
        available_models = get_cborg_models(cborg_api_key)
        if available_models:
            model_choice = st.selectbox(
                "Choose CBORG model:",
                available_models,
                index=0,
                help="Select which CBORG model to use"
            )
            st.session_state.cborg_model = model_choice
        else:
            st.warning("No valid CBORG models found for your API key.")
        
        st.divider()
        
        # Service account check
        if "gcp_service_account" not in st.secrets:
            st.error("❌ Service account not configured")
            st.info("""
            **Setup Instructions:**
            
            1. Create a service account in Google Cloud Console
            2. Download the JSON key file
            3. Share your Google Drive files with the service account email
            4. Add the JSON content to Streamlit secrets as `gcp_service_account`
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
    
    service = get_drive_service()
    if not service:
        st.error("Failed to connect to Google Drive")
        st.stop()
    
    # Document loading interface
    st.header("📂 Load Documents")
    
    load_option = st.segmented_control(
        "Choose loading method:",
        options=["🔗 Paste URL", "📄 Search files", "🗂️ Browse files", "📁 Browse by folder", "🔍 Search folders"]
    )
    
    if load_option == "🔗 Paste URL":
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
            
    elif load_option == "📄 Search files":
        search_term = st.text_input("📄 Search for files", placeholder="Enter keywords to find files...")
        folder_id = None
        folder_search = None
    
        
    elif load_option == "📁 Browse by folder":
        # Allow opening the interactive folder browser
        if st.button("🗂️ Open folder browser", use_container_width=True):
            st.session_state.open_folder_browser = True
            st.session_state.browser_current_folder = 'root'
            st.session_state.browser_path = []
            st.session_state.browser_current_name = 'My Drive'
            st.session_state.selected_items = []
            st.rerun()

        # If the folder browser is open, show it (mode folder)
        if st.session_state.get('open_folder_browser'):
            selected_folder = interactive_browser(service, mode='folder')
            # If a folder was selected, set folder_id
            if st.session_state.get('selected_folder'):
                folder_id = st.session_state.selected_folder
                st.info(f"Selected folder ID: `{folder_id}`")
            else:
                folder_id = None
        else:
            folder_id = st.text_input("📁 Folder ID", placeholder="Paste Google Drive folder ID")
        search_term = None
        folder_search = None
        
    elif load_option == "🔍 Search folders":
        folder_search = st.text_input("🔍 Search for folders", placeholder="Enter folder name keywords...")
        search_term = None
        folder_id = None
        
        if folder_search and st.button("🔍 Find Folders", use_container_width=True):
            folders = list_drive_folders(service, search_term=folder_search)
            
            if folders:
                st.success(f"Found {len(folders)} matching folders:")
                
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
    
    elif load_option == "🗂️ Browse files":
        st.info("Navigate through your Google Drive and select files to load")
        
        selected_file_ids = interactive_browser(service)
        
        if selected_file_ids:
            with st.spinner("Loading selected files..."):
                try:
                    files = []
                    for file_id in selected_file_ids:
                        file_info = service.files().get(
                            fileId=file_id,
                            fields="id, name, mimeType, shortcutDetails"
                        ).execute()
                        files.append(file_info)
                    
                    if files:
                        st.info(f"Processing {len(files)} selected files...")
                        documents = []
                        progress_bar = st.progress(0)
                        
                        for i, file_info in enumerate(files):
                            doc = process_file(service, file_info)
                            if doc:
                                documents.append(doc)
                            progress_bar.progress((i + 1) / len(files))
                        
                        if documents:
                            st.session_state.documents = documents
                            st.session_state.vector_store = create_vector_store(documents, cborg_api_key)
                            st.success(f"✅ Loaded {len(documents)} documents successfully!")
                            # Clear selection after loading
                            st.session_state.selected_items = []
                            st.session_state.browser_path = []
                            st.session_state.browser_current_folder = 'root'
                        else:
                            st.error("No documents could be processed")
                except Exception as e:
                    st.error(f"Error loading files: {e}")

    if st.button("🔄 Load Documents", use_container_width=True):
        with st.spinner("Loading documents from Google Drive..."):
            if hasattr(st.session_state, 'url_file_id') and st.session_state.url_file_id:
                try:
                    file_info = service.files().get(
                        fileId=st.session_state.url_file_id,
                        fields="id, name, mimeType, shortcutDetails"
                    ).execute()
                    
                    files = [file_info]
                    st.info(f"Loading file: {file_info['name']}")
                    del st.session_state.url_file_id
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
                    st.session_state.vector_store = create_vector_store(documents, cborg_api_key)
                    
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
    if st.session_state.documents and st.session_state.vector_store:
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
                    if hasattr(st.session_state.vector_store, 'as_retriever'):
                        retriever = st.session_state.vector_store.as_retriever()
                        relevant_docs = retriever._get_relevant_documents(user_question)
                    else:
                        relevant_docs = st.session_state.documents
                    
                    context = "\n\n".join([doc.page_content for doc in relevant_docs[:3]])
                    
                    with st.spinner("Thinking..."):
                        prompt = f"""Based on the following document content, please answer this question: {user_question}

Document content:
{context}

Please provide a helpful and accurate answer based only on the information provided."""
                        
                        answer = query_cborg(
                            prompt, 
                            st.session_state.cborg_api_key,
                            st.session_state.cborg_model
                        )
                        
                        if answer:
                            st.write(answer)
                            
                            with st.expander("📚 Sources", expanded=True):
                                for idx, doc in enumerate(relevant_docs[:3], 1):
                                    st.subheader(f"Source {idx}: {doc.metadata.get('source', 'Unknown')}")
                                    
                                    content = doc.page_content
                                    
                                    excerpts = find_relevant_excerpts(
                                        content, 
                                        user_question, 
                                        answer=answer,
                                        num_excerpts=2, 
                                        excerpt_length=600
                                    )
                                    
                                    for excerpt_idx, (excerpt, relevance_score) in enumerate(excerpts, 1):
                                        if len(excerpts) > 1:
                                            st.markdown(f"**Relevant section {excerpt_idx}** (relevance score: {relevance_score})")
                                        else:
                                            st.markdown(f"**Most relevant section** (relevance score: {relevance_score})")
                                        
                                        st.text_area(
                                            f"Excerpt {excerpt_idx}",
                                            value=excerpt,
                                            height=150,
                                            disabled=True,
                                            key=f"source_{idx}_excerpt_{excerpt_idx}_{doc.metadata.get('file_id', 'unknown')}",
                                            label_visibility="collapsed"
                                        )
                                    
                                    st.caption(f"📄 Full document: {len(content):,} characters")
                                    
                                    with st.expander(f"📖 View full document"):
                                        st.text_area(
                                            "Full document content",
                                            value=content,
                                            height=400,
                                            disabled=True,
                                            key=f"full_content_{idx}_{doc.metadata.get('file_id', 'unknown')}"
                                        )
                                    
                                    if idx < len(relevant_docs[:3]):
                                        st.divider()
                            
                            st.session_state.chat_history.append((user_question, answer))
                        else:
                            st.error("Failed to get response from CBORG")
                
                except Exception as e:
                    st.error(f"⚠️ Error: {str(e)[:200]}")
                    st.info("📝 Showing relevant document content instead:")
                    
                    try:
                        if hasattr(st.session_state.vector_store, 'as_retriever'):
                            retriever = st.session_state.vector_store.as_retriever()
                            relevant_docs = retriever._get_relevant_documents(user_question)
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
                                st.text_area("Document preview", preview, height=100, key=f"preview_err_{doc.metadata.get('file_id', 'unknown')}")
                    except Exception as fallback_error:
                        st.error(f"Fallback also failed: {fallback_error}")
    else:
        return None

if __name__ == "__main__":
    main()