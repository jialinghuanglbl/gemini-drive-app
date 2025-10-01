# Google Drive AI Chat

A Streamlit application that allows you to chat with your Google Drive documents using AI. Upload, search, and analyze your Google Docs, Sheets, and PDFs with natural language queries.

## Features

- 🔐 Secure Google Drive OAuth integration
- 📂 Support for Google Docs, Sheets, PDFs, and shortcuts
- 🔍 Search documents by keywords or browse folders
- 💬 Natural language chat interface with your documents
- 🛡️ Fallback text search when AI quotas are exceeded
- 📄 Document preview and content viewer

## Setup for Deployment

### 1. Google Cloud Setup (Service Account)

**Create a Service Account:**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Navigate to "IAM & Admin" → "Service Accounts"
4. Click "Create Service Account"
5. Give it a name (e.g., "drive-chat-bot")
6. Grant it "Viewer" role (or no role needed)
7. Click "Create Key" → JSON format
8. Download the JSON file

**Enable APIs:**
1. Go to "APIs & Services" → "Library"
2. Enable "Google Drive API"
3. Enable "Generative Language API"

**Share Google Drive Files:**
1. Open the downloaded JSON file
2. Copy the `client_email` value (looks like `xxx@xxx.iam.gserviceaccount.com`)
3. In Google Drive, share your folders/files with this email address
4. Give it "Viewer" or "Commenter" permission

### 2. Google AI Studio Setup

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Create a new API key for the Gemini API
3. Copy the API key for later use

### 3. Streamlit Cloud Deployment

1. Fork this repository to your GitHub account
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub account and deploy this repository
4. Add secrets in Streamlit Cloud settings:

```toml
# Add the entire contents of your service account JSON file
[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-private-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\nYour-private-key-here\n-----END PRIVATE KEY-----\n"
client_email = "your-service-account@your-project.iam.gserviceaccount.com"
client_id = "your-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "your-cert-url"
```

### 4. Local Development

1. Clone the repository:
```bash
git clone <your-repo-url>
cd google-drive-ai-chat
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.streamlit/secrets.toml`:
```toml
GOOGLE_CLIENT_ID = "your-google-oauth-client-id"
GOOGLE_CLIENT_SECRET = "your-google-oauth-client-secret" 
REDIRECT_URI = "http://localhost:8501"
```

4. Run the app:
```bash
streamlit run app.py
```

## Usage

1. **Connect to Google Drive**: Click the "Connect to Google Drive" button and authorize access
2. **Load Documents**: Search for documents by keywords or specify a folder ID
3. **View Documents**: Preview loaded documents in the expandable section
4. **Chat**: Ask questions about your documents in natural language
5. **Fallback Mode**: If AI quotas are exceeded, the app shows relevant document snippets

## Supported File Types

- Google Docs (`.gdoc`)
- Google Sheets (`.gsheet`)
- PDF files (`.pdf`)
- Word documents (`.docx`)
- Plain text (`.txt`)
- Google Drive shortcuts

## API Quotas

The app uses Google's Gemini API which has free tier limits:
- Limited requests per day/minute
- Limited tokens per request

When quotas are exceeded, the app automatically falls back to simple text search and shows relevant document content directly.

## Security

- OAuth2 authentication for secure Google Drive access
- No document content is stored permanently
- API keys are managed through Streamlit secrets
- Read-only access to Google Drive

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
1. Check the troubleshooting section below
2. Open an issue on GitHub
3. Review Google's API documentation

## Troubleshooting

**OAuth Issues**: Ensure redirect URIs match exactly between Google Cloud Console and your deployment URL

**API Quota Exceeded**: Wait 24 hours for quotas to reset, or upgrade to paid tier

**Document Processing Errors**: Check that files are accessible and not corrupted

**Deployment Issues**: Verify all secrets are properly configured in Streamlit Cloud