#Inngest-Powered RAG API

A high-performance, asynchronous Retrieval-Augmented Generation (RAG) system. This project processes PDF documents, chunks them for semantic search, and provides accurate, context-aware answers using OpenAI’s LLM, all managed by an event-driven background job system.



## Tech Stack
* **Backend:** FastAPI (Python 3.14)
* **Message Broker:** Inngest (for asynchronous PDF processing)
* **Vector Database:** Qdrant
* **Frontend:** Streamlit
* **Embeddings:** OpenAI text-embedding-3-large (3072 dimensions)
* **LLM:** GPT-4o-mini

## Architecture Overview
The system utilizes an event-driven pattern to ensure the user never hangs while waiting for PDF processing:
1. Ingestion: Files uploaded via Streamlit trigger an Inngest event. The background job parses the PDF, chunks the text, embeds it, and upserts vectors into Qdrant.
2. Retrieval: User queries are embedded in real-time, searching the Qdrant collection for semantic similarity.
3. Generation: The retrieved context is injected into a system prompt for the LLM to generate an answer.

## Getting Started

### 1. Database Setup
Ensure you have Docker installed, then start your local Qdrant instance:

```bash
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 2. Environment Variables
Create a .env file in the root directory and add your key:
```bash
OPENAI_API_KEY=sk-your-key-here
```

### 3. Installation
You can use uv (recommended) or standard pip:
# Clone the repository
git clone <your-repo-url>
cd rag-api-project

# Install dependencies
uv sync
# OR if not using uv:
pip install -r requirements.txt

### 4. Running the Application
Open three terminal windows to run the stack:

Terminal 1 (Inngest):
npx inngest-cli@latest dev

Terminal 2 (Backend API):
uvicorn main:app --reload

Terminal 3 (Frontend):
streamlit run streamlit_app.py

## Lessons Learned (The "Gotchas")
* Async/Await Traps: Learned to avoid asyncio.run() in Streamlit due to event loop closures; switched to managing clients without @st.cache_resource when they require persistent connections to active event loops.
* API Constraints: Encountered max_tokens_per_request errors with OpenAI embeddings; implemented batching to ensure data is processed in manageable chunks rather than a single massive payload.
* Silent Data Drops: Debugged silent failures in Qdrant retrieval, discovering the importance of checking response.points explicitly after the v1.0+ API updates and verifying collection vector dimensions (3072 for text-embedding-3-large).
* Resilient Parsing: Replaced brittle default PDF readers with PyMuPDF to handle complex web-exported PDFs, including defensive checks for empty content.

## License
MIT