import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from langchain_community.document_loaders import TextLoader, MarkdownLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
EMBEDDING_MODEL_NAME = os.getenv("LOCAL_EMBEDDING_MODEL_NAME")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 400

def upload_document_to_qdrant(file_path: str, document_type: str, collection_name: str, embeddings, loader):
    """
    Uploads a document to Qdrant with specified metadata.

    Args:
        file_path: Path to the document file.
        document_type: The type of document (e.g., "Regulation", "FAQ").
        collection_name: The name of the Qdrant collection.
        embeddings: The Langchain embedding model to use.
        loader: The Langchain DocumentLoader to use for the file type.
    """
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    documents = loader.load(file_path=file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(documents)

    payload = []
    vector_data = []

    for chunk in chunks:
        vector = embeddings.embed_query(chunk.page_content)
        vector_data.append(vector)
        payload.append({"content": chunk.page_content, "metadata": {"document_type": document_type}})

    client.upsert(
        collection_name=collection_name,
        points=models.Batch(
            ids=list(range(len(chunks))),  # Generate simple IDs for demonstration
            vectors=vector_data,
            payloads=payload
        ),
        wait=True
    )
    print(f"Uploaded {len(chunks)} chunks from {file_path} to collection '{collection_name}' as '{document_type}'")

if __name__ == "__main__":
    # Ensure you have the embedding model initialized
    embeddings_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    collection_name = os.getenv("QDRANT_COLLECTION_NAME")

    data_dir = "data"  # Create a 'data' folder and put your files in subfolders
    regulations_dir = os.path.join(data_dir, "regulations")
    faq_dir = os.path.join(data_dir, "faq")
    related_issues_dir = os.path.join(data_dir, "related_issues")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(regulations_dir):
        os.makedirs(regulations_dir)
    if not os.path.exists(faq_dir):
        os.makedirs(faq_dir)
    if not os.path.exists(related_issues_dir):
        os.makedirs(related_issues_dir)

    # Process Regulations (assuming TXT files)
    if os.path.exists(regulations_dir):
        for filename in os.listdir(regulations_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(regulations_dir, filename)
                upload_document_to_qdrant(file_path, "Regulation", collection_name, embeddings_model, TextLoader())
            elif filename.endswith(".md"):
                file_path = os.path.join(regulations_dir, filename)
                upload_document_to_qdrant(file_path, "Regulation", collection_name, embeddings_model, MarkdownLoader())
            elif filename.endswith(".pdf"):
                file_path = os.path.join(regulations_dir, filename)
                upload_document_to_qdrant(file_path, "Regulation", collection_name, embeddings_model, PyPDFLoader())
    else:
        print(f"Directory not found: {regulations_dir}")

    # Process FAQs (assuming MD files, but can include others)
    if os.path.exists(faq_dir):
        for filename in os.listdir(faq_dir):
            if filename.endswith(".md"):
                file_path = os.path.join(faq_dir, filename)
                upload_document_to_qdrant(file_path, "FAQ", collection_name, embeddings_model, MarkdownLoader())
            elif filename.endswith(".txt"):
                file_path = os.path.join(faq_dir, filename)
                upload_document_to_qdrant(file_path, "FAQ", collection_name, embeddings_model, TextLoader())
            elif filename.endswith(".pdf"):
                file_path = os.path.join(faq_dir, filename)
                upload_document_to_qdrant(file_path, "FAQ", collection_name, embeddings_model, PyPDFLoader())
    else:
        print(f"Directory not found: {faq_dir}")

    # Process Related Issues (assuming MD files, but can include others)
    if os.path.exists(related_issues_dir):
        for filename in os.listdir(related_issues_dir):
            if filename.endswith(".md"):
                file_path = os.path.join(related_issues_dir, filename)
                upload_document_to_qdrant(file_path, "RelatedIssue", collection_name, embeddings_model, MarkdownLoader())
            elif filename.endswith(".txt"):
                file_path = os.path.join(related_issues_dir, filename)
                upload_document_to_qdrant(file_path, "RelatedIssue", collection_name, embeddings_model, TextLoader())
            elif filename.endswith(".pdf"):
                file_path = os.path.join(related_issues_dir, filename)
                upload_document_to_qdrant(file_path, "RelatedIssue", collection_name, embeddings_model, PyPDFLoader())
    else:
        print(f"Directory not found: {related_issues_dir}")

    print("Data upload process finished.")