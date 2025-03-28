import json
import os
import shutil
from typing import List
import uuid
from dotenv import load_dotenv

from langchain.schema import Document

# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings

# from langchain.vectorstores import Qdrant
from langchain_community.vectorstores import Qdrant
import qdrant_client

from qdrant_client.models import Distance, VectorParams

load_dotenv()

class CorpusIndexer:
    """Manages the indexing of the document corpus in Qdrant via LangChain."""
    
    def __init__(self, openai_api_key=None):
        """
        Initializes the indexer with OpenAI embeddings.
        
        Args:
            openai_api_key: OpenAI API key (optional, otherwise uses the OPENAI_API_KEY environment variable)
        """
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_api_key
        )
        
        # Initialize Qdrant client
        self.qdrant_client = None
        self.vectorstore = None
        self.corpus_documents = []
    
    def load_corpus(self, corpus_path: str) -> List[Document]:
        """
        Loads the training corpus from a JSON file.
        
        Args:
            corpus_path: Path to the JSON file containing the training corpus
        
        Returns:
            List of LangChain documents
        """
        # Load corpus data
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)
        
        print(f"Corpus loaded: {len(corpus_data)} documents found")
        
        # Convert to LangChain Documents
        documents = []
        for i, doc_data in enumerate(corpus_data):
            # Extract main information
            content = doc_data.get("content", "")
            doc_type = doc_data.get("type", "")
            source = doc_data.get("source", "")
            
            # Create document with metadata
            doc = Document(
                page_content=content,
                metadata={
                    "id": f"doc_{i+1}",
                    "type": doc_type,
                    "source": source
                }
            )
            documents.append(doc)
        
        # Save documents for future reference
        self.corpus_documents = documents
        
        return documents
    
    def create_index(self, 
                documents: List[Document], 
                collection_name: str = "formations") -> Qdrant:
        """
        Creates a vector index in Qdrant in memory mode.
        
        Args:
            documents: List of documents to index
            collection_name: Name of the Qdrant collection
            
        Returns:
            LangChain Qdrant instance
        """
        # Initialize Qdrant client in memory mode (no disk storage)
        print(f"Creating a Qdrant client in memory mode...")
        self.qdrant_client = qdrant_client.QdrantClient(location=":memory:")
        
        print(f"Creating index in Qdrant (collection: {collection_name})...")
        
        # Create vectorstore with Qdrant - using location parameter instead of client
        self.vectorstore = Qdrant.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=collection_name,
            location=":memory:"  # Use location parameter instead of client
        )
        
        print(f"Index created successfully! {len(documents)} documents indexed.")
        
        return self.vectorstore
    
    def save_index(self, output_path: str = "qdrant_export.json"):
        """
        Saves the state of the index for later use.
        This method is necessary because the in-memory index will disappear at the end of the process.
        
        Args:
            output_path: Path where to save the export
        """
        if not self.corpus_documents or not self.vectorstore:
            print("No index to save.")
            return
        
        # We cannot directly save the in-memory index,
        # but we can save the documents and metadata
        export_data = {
            "documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in self.corpus_documents
            ],
            "collection_name": self.vectorstore.collection_name
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"Index data exported to: {output_path}")
        
    def close(self):
        """Properly closes the Qdrant client"""
        if self.qdrant_client:
            try:
                # In memory mode, no need for complex explicit closure
                self.qdrant_client = None
                print("Qdrant client released.")
            except Exception as e:
                print(f"Error while releasing the Qdrant client: {e}")


if __name__ == "__main__":
    # Path to corpus file
    corpus_path = "data/formation.json"
    
    try:
        # Create the indexer
        indexer = CorpusIndexer(openai_api_key=os.getenv("OPENAI_API_KEY"))
        
        # Load the corpus
        documents = indexer.load_corpus(corpus_path)
        
        # Create the in-memory index
        vectorstore = indexer.create_index(
            documents=documents,
            collection_name="formations"
        )
        
        # Save index information for later reference
        indexer.save_index("data/qdrant_export.json")
        
        print("Indexing completed!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Release resources
        if 'indexer' in locals():
            indexer.close()