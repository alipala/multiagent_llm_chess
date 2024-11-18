from typing import List, Dict, Optional
import logging
import os
from chromadb import PersistentClient
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import json

logger = logging.getLogger(__name__)

class ChessKnowledgeBase:
    """Manages chess knowledge storage and retrieval using ChromaDB"""
    
    def __init__(self, persist_directory: str = "chess_knowledge_db"):
        self.persist_directory = persist_directory
        self.collection_name = "chess_knowledge"
        
        # Initialize ChromaDB client
        self.client = PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Use OpenAI embeddings
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-ada-002"
        )

        # Initialize or get collection
        self._initialize_collection()
        logger.info(f"Initialized ChessKnowledgeBase with persistence at {persist_directory}")

    def _initialize_collection(self):
        """Initialize or get the knowledge collection"""
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "Chess knowledge and game analysis"}
            )
            logger.info("Collection initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing collection: {str(e)}")
            raise

    def add_knowledge(self, documents: List[str], metadata: List[Dict] = None) -> bool:
        """Add new knowledge to the database"""
        try:
            # Generate IDs for documents
            ids = [f"doc_{i}_{hash(doc)}" for i, doc in enumerate(documents)]
            
            # Add documents to collection
            self.collection.add(
                documents=documents,
                metadatas=metadata if metadata else [{}] * len(documents),
                ids=ids
            )
            logger.info(f"Added {len(documents)} documents to knowledge base")
            return True
        except Exception as e:
            logger.error(f"Error adding knowledge: {str(e)}")
            return False

    def query_knowledge(self, query: str, n_results: int = 5) -> List[Dict]:
        """Query the knowledge base"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return [
                {
                    "document": doc,
                    "metadata": meta,
                    "distance": dist
                }
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )
            ]
        except Exception as e:
            logger.error(f"Error querying knowledge base: {str(e)}")
            return []

    def export_knowledge(self, output_file: str = "chess_knowledge.json"):
        """Export knowledge base to JSON file"""
        try:
            all_data = self.collection.get()
            export_data = {
                "documents": all_data["documents"],
                "metadatas": all_data["metadatas"],
                "ids": all_data["ids"]
            }
            
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            logger.info(f"Exported knowledge base to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error exporting knowledge base: {str(e)}")
            return False

    def get_statistics(self) -> Dict:
        """Get statistics about the knowledge base"""
        try:
            all_data = self.collection.get()
            return {
                "total_documents": len(all_data["documents"]),
                "unique_metadata_keys": list(set(
                    key for meta in all_data["metadatas"]
                    for key in meta.keys()
                )),
                "storage_size": os.path.getsize(self.persist_directory) if os.path.exists(self.persist_directory) else 0
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {}