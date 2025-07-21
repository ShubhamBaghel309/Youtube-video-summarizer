import chromadb
from chromadb.utils import embedding_functions
import hashlib

class SummaryDatabase:
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name="video_summaries",
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )

    def generate_id(self, video_id, summary_type):
        """Generate unique ID for video summary"""
        return hashlib.md5(f"{video_id}_{summary_type}".encode()).hexdigest()

    def store_summary(self, video_id, summary_type, summary, video_url):
        """Store video summary in database"""
        unique_id = self.generate_id(video_id, summary_type)
        
        try:
            self.collection.add(
                documents=[summary],
                ids=[unique_id],
                metadatas=[{
                    "video_id": video_id,
                    "summary_type": summary_type,
                    "video_url": video_url
                }]
            )
            return True
        except Exception as e:
            print(f"Error storing summary: {e}")
            return False

    def get_summary(self, video_id, summary_type):
        """Retrieve summary from database if it exists"""
        unique_id = self.generate_id(video_id, summary_type)
        
        try:
            results = self.collection.get(
                ids=[unique_id],
                include=['documents', 'metadatas']
            )
            
            if results and results['documents']:
                return results['documents'][0]
            return None
        except Exception as e:
            print(f"Error retrieving summary: {e}")
            return None

    def clear_summary(self, video_id=None, summary_type=None):
        """Clear specific or all summaries from database"""
        try:
            if video_id and summary_type:
                unique_id = self.generate_id(video_id, summary_type)
                self.collection.delete(ids=[unique_id])
            elif video_id:
                # Delete all summaries for this video
                results = self.collection.get(
                    where={"video_id": video_id},
                    include=["metadatas"]
                )
                if results and results['ids']:
                    self.collection.delete(ids=results['ids'])
            else:
                # Delete all summaries
                self.collection = self.client.get_or_create_collection(
                    name="video_summaries",
                    embedding_function=embedding_functions.DefaultEmbeddingFunction()
                )
            return True
        except Exception as e:
            print(f"Error clearing summaries: {e}")
            return False
