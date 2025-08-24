"""
Vector-based Tool Knowledge System using FAISS
Provides semantic search and storage for tool usage knowledge
"""

import os
import json
import time
import pickle
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

@dataclass
class ToolKnowledgeEntry:
    """A single piece of tool knowledge"""
    id: str
    user_id: str
    tool_name: str
    search_query: str
    result_summary: str
    importance_score: float
    timestamp: float
    usage_count: int = 1
    last_used: float = None
    
    def __post_init__(self):
        if self.last_used is None:
            self.last_used = self.timestamp

class VectorToolKnowledge:
    """Vector-based tool knowledge storage using FAISS"""
    
    def __init__(self, storage_dir: str = "vector_tool_knowledge"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        # Initialize sentence transformer for embeddings
        try:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            print("[VECTOR TOOL KNOWLEDGE] Sentence transformer loaded successfully")
        except Exception as e:
            print(f"[VECTOR TOOL KNOWLEDGE ERROR] Failed to load sentence transformer: {e}")
            self.embedder = None
        
        # FAISS index for vector search
        self.index = None
        self.entries: List[ToolKnowledgeEntry] = []
        self.entry_map: Dict[str, int] = {}  # id -> index in entries list
        
        # Load existing data
        self.load_data()
        
        # Initialize FAISS index
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index"""
        if not self.entries:
            return
            
        try:
            # Get embedding dimension from first entry
            sample_text = self.entries[0].search_query
            sample_embedding = self.embedder.encode([sample_text])
            dimension = sample_embedding.shape[1]
            
            # Create FAISS index
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Add all existing embeddings
            if self.entries:
                embeddings = self._get_embeddings([entry.search_query for entry in self.entries])
                self.index.add(embeddings.astype('float32'))
            
            print(f"[VECTOR TOOL KNOWLEDGE] FAISS index initialized with {len(self.entries)} entries")
            
        except Exception as e:
            print(f"[VECTOR TOOL KNOWLEDGE ERROR] Failed to initialize FAISS index: {e}")
            self.index = None
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts"""
        if not self.embedder:
            # Fallback: return random embeddings (for testing)
            return np.random.rand(len(texts), 384)
        
        try:
            embeddings = self.embedder.encode(texts)
            return embeddings
        except Exception as e:
            print(f"[VECTOR TOOL KNOWLEDGE ERROR] Failed to get embeddings: {e}")
            # Fallback: return random embeddings
            return np.random.rand(len(texts), 384)
    
    def add_tool_knowledge(self, user_id: str, tool_name: str, search_query: str,
                          result_summary: str, importance_score: float) -> str:
        """Add tool knowledge with vector storage"""
        if not self.embedder:
            print("[VECTOR TOOL KNOWLEDGE WARNING] No embedder available, falling back to simple storage")
            return self._add_simple_storage(user_id, tool_name, search_query, result_summary, importance_score)
        
        # Check if similar knowledge already exists
        similar_entries = self.search_tool_knowledge(search_query, limit=3, threshold=0.8)
        
        if similar_entries:
            # Update existing entry if it's very similar
            best_match = similar_entries[0]
            if best_match['similarity'] > 0.9:
                # Update existing entry
                entry_id = best_match['entry'].id
                existing_entry = self.entries[self.entry_map[entry_id]]
                existing_entry.usage_count += 1
                existing_entry.last_used = time.time()
                existing_entry.importance_score = max(existing_entry.importance_score, importance_score)
                existing_entry.result_summary = result_summary  # Update with latest result
                
                print(f"[VECTOR TOOL KNOWLEDGE] Updated existing entry for {tool_name}")
                self.save_data()
                return entry_id
        
        # Create new entry
        entry_id = f"tool_{int(time.time() * 1000)}_{hash(search_query) % 10000}"
        entry = ToolKnowledgeEntry(
            id=entry_id,
            user_id=user_id,
            tool_name=tool_name,
            search_query=search_query,
            result_summary=result_summary,
            importance_score=importance_score,
            timestamp=time.time()
        )
        
        # Add to storage
        self.entries.append(entry)
        self.entry_map[entry_id] = len(self.entries) - 1
        
        # Add to FAISS index
        if self.index is not None:
            try:
                embedding = self._get_embeddings([search_query])
                self.index.add(embedding.astype('float32'))
            except Exception as e:
                print(f"[VECTOR TOOL KNOWLEDGE ERROR] Failed to add to FAISS index: {e}")
        
        # Cleanup old entries if too many
        if len(self.entries) > 1000:
            self._cleanup_old_entries()
        
        print(f"[VECTOR TOOL KNOWLEDGE] Added new tool knowledge for {tool_name}")
        self.save_data()
        return entry_id
    
    def _add_simple_storage(self, user_id: str, tool_name: str, search_query: str,
                           result_summary: str, importance_score: float) -> str:
        """Fallback simple storage when vector system is unavailable"""
        entry_id = f"simple_{int(time.time() * 1000)}_{hash(search_query) % 10000}"
        entry = ToolKnowledgeEntry(
            id=entry_id,
            user_id=user_id,
            tool_name=tool_name,
            search_query=search_query,
            result_summary=result_summary,
            importance_score=importance_score,
            timestamp=time.time()
        )
        
        self.entries.append(entry)
        self.entry_map[entry_id] = len(self.entries) - 1
        return entry_id
    
    def search_tool_knowledge(self, query: str, limit: int = 5, threshold: float = 0.5,
                            user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search tool knowledge using vector similarity"""
        if not self.entries:
            return []
        
        if not self.index or not self.embedder:
            # Fallback to simple text search
            return self._simple_search(query, limit, user_id)
        
        try:
            # Get query embedding
            query_embedding = self._get_embeddings([query])
            
            # Search FAISS index
            similarities, indices = self.index.search(query_embedding.astype('float32'), min(limit * 2, len(self.entries)))
            
            # Process results
            results = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx < 0 or idx >= len(self.entries):
                    continue
                    
                entry = self.entries[idx]
                
                # Filter by user if specified
                if user_id and entry.user_id != user_id:
                    continue
                
                # Apply threshold
                if similarity < threshold:
                    continue
                
                results.append({
                    'entry': entry,
                    'similarity': float(similarity),
                    'relevance_score': float(similarity * entry.importance_score)
                })
            
            # Sort by relevance and return top results
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            return results[:limit]
            
        except Exception as e:
            print(f"[VECTOR TOOL KNOWLEDGE ERROR] Vector search failed: {e}")
            return self._simple_search(query, limit, user_id)
    
    def _simple_search(self, query: str, limit: int, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Simple text-based search fallback"""
        query_words = set(query.lower().split())
        scored_entries = []
        
        for entry in self.entries:
            if user_id and entry.user_id != user_id:
                continue
                
            # Score based on word overlap
            entry_words = set(entry.search_query.lower().split())
            word_overlap = len(query_words & entry_words)
            
            if word_overlap > 0:
                # Simple scoring: word overlap * importance * recency
                recency_factor = 1.0 / (1.0 + (time.time() - entry.last_used) / 86400)  # Decay over days
                score = word_overlap * entry.importance_score * recency_factor
                
                scored_entries.append({
                    'entry': entry,
                    'similarity': min(1.0, word_overlap / len(query_words)),
                    'relevance_score': score
                })
        
        # Sort by relevance and return top results
        scored_entries.sort(key=lambda x: x['relevance_score'], reverse=True)
        return scored_entries[:limit]
    
    def get_tool_knowledge_by_tool(self, tool_name: str, limit: int = 10) -> List[ToolKnowledgeEntry]:
        """Get all knowledge for a specific tool"""
        tool_entries = [entry for entry in self.entries if entry.tool_name == tool_name]
        tool_entries.sort(key=lambda x: x.importance_score, reverse=True)
        return tool_entries[:limit]
    
    def get_user_tool_knowledge(self, user_id: str, limit: int = 20) -> List[ToolKnowledgeEntry]:
        """Get all tool knowledge for a specific user"""
        user_entries = [entry for entry in self.entries if entry.user_id == user_id]
        user_entries.sort(key=lambda x: x.importance_score, reverse=True)
        return user_entries[:limit]
    
    def _cleanup_old_entries(self):
        """Remove old, low-importance entries"""
        if len(self.entries) <= 500:
            return
        
        # Sort by importance and recency
        scored_entries = []
        current_time = time.time()
        
        for entry in self.entries:
            # Score: importance * recency_factor
            days_since_used = (current_time - entry.last_used) / 86400
            recency_factor = 1.0 / (1.0 + days_since_used)
            score = entry.importance_score * recency_factor
            
            scored_entries.append((score, entry))
        
        # Keep top 500 entries
        scored_entries.sort(key=lambda x: x[0], reverse=True)
        keep_entries = scored_entries[:500]
        
        # Rebuild storage
        self.entries = [entry for _, entry in keep_entries]
        self.entry_map = {entry.id: idx for idx, entry in enumerate(self.entries)}
        
        # Rebuild FAISS index
        if self.embedder and self.entries:
            try:
                dimension = self._get_embeddings([self.entries[0].search_query]).shape[1]
                self.index = faiss.IndexFlatIP(dimension)
                embeddings = self._get_embeddings([entry.search_query for entry in self.entries])
                self.index.add(embeddings.astype('float32'))
                print(f"[VECTOR TOOL KNOWLEDGE] Cleaned up and rebuilt index with {len(self.entries)} entries")
            except Exception as e:
                print(f"[VECTOR TOOL KNOWLEDGE ERROR] Failed to rebuild index after cleanup: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        total_entries = len(self.entries)
        unique_tools = len(set(entry.tool_name for entry in self.entries))
        unique_users = len(set(entry.user_id for entry in self.entries))
        
        # Calculate average importance and usage
        if total_entries > 0:
            avg_importance = sum(entry.importance_score for entry in self.entries) / total_entries
            avg_usage = sum(entry.usage_count for entry in self.entries) / total_entries
        else:
            avg_importance = 0.0
            avg_usage = 0.0
        
        return {
            "total_entries": total_entries,
            "unique_tools": unique_tools,
            "unique_users": unique_users,
            "average_importance": round(avg_importance, 3),
            "average_usage_count": round(avg_usage, 1),
            "vector_index_active": self.index is not None,
            "embedder_available": self.embedder is not None,
            "storage_directory": self.storage_dir
        }
    
    def save_data(self):
        """Save data to disk"""
        try:
            # Save entries
            entries_file = os.path.join(self.storage_dir, "tool_knowledge_entries.json")
            entries_data = [asdict(entry) for entry in self.entries]
            
            with open(entries_file, 'w') as f:
                json.dump(entries_data, f, indent=2)
            
            # Save FAISS index if available
            if self.index is not None:
                index_file = os.path.join(self.storage_dir, "faiss_index.bin")
                faiss.write_index(self.index, index_file)
            
            print(f"[VECTOR TOOL KNOWLEDGE] Saved {len(self.entries)} entries to disk")
            
        except Exception as e:
            print(f"[VECTOR TOOL KNOWLEDGE ERROR] Failed to save data: {e}")
    
    def load_data(self):
        """Load data from disk"""
        try:
            # Load entries
            entries_file = os.path.join(self.storage_dir, "tool_knowledge_entries.json")
            if os.path.exists(entries_file):
                with open(entries_file, 'r') as f:
                    entries_data = json.load(f)
                
                self.entries = [ToolKnowledgeEntry(**data) for data in entries_data]
                self.entry_map = {entry.id: idx for idx, entry in enumerate(self.entries)}
                
                print(f"[VECTOR TOOL KNOWLEDGE] Loaded {len(self.entries)} entries from disk")
            else:
                print("[VECTOR TOOL KNOWLEDGE] No existing data found, starting fresh")
                
        except Exception as e:
            print(f"[VECTOR TOOL KNOWLEDGE ERROR] Failed to load data: {e}")
            self.entries = []
            self.entry_map = {}
    
    def get_persistent_state(self) -> Dict[str, Any]:
        """Get persistent state for saving to disk"""
        return {
            'storage_directory': self.storage_dir,
            'total_entries': len(self.entries),
            'last_save': time.time()
        }
    
    def load_persistent_state(self, state: Dict[str, Any]) -> None:
        """Load persistent state from disk"""
        try:
            if 'storage_directory' in state:
                self.storage_dir = state['storage_directory']
                os.makedirs(self.storage_dir, exist_ok=True)
            
            # Load actual data
            self.load_data()
            
            # Reinitialize index
            self._initialize_index()
            
            print(f"[VECTOR TOOL KNOWLEDGE] Loaded persistent state")
            
        except Exception as e:
            print(f"[VECTOR TOOL KNOWLEDGE ERROR] Failed to load persistent state: {e}")

# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("[TESTING] Vector Tool Knowledge System")
    
    # Initialize system
    vtk = VectorToolKnowledge()
    
    # Test adding knowledge
    print("\n[TEST] Adding tool knowledge...")
    vtk.add_tool_knowledge(
        user_id="test_user",
        tool_name="search",
        search_query="What is Python programming language?",
        result_summary="Python is a high-level programming language known for simplicity and readability.",
        importance_score=0.8
    )
    
    vtk.add_tool_knowledge(
        user_id="test_user",
        tool_name="search",
        search_query="How to install Python packages?",
        result_summary="Use pip install package_name to install Python packages from PyPI.",
        importance_score=0.7
    )
    
    # Test search
    print("\n[TEST] Searching for Python...")
    results = vtk.search_tool_knowledge("Python programming", limit=3)
    for i, result in enumerate(results):
        print(f"  {i+1}. {result['entry'].tool_name}: {result['entry'].search_query[:50]}... (similarity: {result['similarity']:.3f})")
    
    # Test stats
    print("\n[TEST] System stats:")
    stats = vtk.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n[TEST] Vector Tool Knowledge System test completed!")
