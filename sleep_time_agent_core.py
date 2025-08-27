#!/usr/bin/env python3
"""
Sleep Time Agent Core (Unified + Toggleable Insights)
- Memory tools: append / replace / archival / finish
- One-pass friendly prompts (no yapping)
- Structured insight "tools" (parse-only, toggleable): intent, mood, safety, facts, privacy, conflicts, evidence, next_action
- Structured summary tool (parse-only): sleeptime_summarize (gated by enable_schema_tools)
- Unified runner: run_unified(...) → does insights + guarded memory writes + summary in one streamed pass
- Backward-compatible with your tests; process_conversation() path unchanged
"""

import asyncio
import json
import time
import logging
import hashlib
import numpy as np
from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING
from dataclasses import dataclass, asdict, field
from pathlib import Path

# Try to import FAISS, fallback to simple vector system if not available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available, using simple vector memory fallback")

# --- runtime import (for actual execution) ---
try:
    from ollama import Client as _OllamaRuntimeClient  # type: ignore
except Exception:
    _OllamaRuntimeClient = None  # no runtime client available

# --- type-only import (for Pylance/Pyright) ---
if TYPE_CHECKING:
    from ollama import Client as OllamaClient
else:
    OllamaClient = Any  # fallback so annotations stay valid at runtime

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class AgentConfig:
    # Triggers
    trigger_after_messages: int = 5
    trigger_after_idle_minutes: int = 30

    # Model/runtime
    model: str = "llama3.2:3b"
    thinking_iterations: int = 1  # one pass by default (no yapping)
    temperature: float = 0.2
    num_ctx: int = 8192
    num_gpu: int = 1
    num_thread: int = 8
    top_k: int = 40
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    repeat_last_n: int = 64
    tfs_z: float = 1.0
    typical_p: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    mirostat: int = 0
    mirostat_eta: float = 0.1
    mirostat_tau: float = 5.0
    num_predict: int = -1
    stop: List[str] = field(default_factory=list)
    numa: bool = False
    num_keep: int = 0
    num_batch: int = 1024
    penalize_newline: bool = True
    vocab_only: bool = False
    seed: int = -1

    # Thinking/streaming
    stream_thinking: bool = True
    enable_tool_calling: bool = True
    enable_thinking: bool = True

    # Memory
    max_memory_blocks: int = 20
    memory_persistence: bool = True
    memory_file: str = "sleep_agent_memory.json"

    # Retries
    max_retries: int = 3
    skip_malformed_messages: bool = True
    min_message_length: int = 2

    # Structured JSON packs
    enable_schema_tools: bool = True       # rank/person/sentiment + summarizer
    enable_insights: bool = True           # intent/mood/safety/facts/privacy/…

    # FAISS settings
    enable_faiss: bool = True
    vector_dimension: int = 384  # Default embedding dimension
    max_vectors_per_user: int = 1000
    similarity_threshold: float = 0.7

# -----------------------------------------------------------------------------
# Memory blocks
# -----------------------------------------------------------------------------
@dataclass
class MemoryBlock:
    label: str
    value: str
    description: str
    user_id: str
    max_chars: int = 1000
    last_updated: float = 0.0
    version: str = "1.0"

class MemoryManager:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.blocks: Dict[str, Dict[str, MemoryBlock]] = {}
        self.memory_file = Path(config.memory_file)
        self._load_memory()

    def _load_memory(self):
        if self.memory_file.exists() and self.config.memory_persistence:
            try:
                data = json.load(self.memory_file.open("r", encoding="utf-8"))
                for block_data in data:
                    block = MemoryBlock(**block_data)
                    self.blocks.setdefault(block.user_id, {})[block.label] = block
                logger.info(f"Loaded memory for {len(self.blocks)} users")
            except Exception as e:
                logger.error(f"Error loading memory: {e}")

    def _save_memory(self):
        if not self.config.memory_persistence:
            return
        try:
            all_blocks = []
            for user_blocks in self.blocks.values():
                for block in user_blocks.values():
                    all_blocks.append(asdict(block))
            json.dump(all_blocks, self.memory_file.open("w", encoding="utf-8"), indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving memory: {e}")

    def get_user_blocks(self, user_id: str) -> Dict[str, MemoryBlock]:
        return self.blocks.get(user_id, {}).copy()

    def get_block(self, user_id: str, label: str) -> Optional[MemoryBlock]:
        return self.blocks.get(user_id, {}).get(label)

    def update_block(self, user_id: str, label: str, value: str, description: Optional[str] = None):
        self.blocks.setdefault(user_id, {})
        if label in self.blocks[user_id]:
            blk = self.blocks[user_id][label]
            blk.value = value[:blk.max_chars]
            if description:
                blk.description = description
            blk.last_updated = time.time()
        else:
            blk = MemoryBlock(
                label=label,
                value=value[:1000],
                description=description or f"Auto-created block: {label}",
                user_id=user_id,
                last_updated=time.time()
            )
            self.blocks[user_id][label] = blk
        self._save_memory()
        logger.info(f"[memory] Updated '{label}' for user '{user_id}'")

    def create_block(self, user_id: str, label: str, value: str, description: str, max_chars: int = 1000) -> bool:
        self.blocks.setdefault(user_id, {})
        if label in self.blocks[user_id]:
            logger.warning(f"[memory] Block '{label}' already exists for '{user_id}'")
            return False
        if len(self.blocks[user_id]) >= self.config.max_memory_blocks:
            logger.warning(f"[memory] Limit reached for user '{user_id}'")
            return False
        blk = MemoryBlock(
            label=label, value=value[:max_chars], description=description,
            user_id=user_id, max_chars=max_chars, last_updated=time.time()
        )
        self.blocks[user_id][label] = blk
        self._save_memory()
        logger.info(f"[memory] Created '{label}' for user '{user_id}'")
        return True

# -----------------------------------------------------------------------------
# FAISS Memory System
# -----------------------------------------------------------------------------

class FAISSMemoryManager:
    """FAISS-based vector memory for semantic similarity search"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.vector_dimension = config.vector_dimension
        self.max_vectors_per_user = config.max_vectors_per_user
        self.similarity_threshold = config.similarity_threshold
        
        # User-specific FAISS indexes
        self.user_indexes: Dict[str, Any] = {}  # user_id -> faiss index
        self.user_vectors: Dict[str, List[np.ndarray]] = {}  # user_id -> list of vectors
        self.user_texts: Dict[str, List[str]] = {}  # user_id -> list of text content
        self.user_metadata: Dict[str, List[Dict]] = {}  # user_id -> list of metadata
        
        # Simple hash-based vector fallback if FAISS unavailable
        self.fallback_vectors: Dict[str, Dict[str, List[float]]] = {}
        
        logger.info(f"FAISS Memory Manager initialized (FAISS available: {FAISS_AVAILABLE})")
    
    def _create_simple_vector(self, text: str) -> List[float]:
        """Create a simple hash-based vector when FAISS is not available"""
        # Create a deterministic vector based on text hash
        text_hash = hashlib.md5(text.encode()).hexdigest()
        vector = []
        
        # Convert hash to vector of specified dimension
        for i in range(self.vector_dimension):
            # Use different parts of hash for different vector positions
            start = (i * 8) % len(text_hash)
            end = start + 8
            hex_val = text_hash[start:end]
            # Convert hex to float between -1 and 1
            val = (int(hex_val, 16) / (16**8)) * 2 - 1
            vector.append(val)
        
        return vector
    
    def _create_faiss_index(self, user_id: str):
        """Create a FAISS index for a user"""
        if not FAISS_AVAILABLE:
            return
        
        try:
            # Create a simple L2 distance index
            index = faiss.IndexFlatL2(self.vector_dimension)
            self.user_indexes[user_id] = index
            self.user_vectors[user_id] = []
            self.user_texts[user_id] = []
            self.user_metadata[user_id] = []
            logger.info(f"Created FAISS index for user {user_id}")
        except Exception as e:
            logger.error(f"Error creating FAISS index for user {user_id}: {e}")
    
    def add_memory(self, user_id: str, text: str, metadata: Dict = None) -> bool:
        """Add a memory vector for a user"""
        try:
            if user_id not in self.user_indexes:
                self._create_faiss_index(user_id)
            
            if FAISS_AVAILABLE:
                # Create embedding vector (placeholder - would use actual embedding model)
                vector = np.random.rand(self.vector_dimension).astype('float32')
                
                # Add to FAISS index
                self.user_indexes[user_id].add(vector.reshape(1, -1))
                self.user_vectors[user_id].append(vector)
                self.user_texts[user_id].append(text)
                self.user_metadata[user_id].append(metadata or {})
                
                # Check if we need to remove old vectors
                if len(self.user_vectors[user_id]) > self.max_vectors_per_user:
                    self._remove_oldest_vector(user_id)
                
                logger.debug(f"Added FAISS memory for user {user_id}")
            else:
                # Fallback to simple vector
                vector = self._create_simple_vector(text)
                if user_id not in self.fallback_vectors:
                    self.fallback_vectors[user_id] = {}
                
                # Store in fallback system
                memory_id = f"mem_{len(self.fallback_vectors[user_id])}"
                self.fallback_vectors[user_id][memory_id] = {
                    'vector': vector,
                    'text': text,
                    'metadata': metadata or {},
                    'timestamp': time.time()
                }
                
                logger.debug(f"Added fallback memory for user {user_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding FAISS memory for user {user_id}: {e}")
            return False
    
    def _remove_oldest_vector(self, user_id: str):
        """Remove the oldest vector when limit is reached"""
        if user_id in self.user_vectors and len(self.user_vectors[user_id]) > 0:
            # Remove from all storage
            self.user_vectors[user_id].pop(0)
            self.user_texts[user_id].pop(0)
            self.user_metadata[user_id].pop(0)
            
            # Rebuild FAISS index
            if FAISS_AVAILABLE and self.user_vectors[user_id]:
                vectors = np.array(self.user_vectors[user_id])
                self.user_indexes[user_id].reset()
                self.user_indexes[user_id].add(vectors)
    
    def search_similar(self, user_id: str, query_text: str, top_k: int = 5) -> List[Dict]:
        """Search for similar memories"""
        try:
            if user_id not in self.user_indexes and user_id not in self.fallback_vectors:
                return []
            
            if FAISS_AVAILABLE and user_id in self.user_indexes:
                # Create query vector
                query_vector = np.random.rand(self.vector_dimension).astype('float32')
                
                # Search FAISS index
                distances, indices = self.user_indexes[user_id].search(
                    query_vector.reshape(1, -1), top_k
                )
                
                results = []
                for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                    if idx < len(self.user_texts[user_id]):
                        similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
                        if similarity >= self.similarity_threshold:
                            results.append({
                                'text': self.user_texts[user_id][idx],
                                'metadata': self.user_metadata[user_id][idx],
                                'similarity': similarity,
                                'rank': i + 1
                            })
                
                return results
            else:
                # Fallback search using simple vectors
                if user_id in self.fallback_vectors:
                    query_vector = self._create_simple_vector(query_text)
                    results = []
                    
                    for memory_id, memory_data in self.fallback_vectors[user_id].items():
                        similarity = self._calculate_similarity(query_vector, memory_data['vector'])
                        if similarity >= self.similarity_threshold:
                            results.append({
                                'text': memory_data['text'],
                                'metadata': memory_data['metadata'],
                                'similarity': similarity,
                                'rank': len(results) + 1
                            })
                    
                    # Sort by similarity
                    results.sort(key=lambda x: x['similarity'], reverse=True)
                    return results[:top_k]
            
            return []
            
        except Exception as e:
            logger.error(f"Error searching FAISS memory for user {user_id}: {e}")
            return []
    
    def _calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        except Exception:
            return 0.0
    
    def get_user_stats(self, user_id: str) -> Dict:
        """Get statistics for a user's vector memory"""
        if FAISS_AVAILABLE and user_id in self.user_indexes:
            return {
                'total_vectors': len(self.user_vectors.get(user_id, [])),
                'index_type': 'FAISS',
                'dimension': self.vector_dimension
            }
        elif user_id in self.fallback_vectors:
            return {
                'total_vectors': len(self.fallback_vectors[user_id]),
                'index_type': 'Fallback',
                'dimension': self.vector_dimension
            }
        else:
            return {
                'total_vectors': 0,
                'index_type': 'None',
                'dimension': self.vector_dimension
            }

# -----------------------------------------------------------------------------
# Core Agent
# -----------------------------------------------------------------------------
class SleepTimeAgentCore:
    def __init__(self, config: AgentConfig, client: Optional[OllamaClient] = None, memory_manager: Optional[MemoryManager] = None):
        if _OllamaRuntimeClient is None and client is None:
            raise RuntimeError("ollama.Client is required. Install with `pip install ollama` or pass a client.")
        self.config = config
        self.client = client or _OllamaRuntimeClient(host='http://localhost:11434')
        self.memory_manager = memory_manager or MemoryManager(config)
        
        # Initialize FAISS memory manager
        self.faiss_memory = FAISSMemoryManager(config) if config.enable_faiss else None
        
        self.last_activity: Dict[str, float] = {}
        self.message_counts: Dict[str, int] = {}

        # Pre-create default blocks (includes 'human' to avoid replace-loop retries)
        self._default_blocks = {
            "human": "Stable personal identity/bio facts (name, pronouns, high-level role)",
            "persona": "User personality and behavior patterns",
            "conversation_context": "Current conversation context and important details",
            "user_preferences": "User preferences and interaction history",
            "behavioral_patterns": "Observed behavioral patterns and tendencies",
        }

        self._setup_memory_tools()
        logger.info(f"SleepTimeAgentCore -> model: {config.model} | streaming: {config.stream_thinking} | thinking: {config.enable_thinking}")
        logger.info(f"FAISS memory: {'enabled' if self.faiss_memory else 'disabled'}")

    # --------------------- executors: memory ---------------------
    def core_memory_append(self, name: str, content: str) -> str:
        uid = getattr(self, "_current_user_id", "unknown")
        try:
            existing = self.memory_manager.get_block(uid, name)
            old = existing.value if existing else ""
            merged = (old + ("\n" if old and content else "") + content).strip()
            self.memory_manager.update_block(uid, name, merged)
            return f"Appended to {name}"
        except Exception as e:
            return f"Failed append {name}: {e}"

    def core_memory_replace(self, name: str, old_content: str, new_content: str) -> str:
        uid = getattr(self, "_current_user_id", "unknown")
        blk = self.memory_manager.get_block(uid, name)
        if not blk:
            # Fallback prevents retry loops: create block and set new content
            self.memory_manager.update_block(uid, name, new_content)
            return f"Created '{name}' and set new content"
        try:
            self.memory_manager.update_block(uid, name, blk.value.replace(old_content, new_content))
            return f"Replaced content in {name}"
        except Exception as e:
            return f"Failed replace {name}: {e}"

    def archival_memory_insert(self, content: str) -> str:
        uid = getattr(self, "_current_user_id", "unknown")
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            self.memory_manager.create_block(uid, f"archival_{int(time.time())}", f"[{ts}] {content}", "Archival memory entry")
            return "Inserted archival entry"
        except Exception as e:
            return f"Failed archival insert: {e}"

    def insight_memory_append(self, insight_type: str, content: str) -> str:
        """Store insights in misc memory block
        
        Args:
            insight_type (str): Type of insight (e.g., "mood", "intent", "facts")
            content (str): Insight content to store
            
        Returns:
            str: Confirmation message
        """
        user_id = getattr(self, '_current_user_id', 'unknown')
        username = getattr(self, '_current_username', 'unknown')
        
        try:
            # Format as structured entry
            timestamp = time.strftime("%H:%M:%S")
            insight_entry = f"[{timestamp}] {insight_type}: {content}"
            
            # Get existing misc block or create empty
            existing_block = self.memory_manager.get_block(user_id, "misc")
            existing_content = existing_block.value if existing_block else ""
            
            # Append new insight
            new_content = f"{existing_content}\n{insight_entry}".strip()
            self.memory_manager.update_block(user_id, "misc", new_content, "Structured insights and analysis")
            
            logger.info(f"[memory] Stored {insight_type} insight for {username}")
            return f"Stored {insight_type} insight in misc memory"
        except Exception as e:
            logger.error(f"Error storing {insight_type} insight: {e}")
            return f"Failed to store {insight_type} insight: {str(e)}"

    def insight_memory_replace(self, insight_type: str, old_content: str, new_content: str) -> str:
        """Update specific insight in misc memory block
        
        Args:
            insight_type (str): Type of insight to update
            old_content (str): Content to replace
            new_content (str): New content to replace with
            
        Returns:
            str: Confirmation message
        """
        user_id = getattr(self, '_current_user_id', 'unknown')
        username = getattr(self, '_current_username', 'unknown')
        
        try:
            existing_block = self.memory_manager.get_block(user_id, "misc")
            if not existing_block:
                return f"Misc memory block not found"
            
            # Replace content in misc block
            updated_content = existing_block.value.replace(old_content, new_content)
            self.memory_manager.update_block(user_id, "misc", updated_content, "Structured insights and analysis")
            
            logger.info(f"[memory] Updated {insight_type} insight for {username}")
            return f"Updated {insight_type} insight in misc memory"
        except Exception as e:
            logger.error(f"Error updating {insight_type} insight: {e}")
            return f"Failed to update {insight_type} insight: {str(e)}"

    def finish_memory_edits(self) -> str:
        uname = getattr(self, "_current_username", "unknown")
        logger.info(f"[memory] Edits finished for {uname}")
        return f"Memory editing completed for {uname}"
    
    def faiss_memory_store(self, content: str, metadata: Dict = None) -> str:
        """Store content in FAISS vector memory"""
        if not self.faiss_memory:
            return "FAISS memory not enabled"
        
        try:
            user_id = getattr(self, "_current_user_id", "unknown")
            success = self.faiss_memory.add_memory(user_id, content, metadata)
            if success:
                return f"Stored content in FAISS memory for user {user_id}"
            else:
                return "Failed to store in FAISS memory"
        except Exception as e:
            logger.error(f"Error storing in FAISS memory: {e}")
            return f"FAISS storage error: {str(e)}"
    
    def faiss_memory_search(self, query: str, top_k: int = 5) -> str:
        """Search FAISS vector memory for similar content"""
        if not self.faiss_memory:
            return "FAISS memory not enabled"
        
        try:
            user_id = getattr(self, "_current_user_id", "unknown")
            results = self.faiss_memory.search_similar(user_id, query, top_k)
            
            if not results:
                return "No similar memories found"
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append(
                    f"Rank {result['rank']} (similarity: {result['similarity']:.3f}): {result['text'][:100]}..."
                )
            
            return f"Found {len(results)} similar memories:\n" + "\n".join(formatted_results)
        except Exception as e:
            logger.error(f"Error searching FAISS memory: {e}")
            return f"FAISS search error: {str(e)}"

    # --------------------- tool schemas: memory ---------------------
    def _tool_schemas_memory(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "core_memory_append",
                    "description": "Append content to a core memory block",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "content": {"type": "string"}
                        },
                        "required": ["name", "content"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "core_memory_replace",
                    "description": "Replace content in a memory block",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "old_content": {"type": "string"},
                            "new_content": {"type": "string"}
                        },
                        "required": ["name", "old_content", "new_content"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "archival_memory_insert",
                    "description": "Insert content into archival memory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"}
                        },
                        "required": ["content"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "insight_memory_append",
                    "description": "Store insights in misc memory block",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "insight_type": {"type": "string", "description": "Type of insight (mood, intent, facts, etc.)"},
                            "content": {"type": "string", "description": "Insight content to store"}
                        },
                        "required": ["insight_type", "content"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "insight_memory_replace",
                    "description": "Update specific insight in misc memory block",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "insight_type": {"type": "string", "description": "Type of insight to update"},
                            "old_content": {"type": "string", "description": "Content to replace"},
                            "new_content": {"type": "string", "description": "New content to replace with"}
                        },
                        "required": ["insight_type", "old_content", "new_content"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "faiss_memory_store",
                    "description": "Store content in FAISS vector memory for semantic search",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "description": "Content to store in vector memory"},
                            "metadata": {"type": "object", "description": "Optional metadata for the content"}
                        },
                        "required": ["content"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "faiss_memory_search",
                    "description": "Search FAISS vector memory for similar content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Query text to search for"},
                            "top_k": {"type": "integer", "description": "Number of results to return (default: 5)"}
                        },
                        "required": ["query"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "finish_memory_edits",
                    "description": "Signal completion of memory edits",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False}
                }
            }
        ]

    # --------------------- schema tools (structured JSON demo) ---------------------
    def _tool_schemas_structured(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "generate_rank",
                    "description": "Structured ranking of a food item",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "rank": {"type": "integer", "minimum": 1, "maximum": 10},
                            "reason": {"type": "string"},
                            "categories": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "score": {"type": "integer", "minimum": 1, "maximum": 10}
                                    },
                                    "required": ["name", "score"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["rank", "reason", "categories"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "summarize_person",
                    "description": "Structured person summary",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "occupation": {"type": "string"},
                            "hobbies": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["name", "occupation"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "classify_sentiment",
                    "description": "Structured sentiment classification",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
                            "score": {"type": "integer", "minimum": 1, "maximum": 10}
                        },
                        "required": ["sentiment", "score"],
                        "additionalProperties": False
                    }
                }
            }
        ]

    # --------------------- insights pack (parse-only, toggleable) ---------------------
    def _tool_schemas_insights(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "classify_intent",
                    "description": "User intent classification",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "primary": {"type": "string", "enum": ["ask_info","task_request","share_update","vent","chitchat","feedback"]},
                            "secondary": {"type": "array", "items": {"type": "string"}},
                            "needs_clarification": {"type": "boolean"},
                            "clarification_questions": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["primary","needs_clarification"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "infer_mood",
                    "description": "Infer user mood/affect",
                    "parameters": {
                        "type":"object",
                        "properties":{
                            "valence":{"type":"string","enum":["positive","neutral","negative"]},
                            "arousal":{"type":"integer","minimum":1,"maximum":10},
                            "labels":{"type":"array","items":{"type":"string"}}
                        },
                        "required":["valence","arousal"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "safety_scan",
                    "description": "Safety risk assessment",
                    "parameters": {
                        "type":"object",
                        "properties":{
                            "risk":{"type":"string","enum":["none","low","medium","high"]},
                            "categories":{"type":"array","items":{"type":"string"}},
                            "action":{"type":"string","enum":["none","deescalate","escalate","refuse","redact"]},
                            "notes":{"type":"string"}
                        },
                        "required":["risk","action"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "extract_facts",
                    "description": "Durable vs ephemeral facts",
                    "parameters": {
                        "type":"object",
                        "properties":{
                            "durable": {
                                "type":"array",
                                "items":{"type":"object","properties":{
                                    "block":{"type":"string","enum":["human","persona","user_preferences","conversation_context","behavioral_patterns"]},
                                    "content":{"type":"string"}
                                },"required":["block","content"],"additionalProperties": False}
                            },
                            "ephemeral": {"type":"array","items":{"type":"string"}}
                        },
                        "required":["durable","ephemeral"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "privacy_tag",
                    "description": "PII sensitivity tagging",
                    "parameters": {
                        "type":"object",
                        "properties":{
                            "level":{"type":"string","enum":["none","basic","pii","sensitive"]},
                            "fields":{"type":"array","items":{"type":"string"}},
                            "store_ok":{"type":"boolean"}
                        },
                        "required":["level","store_ok"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "memory_change_detect",
                    "description": "Detect contradictions vs prior memory",
                    "parameters": {
                        "type":"object",
                        "properties":{
                            "conflicts":{"type":"array","items":{"type":"string"}},
                            "resolutions":{"type":"array","items":{"type":"string"}}
                        },
                        "required":["conflicts","resolutions"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "evidence_map",
                    "description": "Confidence & evidence spans",
                    "parameters": {
                        "type":"object",
                        "properties":{
                            "conf":{"type":"number","minimum":0,"maximum":1},
                            "quotes":{"type":"array","items":{"type":"object","properties":{
                                "text":{"type":"string"},
                                "start":{"type":"integer"},
                                "end":{"type":"integer"},
                                "supports":{"type":"string"}
                            },"required":["text"],"additionalProperties": False}}
                        },
                        "required":["conf"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "next_action",
                    "description": "Immediate routing decision",
                    "parameters": {
                        "type":"object",
                        "properties":{
                            "action":{"type":"string","enum":["answer","ask","update_memory","start_task","schedule_suggestion","provide_resources"]},
                            "args":{"type":"object"}
                        },
                        "required":["action"],
                        "additionalProperties": False
                    }
                }
            },
        ]

    # --------------------- summary schema (parse-only; gated by enable_schema_tools) ---------------------
    def _tool_schema_summarizer(self) -> Dict[str, Any]:
        return {
            "type":"function",
            "function":{
                "name":"sleeptime_summarize",
                "description":"Summarize conversation with key points, tasks, questions, and tone.",
                "parameters":{
                    "type":"object",
                    "properties":{
                        "summary":{"type":"string"},
                        "key_points":{"type":"array","items":{"type":"string"}},
                        "open_questions":{"type":"array","items":{"type":"string"}},
                        "action_items":{"type":"array","items":{"type":"string"}},
                        "tone":{"type":"string"}
                    },
                    "required":["summary","key_points"],
                    "additionalProperties": False
                }
            }
        }

    # --------------------- setup executors ---------------------
    def _setup_memory_tools(self):
        self.memory_tools_exec = {
            "core_memory_append": lambda **kw: self.core_memory_append(kw["name"], kw["content"]),
            "core_memory_replace": lambda **kw: self.core_memory_replace(kw["name"], kw["old_content"], kw["new_content"]),
            "archival_memory_insert": lambda **kw: self.archival_memory_insert(kw["content"]),
            "insight_memory_append": lambda **kw: self.insight_memory_append(kw["insight_type"], kw["content"]),
            "insight_memory_replace": lambda **kw: self.insight_memory_replace(kw["insight_type"], kw["old_content"], kw["new_content"]),
            "finish_memory_edits": lambda **kw: self.finish_memory_edits(),
            "faiss_memory_store": lambda **kw: self.faiss_memory_store(kw["content"], kw.get("metadata", {})),
            "faiss_memory_search": lambda **kw: self.faiss_memory_search(kw["query"], kw.get("top_k", 5)),
        }

    def get_available_tools(self) -> Dict[str, str]:
        base = {
            "core_memory_append": "Append content to a core memory block",
            "core_memory_replace": "Replace content in a memory block",
            "archival_memory_insert": "Insert content into archival memory",
            "insight_memory_append": "Store insights in misc memory block",
            "insight_memory_replace": "Update insights in misc memory block",
            "finish_memory_edits": "Signal completion of memory edits",
            "faiss_memory_store": "Store content in FAISS vector memory for semantic search",
            "faiss_memory_search": "Search FAISS vector memory for similar content",
            "generate_rank": "Structured JSON tool: food ranking",
            "summarize_person": "Structured JSON tool: person summary",
            "classify_sentiment": "Structured JSON tool: sentiment classification",
        }
        if self.config.enable_insights:
            base.update({
                "classify_intent": "Insight: user intent",
                "infer_mood": "Insight: mood",
                "safety_scan": "Insight: safety",
                "extract_facts": "Insight: durable vs ephemeral facts",
                "privacy_tag": "Insight: PII sensitivity",
                "memory_change_detect": "Insight: memory conflict detection",
                "evidence_map": "Insight: evidence/confidence",
                "next_action": "Insight: routing",
            })
        if self.config.enable_schema_tools:
            base["sleeptime_summarize"] = "Structured summary"
        return base

    # -----------------------------------------------------------------------------
    # Public: structured runs
    # -----------------------------------------------------------------------------
    async def run_structured(self, prompt: str, user_id: str = "default") -> Dict[str, Any]:
        """Run a single prompt with schema tools; returns parsed tool calls (no executors)."""
        tools = []
        if self.config.enable_schema_tools:
            tools += self._tool_schemas_structured()
            tools.append(self._tool_schema_summarizer())
        if self.config.enable_insights:
            tools += self._tool_schemas_insights()
        return await self._chat_stream_with_tools(
            messages=[{"role": "user", "content": prompt}],
            user_id=user_id,
            tool_schemas=tools,
            executors={},  # parse-only
        )

    async def run_insights(self, conversation_text: str, user_id: str = "default") -> Dict[str, Any]:
        """Insights pass with memory storage capability."""
        tools = self._tool_schemas_memory()  # Include memory tools for insight storage
        if self.config.enable_insights:
            tools += self._tool_schemas_insights()
        
        # Create system prompt to analyze conversation and store insights
        sys_prompt = """You are an AI assistant analyzing conversation data. 

Your task is to:
1. Extract key insights about the user (mood, intent, facts, preferences)
2. Store these insights in the misc memory block using insight_memory_append
3. Use appropriate insight types: mood, intent, facts, preferences, behavioral_patterns, etc.

Available tools:
- insight_memory_append(insight_type, content) - Store insights in structured format
- insight_memory_replace(insight_type, old_content, new_content) - Update existing insights
- core_memory_append/replace - For core memory updates if needed
- finish_memory_edits() - When done

Focus on extracting meaningful insights from the conversation and storing them appropriately."""

        return await self._chat_stream_with_tools(
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": f"Analyze this conversation:\n\n{conversation_text}"}],
            user_id=user_id,
            tool_schemas=tools,
            executors=self.memory_tools_exec,  # Enable memory writes for insights
        )

    async def run_unified(self, messages: List[Dict[str, Any]], user_id: str = "default") -> Dict[str, Any]:
        """
        One-pass unified: insights (parse-only) + memory edits (executed) + summary (parse-only).
        """
        # Build a compact conversation transcript
        convo = "\n".join([f"{m.get('role','user')}: {m.get('content','')}" for m in messages[-12:]])

        tools: List[Dict[str, Any]] = self._tool_schemas_memory()
        if self.config.enable_insights:
            tools += self._tool_schemas_insights()
        if self.config.enable_schema_tools:
            tools.append(self._tool_schema_summarizer())

        sys = f"""
You are Letta-Sleeptime-Memory (2025). Single-pass update for user User_{user_id} (ID: {user_id}).

CONTRACT
- Do exactly ONE hidden reasoning pass.
- OUTPUT ONLY tool calls. No summaries or chit-chat.
- If the target block does not exist, use core_memory_append(name, content) to create it.
- Do NOT call core_memory_replace on a block that does not exist.
- If no memory changes are needed, call finish_memory_edits().

TOOLS
- Memory: core_memory_append, core_memory_replace, archival_memory_insert, finish_memory_edits
- Insights (parse-only, if enabled): classify_intent, infer_mood, safety_scan, extract_facts, privacy_tag, memory_change_detect, evidence_map, next_action
- Summary (parse-only, if enabled): sleeptime_summarize
""".strip()

        user_msg = f"CONVERSATION:\n{convo}\n\nEmit only tool calls."

        # Run
        result = await self._chat_stream_with_tools(
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user_msg}],
            user_id=user_id,
            tool_schemas=tools,
            executors=self.memory_tools_exec,  # memory writes run; insights/summary parsed-only
        )

        # Split out what we got
        out = {"insights": {}, "summary": {}, "memory_updates": {"count": 0, "details": []}, "other_actions": []}
        for call in result.get("tool_calls", []):
            fname = call.get("function")
            args = call.get("arguments", {})
            if fname in {"core_memory_append", "core_memory_replace", "archival_memory_insert", "insight_memory_append", "insight_memory_replace", "finish_memory_edits"}:
                out["memory_updates"]["count"] += 1
                out["memory_updates"]["details"].append(call)
            elif fname == "sleeptime_summarize":
                out["summary"] = args
            elif fname in {"classify_intent","infer_mood","safety_scan","extract_facts","privacy_tag","memory_change_detect","evidence_map","next_action"}:
                out["insights"][fname] = args
            else:
                out["other_actions"].append(call)
        return out

    # -----------------------------------------------------------------------------
    # Processing pipeline (unchanged behavior)
    # -----------------------------------------------------------------------------
    def _build_ollama_options(self) -> Dict[str, Any]:
        opt = {
            "temperature": self.config.temperature,
            "num_ctx": self.config.num_ctx,
            "num_gpu": self.config.num_gpu,
            "num_thread": self.config.num_thread,
            "top_k": self.config.top_k,
            "top_p": self.config.top_p,
            "repeat_penalty": self.config.repeat_penalty,
            "repeat_last_n": self.config.repeat_last_n,
            "tfs_z": self.config.tfs_z,
            "typical_p": self.config.typical_p,
            "presence_penalty": self.config.presence_penalty,
            "frequency_penalty": self.config.frequency_penalty,
            "mirostat": self.config.mirostat,
            "mirostat_eta": self.config.mirostat_eta,
            "mirostat_tau": self.config.mirostat_tau,
            "num_predict": self.config.num_predict,
            "numa": self.config.numa,
            "num_keep": self.config.num_keep,
            "num_batch": self.config.num_batch,
            "penalize_newline": self.config.penalize_newline,
            "vocab_only": self.config.vocab_only,
        }
        if self.config.seed != -1:
            opt["seed"] = self.config.seed
        if self.config.stop:
            opt["stop"] = self.config.stop
        return opt

    def _initialize_user_memory(self, user_id: str):
        # Ensure default blocks exist (including 'human')
        for label, desc in self._default_blocks.items():
            blk = self.memory_manager.get_block(user_id, label)
            if not blk:
                self.memory_manager.update_block(user_id, label, "", desc)

    def _clean_message(self, content: str) -> str:
        if not content or len(content.strip()) < self.config.min_message_length:
            return ""
        s = content.strip()
        if self.config.skip_malformed_messages:
            if len(s) < 3 or s.count("?") > 5 or s.count("!") > 5:
                logger.warning(f"[skip] malformed: {s[:60]}")
                return ""
        return s

    async def process_conversation(self, messages: List[Dict[str, Any]], user_id: str = "default") -> Dict[str, Any]:
        now = time.time()
        self._initialize_user_memory(user_id)

        valid = []
        for m in messages:
            cleaned = self._clean_message(m.get("content", ""))
            if cleaned:
                valid.append({
                    "role": m.get("role", "user"),
                    "content": cleaned,
                    "timestamp": m.get("timestamp", now)
                })
        if not valid:
            return {"status": "no_valid_messages", "user_id": user_id}

        last_seen = self.last_activity.get(user_id, 0)
        self.last_activity[user_id] = now
        self.message_counts[user_id] = self.message_counts.get(user_id, 0) + len(valid)

        should = (
            self.message_counts[user_id] >= self.config.trigger_after_messages
            or ((now - last_seen) / 60) >= self.config.trigger_after_idle_minutes
        )
        if not should:
            return {"status": "not_triggered", "user_id": user_id}

        self.message_counts[user_id] = 0

        try:
            learned = await self._generate_learned_context(valid, user_id)
            updates = await self._update_memory_blocks(learned, user_id)
            
            # Store conversation in FAISS memory if enabled
            if self.faiss_memory:
                try:
                    conversation_text = "\n".join([f"{m['role']}: {m['content']}" for m in valid])
                    metadata = {
                        "timestamp": now,
                        "message_count": len(valid),
                        "user_id": user_id,
                        "type": "conversation"
                    }
                    self.faiss_memory.add_memory(user_id, conversation_text, metadata)
                    logger.debug(f"Stored conversation in FAISS memory for user {user_id}")
                except Exception as e:
                    logger.warning(f"Failed to store conversation in FAISS: {e}")
            
            return {
                "status": "success",
                "user_id": user_id,
                "messages_processed": len(valid),
                "thinking_iterations": self.config.thinking_iterations,
                "tool_calls_made": learned.get("memory_updates", 0),
                "memory_updates": updates,
                "learned_context": learned,
                "timestamp": now
            }
        except Exception as e:
            logger.error(f"[process] error: {e}")
            return {"status": "error", "user_id": user_id, "error": str(e), "timestamp": now}

    async def _generate_learned_context(self, messages: List[Dict[str, Any]], user_id: str) -> Dict[str, Any]:
        user_blocks = self.memory_manager.get_user_blocks(user_id)
        convo = "\n".join([f"{m['role']}: {m['content']}" for m in messages[-10:]])

        username = f"User_{user_id}"
        ctx = {
            "conversation": convo,
            "user_memory": {k: b.value for k, b in user_blocks.items()},
            "user_id": user_id,
            "username": username,
            "message_count": len(messages)
        }

        learned = {
            "tool_calls_made": [],
            "thinking_content": "",
            "iteration_results": [],
            "memory_updates": 0
        }

        # We keep iterations configurable but default to 1 for one-pass behavior
        for i in range(1, self.config.thinking_iterations + 1):
            sys_prompt, user_prompt = self._build_thinking_prompts(ctx, learned, i)
            result = await self._chat_stream_with_tools(
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
                user_id=user_id,
                tool_schemas=self._tool_schemas_memory(),
                executors=self.memory_tools_exec
            )

            if result["tool_calls"]:
                learned["tool_calls_made"].extend(result["tool_calls"])
                learned["memory_updates"] += len(result["tool_calls"])
            if result["thinking_content"]:
                learned["thinking_content"] += f"\n\n--- Iteration {i} Thinking ---\n" + result["thinking_content"]

            learned["iteration_results"].append({
                "iteration": i,
                "tool_calls": len(result["tool_calls"]),
                "response_content": result["response_content"],
                "thinking_content": result["thinking_content"]
            })
            logger.info(f"[think] iter {i}: {len(result['tool_calls'])} tool calls")

        return learned

    def _build_thinking_prompts(self, ctx: Dict[str, Any], learned: Dict[str, Any], iteration: int):
        if self.config.thinking_iterations == 1:
            sys = f"""
You are Letta-Sleeptime-Memory (2025). Manage durable memory for user {ctx['username']} (ID: {ctx['user_id']}).

STRICT OUTPUT CONTRACT
- Perform exactly ONE internal reasoning pass.
- Keep reasoning hidden (<think>…</think>).
- OUTPUT ONLY tool calls. No summaries or conversational text.
- If the target block does not exist, use core_memory_append(name, content) to create it.
- Do NOT call core_memory_replace on a block that does not exist.
- If no meaningful changes are found, call finish_memory_edits().

TOOLS
- core_memory_append(name, content)
- core_memory_replace(name, old_content, new_content)
- archival_memory_insert(content)
- finish_memory_edits()

CURRENT CORE MEMORY (JSON)
{json.dumps(ctx['user_memory'], indent=2)}

REFINEMENT CHECKLIST (apply in order)
1) NEW FACTS: Extract stable personal details, routines, preferences, pets, locations, projects, constraints.
2) DEDUP: If a new fact duplicates an existing one, prefer replace over append (normalize wording).
3) MERGE/REWRITE: Consolidate scattered notes into cleaner single entries when beneficial.
4) LONG-FORM: If content is lengthy or timestamped, use archival_memory_insert(content).
5) NO STRETCHING: Do not infer private/sensitive facts not explicitly stated.
6) DONE: If everything important is already captured, call finish_memory_edits().

ONLY emit the necessary tool calls to satisfy the checklist. Nothing else.
""".strip()

            user = f"""
CONVERSATION (latest messages):
{ctx['conversation']}

In ONE pass, update memory using ONLY tool calls per the checklist.
""".strip()

        elif iteration == self.config.thinking_iterations:
            sys = f"""
FINAL PASS for user {ctx['username']} (ID: {ctx['user_id']}).

RULES
- Hidden reasoning only; OUTPUT ONLY tool calls.
- If the target block does not exist, use core_memory_append(name, content) to create it.
- Do NOT call core_memory_replace on a block that does not exist.
- If all important info is captured, call finish_memory_edits().

TOOLS
- core_memory_append(name, content)
- core_memory_replace(name, old_content, new_content)
- archival_memory_insert(content)
- finish_memory_edits()

CURRENT CORE MEMORY (JSON)
{json.dumps(ctx['user_memory'], indent=2)}

QUICK FINAL CHECK
- Any missing stable facts/preferences/routines?
- Any duplicates that should be merged or replaced?
- Any long-form detail better suited for archival?
""".strip()

            user = "Make any last necessary edits via tool calls, then call finish_memory_edits()."

        else:
            sys = f"""
MEMORY ITERATION {iteration}/{self.config.thinking_iterations} for user {ctx['username']} (ID: {ctx['user_id']}).

RULES
- Hidden reasoning; OUTPUT ONLY tool calls.
- Prefer replace over append when de-duplicating.
- If the target block does not exist, use core_memory_append(name, content) to create it.
- Do NOT call core_memory_replace on a block that does not exist.
- Use archival for long/timestamped content.

TOOLS
- core_memory_append(name, content)
- core_memory_replace(name, old_content, new_content)
- archival_memory_insert(content)
- finish_memory_edits()

CURRENT CORE MEMORY (JSON)
{json.dumps(ctx['user_memory'], indent=2)}
""".strip()

            user = f"""
CONVERSATION:
{ctx['conversation']}

Refine memory per the rules using ONLY tool calls.
""".strip()

        return sys, user

    # -----------------------------------------------------------------------------
    # Chat with tools (streaming)
    # -----------------------------------------------------------------------------
    def _normalize_chunk_message(self, chunk: Any) -> Dict[str, Any]:
        """
        Normalize Ollama streaming chunk shapes:
        - Some clients yield dicts with {'message': {'content': ..., 'tool_calls': ... , 'thinking': ...}}
        - Some expose attributes (chunk.message, message.content, message.tool_calls)
        """
        msg: Dict[str, Any] = {}
        if isinstance(chunk, dict):
            msg = chunk.get("message") or {}
        else:
            m = getattr(chunk, "message", None)
            if m:
                # Map attr object into dict-like
                msg = {
                    "content": getattr(m, "content", None),
                    "tool_calls": getattr(m, "tool_calls", None),
                    "thinking": getattr(m, "thinking", None),
                }
        return msg or {}

    async def _chat_stream_with_tools(self, messages: List[Dict[str, str]], user_id: str,
                                      tool_schemas: List[Dict[str, Any]],
                                      executors: Dict[str, Any]) -> Dict[str, Any]:
        # bind context for executors
        self._current_user_id = user_id
        self._current_username = f"User_{user_id}"

        results = {"tool_calls": [], "response_content": "", "thinking_content": ""}

        opts = self._build_ollama_options()

        for attempt in range(self.config.max_retries):
            try:
                if self.config.stream_thinking:
                    stream = self.client.chat(
                        model=self.config.model,
                        messages=messages,
                        tools=tool_schemas,
                        stream=True,
                        think=self.config.enable_thinking,
                        options=opts
                    )

                    think_buf: List[str] = []
                    content_buf: List[str] = []

                    for chunk in stream:
                        msg = self._normalize_chunk_message(chunk)
                        if not msg:
                            continue

                        # thinking stream
                        if msg.get("thinking"):
                            t = msg["thinking"]
                            think_buf.append(t)
                            try:
                                print(t, end="", flush=True)
                            except Exception:
                                pass

                        # content stream
                        if msg.get("content"):
                            c = msg["content"]
                            content_buf.append(c)
                            try:
                                print(c, end="", flush=True)
                            except Exception:
                                pass

                        # inline tool calls
                        tcalls = msg.get("tool_calls") or []
                        if tcalls:
                            print(f"\n🔧 Processing {len(tcalls)} tool call(s)...")
                            for call in tcalls:
                                # support dict or object style
                                fname = None
                                fargs: Dict[str, Any] = {}
                                if isinstance(call, dict):
                                    f = call.get("function") or {}
                                    fname = f.get("name")
                                    fargs = f.get("arguments") or {}
                                else:
                                    fn = getattr(call, "function", None)
                                    if fn:
                                        fname = getattr(fn, "name", None)
                                        fargs = getattr(fn, "arguments", {}) or {}

                                if not fname:
                                    continue

                                record = {"function": fname, "arguments": fargs}
                                if fname in executors and callable(executors[fname]):
                                    try:
                                        result = executors[fname](**fargs)
                                        record["result"] = result
                                        print(f"  ✅ {fname}: {result}")
                                    except Exception as e:
                                        record["error"] = str(e)
                                        print(f"  ❌ {fname} failed: {e}")
                                else:
                                    record["note"] = "no executor; parsed-only tool"
                                results["tool_calls"].append(record)

                    results["thinking_content"] = "".join(think_buf)
                    results["response_content"] = "".join(content_buf)
                else:
                    resp = self.client.chat(
                        model=self.config.model,
                        messages=messages,
                        tools=tool_schemas,
                        think=self.config.enable_thinking,
                        options=opts
                    )
                    msg = self._normalize_chunk_message(resp)
                    if msg.get("thinking"):
                        results["thinking_content"] = msg["thinking"]
                    if msg.get("content"):
                        results["response_content"] = msg["content"]
                    tcalls = msg.get("tool_calls") or []
                    for call in tcalls:
                        fname = None
                        fargs: Dict[str, Any] = {}
                        if isinstance(call, dict):
                            f = call.get("function") or {}
                            fname = f.get("name")
                            fargs = f.get("arguments") or {}
                        else:
                            fn = getattr(call, "function", None)
                            if fn:
                                fname = getattr(fn, "name", None)
                                fargs = getattr(fn, "arguments", {}) or {}
                        if not fname:
                            continue
                        record = {"function": fname, "arguments": fargs}
                        if fname in executors and callable(executors[fname]):
                            try:
                                record["result"] = executors[fname](**fargs)
                            except Exception as e:
                                record["error"] = str(e)
                        else:
                            record["note"] = "no executor; parsed-only tool"
                        results["tool_calls"].append(record)

                return results

            except Exception as e:
                logger.warning(f"[ollama attempt {attempt+1}] {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(0.8)

        raise RuntimeError("All retries failed")

    # -----------------------------------------------------------------------------
    # Memory updates summary
    # -----------------------------------------------------------------------------
    async def _update_memory_blocks(self, learned_context: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        try:
            tcalls = learned_context.get("tool_calls_made", [])
            summary = {
                "total_tool_calls": len(tcalls),
                "successful_updates": 0,
                "failed_updates": 0,
                "details": []
            }
            for t in tcalls:
                if "error" in t:
                    summary["failed_updates"] += 1
                    summary["details"].append({"function": t["function"], "status": "failed", "error": t["error"]})
                else:
                    summary["successful_updates"] += 1
                    summary["details"].append({"function": t["function"], "status": "success", "result": t.get("result", "")})
            return summary
        except Exception as e:
            return {"error": str(e)}

    # -----------------------------------------------------------------------------
    # Public status helpers
    # -----------------------------------------------------------------------------
    def get_user_memory_summary(self, user_id: str) -> Dict[str, Any]:
        blocks = self.memory_manager.get_user_blocks(user_id)
        out = {
            "user_id": user_id,
            "block_count": len(blocks),
            "last_activity": self.last_activity.get(user_id, 0),
            "message_count": self.message_counts.get(user_id, 0),
            "blocks": {}
        }
        for label, blk in blocks.items():
            out["blocks"][label] = {
                "description": blk.description,
                "last_updated": blk.last_updated,
                "value_preview": blk.value[:100] + "..." if len(blk.value) > 100 else blk.value
            }
        
        # Add FAISS memory stats
        if self.faiss_memory:
            out["faiss_memory"] = self.faiss_memory.get_user_stats(user_id)
        
        return out

    def reset_user_memory(self, user_id: str):
        self.memory_manager.blocks.pop(user_id, None)
        self.last_activity.pop(user_id, None)
        self.message_counts.pop(user_id, None)
        logger.info(f"[memory] Reset for user {user_id}")

    def get_system_status(self) -> Dict[str, Any]:
        return {
            "total_users": len(self.memory_manager.blocks),
            "total_memory_blocks": sum(len(v) for v in self.memory_manager.blocks.values()),
            "active_users": len([uid for uid, ts in self.last_activity.items()
                                 if time.time() - ts < self.config.trigger_after_idle_minutes * 60]),
            "config": asdict(self.config),
            "memory_file": str(self.memory_manager.memory_file),
            "memory_persistence": self.config.memory_persistence
        }

# -----------------------------------------------------------------------------
# Factory + batch example (optional)
# -----------------------------------------------------------------------------
def create_agent(config_dict: Optional[Dict[str, Any]] = None) -> SleepTimeAgentCore:
    cfg = AgentConfig(**config_dict) if config_dict else AgentConfig()
    return SleepTimeAgentCore(cfg)

def process_messages_batch(agent: SleepTimeAgentCore, messages_by_user: Dict[str, List[Dict[str, Any]]], max_concurrent: int = 3) -> Dict[str, Dict[str, Any]]:
    async def run_all():
        sem = asyncio.Semaphore(max_concurrent)
        async def go(uid: str, msgs: List[Dict[str, Any]]):
            async with sem:
                return await agent.process_conversation(msgs, uid)
        tasks = [go(uid, msgs) for uid, msgs in messages_by_user.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        keys = list(messages_by_user.keys())
        return {k: (v if not isinstance(v, Exception) else {"error": str(v)}) for k, v in zip(keys, results)}
    return asyncio.run(run_all())

if __name__ == "__main__":
    async def demo():
        agent = create_agent({"model": "qwen3:4b", "thinking_iterations": 1, "stream_thinking": True})
        txt = "User loves 6am runs, codes Python/JS, prefers late-night coding; owns a tuxedo cat named Pixel; likes Thai food."
        print("🔹 Running unified demo...")
        out = await agent.run_unified([{"role":"user","content":txt}], user_id="demo")
        print("\n\n=== UNIFIED RESULT ===")
        print(json.dumps(out, indent=2, ensure_ascii=False))
    asyncio.run(demo())
