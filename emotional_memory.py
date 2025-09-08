#!/usr/bin/env python3
"""
Emotional Memory System for Discord Bot (2025)
- User emotional profiles with persistent storage
- Dynamic personality evolution
- Memory-driven emotional responses
- Anti-context-pollution safety measures
"""

import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import os

# =============================================================================
# EMOTIONAL MEMORY DATA STRUCTURES
# =============================================================================

@dataclass
class EmotionalMemory:
    """Individual emotional memory entry"""
    content: str
    memory_type: str  # MEMORY, PERSONAL, TOOL_KNOWLEDGE, EMOTIONAL
    importance_score: float  # 0.0 to 1.0
    emotional_context: str  # How the bot felt about this
    timestamp: float
    metadata: Dict[str, Any] = None

@dataclass
class UserEmotionalProfile:
    """Complete emotional profile for a user"""
    user_id: str
    username: str
    current_mood: str = "neutral"
    relationship_level: str = "stranger"  # stranger, acquaintance, friend, close_friend
    trust_score: float = 0.5  # 0.0 to 1.0
    familiarity_level: float = 0.0  # 0.0 to 1.0
    conversation_count: int = 0
    last_interaction: float = 0.0
    
    # Emotional state
    mood_points: float = 0.0  # -100 to +100
    emotional_stability: float = 0.8  # 0.0 to 1.0
    
    # Personality traits (evolve over time)
    personality_traits: Dict[str, float] = None  # trait_name: strength
    
    # Memory storage
    memories: List[EmotionalMemory] = None
    emotional_history: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.personality_traits is None:
            self.personality_traits = {
                "playfulness": 0.5,
                "curiosity": 0.7,
                "empathy": 0.6,
                "humor": 0.4,
                "seriousness": 0.3
            }
        if self.memories is None:
            self.memories = []
        if self.emotional_history is None:
            self.emotional_history = []

# =============================================================================
# EMOTIONAL MEMORY STORAGE
# =============================================================================

class EmotionalMemoryStorage:
    """Persistent storage for emotional memory system"""
    
    def __init__(self, storage_dir: str = "emotional_memory"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.profiles_file = self.storage_dir / "user_profiles.json"
        self.backup_dir = self.storage_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # Load existing profiles
        self.user_profiles: Dict[str, UserEmotionalProfile] = {}
        self.load_all_profiles()
        
        # Auto-save settings
        self.last_save = time.time()
        self.auto_save_interval = 300  # 5 minutes
        
    def load_all_profiles(self) -> None:
        """Load all user profiles from storage"""
        if not self.profiles_file.exists():
            print(f"[EMOTIONAL MEMORY] No existing profiles found, starting fresh")
            return
            
        try:
            with open(self.profiles_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for user_id, profile_data in data.items():
                # Fix: Convert memories back to EmotionalMemory dataclass instances
                if 'memories' in profile_data and isinstance(profile_data['memories'], list):
                    memories = []
                    for memory_data in profile_data['memories']:
                        if isinstance(memory_data, dict):
                            # Convert dict back to EmotionalMemory dataclass
                            memory = EmotionalMemory(**memory_data)
                            memories.append(memory)
                        else:
                            # Already an EmotionalMemory object
                            memories.append(memory_data)
                    profile_data['memories'] = memories
                
                # Convert back to UserEmotionalProfile
                profile = UserEmotionalProfile(**profile_data)
                self.user_profiles[user_id] = profile
                
            print(f"[EMOTIONAL MEMORY] Loaded {len(self.user_profiles)} user profiles")
            
        except Exception as e:
            print(f"[EMOTIONAL MEMORY] Error loading profiles: {e}")
            # Create backup of corrupted file
            backup_file = self.backup_dir / f"corrupted_profiles_{int(time.time())}.json"
            if self.profiles_file.exists():
                self.profiles_file.rename(backup_file)
            print(f"[EMOTIONAL MEMORY] Created backup of corrupted file: {backup_file}")
    
    def save_all_profiles(self) -> None:
        """Save all user profiles to storage"""
        try:
            # Convert profiles to dict format
            data = {}
            for user_id, profile in self.user_profiles.items():
                data[user_id] = asdict(profile)
            
            # Create backup before saving
            if self.profiles_file.exists():
                backup_file = self.backup_dir / f"profiles_backup_{int(time.time())}.json"
                self.profiles_file.rename(backup_file)
                print(f"[EMOTIONAL MEMORY] Created backup: {backup_file}")
            
            # Save new profiles
            with open(self.profiles_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.last_save = time.time()
            print(f"[EMOTIONAL MEMORY] Saved {len(self.user_profiles)} user profiles")
            
        except Exception as e:
            print(f"[EMOTIONAL MEMORY] Error saving profiles: {e}")
    
    def auto_save_check(self) -> None:
        """Check if auto-save is needed"""
        if time.time() - self.last_save > self.auto_save_interval:
            self.save_all_profiles()
    
    def get_user_profile(self, user_id: str) -> UserEmotionalProfile:
        """Get or create user profile"""
        if user_id not in self.user_profiles:
            # Create new profile
            profile = UserEmotionalProfile(
                user_id=user_id,
                username="Unknown",
                last_interaction=time.time()
            )
            self.user_profiles[user_id] = profile
            print(f"[EMOTIONAL MEMORY] Created new profile for user {user_id}")
        
        return self.user_profiles[user_id]
    
    def update_user_profile(self, user_id: str, **updates) -> None:
        """Update user profile with new data"""
        profile = self.get_user_profile(user_id)
        
        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        
        # Update last interaction
        profile.last_interaction = time.time()
        
        # Check for auto-save
        self.auto_save_check()
    
    def get_persistent_state(self) -> Dict[str, Any]:
        """Get persistent state for saving to disk"""
        return {
            'user_profiles': {
                user_id: asdict(profile)
                for user_id, profile in self.user_profiles.items()
            },
            'last_save': self.last_save
        }
    
    def load_persistent_state(self, state: Dict[str, Any]) -> None:
        """Load persistent state from disk"""
        try:
            if 'user_profiles' in state:
                for user_id, profile_data in state['user_profiles'].items():
                    # Fix: Convert memories back to EmotionalMemory dataclass instances
                    if 'memories' in profile_data and isinstance(profile_data['memories'], list):
                        memories = []
                        for memory_data in profile_data['memories']:
                            if isinstance(memory_data, dict):
                                # Convert dict back to EmotionalMemory dataclass
                                memory = EmotionalMemory(**memory_data)
                                memories.append(memory)
                            else:
                                # Already an EmotionalMemory object
                                memories.append(memory_data)
                        profile_data['memories'] = memories
                    
                    profile = UserEmotionalProfile(**profile_data)
                    self.user_profiles[user_id] = profile
                
                print(f"[EMOTIONAL MEMORY] Loaded {len(self.user_profiles)} user profiles from persistent state")
            
            if 'last_save' in state:
                self.last_save = state['last_save']
                
        except Exception as e:
            print(f"[EMOTIONAL MEMORY ERROR] Failed to load persistent state: {e}")

# =============================================================================
# EMOTIONAL STATE ENGINE
# =============================================================================

class EmotionalStateEngine:
    """Manages emotional state transitions and personality evolution"""
    
    def __init__(self, storage: EmotionalMemoryStorage):
        self.storage = storage
        
        # Emotional state constants
        self.MOOD_RANGE = (-100, 100)
        self.MOOD_CHANGE_RATE = 0.1  # How quickly mood changes
        self.PERSONALITY_EVOLUTION_RATE = 0.05  # How quickly personality evolves
        
        # Mood categories
        self.MOOD_CATEGORIES = {
            "ecstatic": (80, 100),
            "excited": (60, 79),
            "happy": (40, 59),
            "content": (20, 39),
            "neutral": (-19, 19),
            "concerned": (-39, -20),
            "sad": (-59, -40),
            "upset": (-79, -60),
            "devastated": (-100, -80)
        }
    
    def get_mood_category(self, mood_points: float) -> str:
        """Convert mood points to mood category"""
        for category, (min_val, max_val) in self.MOOD_CATEGORIES.items():
            if min_val <= mood_points <= max_val:
                return category
        return "neutral"
    
    def update_user_mood(self, user_id: str, mood_change: float, reason: str = "") -> None:
        """Update user's mood based on interaction"""
        profile = self.storage.get_user_profile(user_id)
        
        # Calculate new mood with stability factor
        stability_factor = profile.emotional_stability
        actual_change = mood_change * stability_factor * self.MOOD_CHANGE_RATE
        
        # Update mood points
        profile.mood_points = max(self.MOOD_RANGE[0], 
                                min(self.MOOD_RANGE[1], 
                                    profile.mood_points + actual_change))
        
        # Update current mood category
        profile.current_mood = self.get_mood_category(profile.mood_points)
        
        # Record emotional history
        emotional_entry = {
            "timestamp": time.time(),
            "mood_change": mood_change,
            "new_mood": profile.current_mood,
            "mood_points": profile.mood_points,
            "reason": reason
        }
        profile.emotional_history.append(emotional_entry)
        
        # Keep only last 50 emotional entries
        if len(profile.emotional_history) > 50:
            profile.emotional_history = profile.emotional_history[-50:]
        
        # Update storage
        self.storage.update_user_profile(user_id, 
                                       mood_points=profile.mood_points,
                                       current_mood=profile.current_mood,
                                       emotional_history=profile.emotional_history)
        
        print(f"[EMOTIONAL ENGINE] User {user_id} mood updated: {profile.current_mood} ({profile.mood_points:.1f})")
    
    def evolve_personality(self, user_id: str, interaction_quality: float) -> None:
        """Evolve personality based on interaction quality"""
        profile = self.storage.get_user_profile(user_id)
        
        # Interaction quality affects which traits evolve
        if interaction_quality > 0.7:  # Positive interaction
            # Strengthen positive traits
            profile.personality_traits["playfulness"] = min(1.0, 
                profile.personality_traits["playfulness"] + self.PERSONALITY_EVOLUTION_RATE)
            profile.personality_traits["empathy"] = min(1.0, 
                profile.personality_traits["empathy"] + self.PERSONALITY_EVOLUTION_RATE)
        elif interaction_quality < 0.3:  # Negative interaction
            # Strengthen defensive traits
            profile.personality_traits["seriousness"] = min(1.0, 
                profile.personality_traits["seriousness"] + self.PERSONALITY_EVOLUTION_RATE)
        
        # Always evolve curiosity slightly
        profile.personality_traits["curiosity"] = min(1.0, 
            profile.personality_traits["curiosity"] + self.PERSONALITY_EVOLUTION_RATE * 0.5)
        
        # Update storage
        self.storage.update_user_profile(user_id, 
                                       personality_traits=profile.personality_traits)
        
        print(f"[EMOTIONAL ENGINE] User {user_id} personality evolved based on interaction quality {interaction_quality:.2f}")

# =============================================================================
# MAIN EMOTIONAL MEMORY MANAGER
# =============================================================================

class EmotionalMemoryManager:
    """Main interface for the emotional memory system"""
    
    def __init__(self, storage_dir: str = "emotional_memory"):
        self.storage = EmotionalMemoryStorage(storage_dir)
        self.emotional_engine = EmotionalStateEngine(self.storage)
        
        print(f"[EMOTIONAL MEMORY] System initialized with storage: {storage_dir}")
    
    def get_user_profile(self, user_id: str) -> UserEmotionalProfile:
        """Get user's emotional profile"""
        return self.storage.get_user_profile(user_id)
    
    def update_user_mood(self, user_id: str, mood_change: float, reason: str = "") -> None:
        """Update user's mood"""
        self.emotional_engine.update_user_mood(user_id, mood_change, reason)
    
    def evolve_personality(self, user_id: str, interaction_quality: float) -> None:
        """Evolve user's personality"""
        self.emotional_engine.evolve_personality(user_id, interaction_quality)
    
    def add_memory(self, user_id: str, content: str, memory_type: str, 
                   importance_score: float, emotional_context: str, 
                   metadata: Dict[str, Any] = None) -> None:
        """Add a new memory for a user"""
        profile = self.storage.get_user_profile(user_id)
        
        memory = EmotionalMemory(
            content=content,
            memory_type=memory_type,
            importance_score=importance_score,
            emotional_context=emotional_context,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        profile.memories.append(memory)
        
        # Update relationship metrics
        profile.conversation_count += 1
        profile.last_interaction = time.time()
        
        # Update familiarity based on number of interactions
        # Familiarity increases logarithmically (0.0 to 1.0)
        profile.familiarity_level = min(1.0, profile.conversation_count / 200.0)
        
        # Update relationship level based on conversation count and familiarity
        if profile.conversation_count >= 100 and profile.familiarity_level >= 0.7:
            profile.relationship_level = "close_friend"
        elif profile.conversation_count >= 50 and profile.familiarity_level >= 0.4:
            profile.relationship_level = "friend"  
        elif profile.conversation_count >= 20 and profile.familiarity_level >= 0.1:
            profile.relationship_level = "acquaintance"
        else:
            profile.relationship_level = "stranger"
        
        # Update trust score slightly with each positive interaction
        if "positive" in emotional_context.lower() or importance_score > 0.7:
            profile.trust_score = min(1.0, profile.trust_score + 0.005)
        elif "negative" in emotional_context.lower() and importance_score > 0.5:
            profile.trust_score = max(0.0, profile.trust_score - 0.01)
        
        # Keep only most important memories (top 100 by importance)
        profile.memories.sort(key=lambda m: m.importance_score, reverse=True)
        if len(profile.memories) > 100:
            profile.memories = profile.memories[:100]
        
        # Update storage with all changes
        self.storage.update_user_profile(
            user_id, 
            memories=profile.memories,
            conversation_count=profile.conversation_count,
            familiarity_level=profile.familiarity_level,
            relationship_level=profile.relationship_level,
            trust_score=profile.trust_score,
            last_interaction=profile.last_interaction
        )
        
        print(f"[EMOTIONAL MEMORY] Added {memory_type} memory for user {user_id} (conv #{profile.conversation_count}, fam: {profile.familiarity_level:.1%}, rel: {profile.relationship_level}): {content[:50]}...")
    
    def recalculate_user_stats(self, user_id: str) -> None:
        """Recalculate conversation count and relationship metrics based on existing memories"""
        profile = self.storage.get_user_profile(user_id)
        
        # Update conversation count to match number of memories
        actual_memory_count = len(profile.memories)
        if actual_memory_count > profile.conversation_count:
            profile.conversation_count = actual_memory_count
            
        # Recalculate familiarity
        profile.familiarity_level = min(1.0, profile.conversation_count / 200.0)
        
        # Recalculate relationship level
        if profile.conversation_count >= 100 and profile.familiarity_level >= 0.7:
            profile.relationship_level = "close_friend"
        elif profile.conversation_count >= 50 and profile.familiarity_level >= 0.4:
            profile.relationship_level = "friend"  
        elif profile.conversation_count >= 20 and profile.familiarity_level >= 0.1:
            profile.relationship_level = "acquaintance"
        else:
            profile.relationship_level = "stranger"
            
        # Update last interaction from most recent memory
        if profile.memories:
            profile.last_interaction = max(memory.timestamp for memory in profile.memories)
        
        # Update storage
        self.storage.update_user_profile(
            user_id,
            conversation_count=profile.conversation_count,
            familiarity_level=profile.familiarity_level,
            relationship_level=profile.relationship_level,
            last_interaction=profile.last_interaction
        )
        
        print(f"[EMOTIONAL MEMORY] Recalculated stats for user {user_id}: {profile.conversation_count} conversations, {profile.familiarity_level:.1%} familiarity, {profile.relationship_level}")
    
    def get_relevant_memories(self, user_id: str, query: str, limit: int = 5) -> List[EmotionalMemory]:
        """Get most relevant memories for a user (simple keyword matching for now)"""
        profile = self.storage.get_user_profile(user_id)
        
        if not profile.memories:
            return []
        
        # Simple relevance scoring (will be enhanced with FAISS later)
        scored_memories = []
        query_words = set(query.lower().split())
        
        for memory in profile.memories:
            memory_words = set(memory.content.lower().split())
            word_overlap = len(query_words & memory_words)
            relevance_score = word_overlap * memory.importance_score
            
            scored_memories.append((relevance_score, memory))
        
        # Sort by relevance and return top results
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in scored_memories[:limit]]
    
    def save_all_profiles(self) -> None:
        """Save all profiles to storage"""
        self.storage.save_all_profiles()
    
    # Tool knowledge methods removed - now handled by VectorToolKnowledge system
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        total_users = len(self.storage.user_profiles)
        total_memories = sum(len(p.memories) for p in self.storage.user_profiles.values())
        
        return {
            "total_users": total_users,
            "total_memories": total_memories,
            "storage_directory": str(self.storage.storage_dir),
            "last_save": self.storage.last_save
        }
    
    def get_persistent_state(self) -> Dict[str, Any]:
        """Get persistent state for saving to disk"""
        return self.storage.get_persistent_state()
    
    def load_persistent_state(self, state: Dict[str, Any]) -> None:
        """Load persistent state from disk"""
        try:
            self.storage.load_persistent_state(state)
            print(f"[EMOTIONAL MEMORY] Loaded persistent state from disk")
        except Exception as e:
            print(f"[EMOTIONAL MEMORY ERROR] Failed to load persistent state: {e}")

# =============================================================================
# TESTING AND EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Test the emotional memory system
    print("[TESTING] Emotional Memory System")
    
    # Initialize system
    emm = EmotionalMemoryManager("test_emotional_memory")
    
    # Test user profile creation
    user_id = "test_user_123"
    profile = emm.get_user_profile(user_id)
    print(f"[TEST] Created profile for {user_id}: {profile.current_mood}")
    
    # Test mood updates
    emm.update_user_mood(user_id, 20, "User was very friendly")
    emm.update_user_mood(user_id, -10, "User seemed upset")
    
    # Test personality evolution
    emm.evolve_personality(user_id, 0.8)
    
    # Test memory storage
    emm.add_memory(user_id, "User likes pizza", "PERSONAL", 0.8, "positive")
    emm.add_memory(user_id, "User asked about programming", "MEMORY", 0.6, "curious")
    
    # Test memory retrieval
    memories = emm.get_relevant_memories(user_id, "pizza")
    print(f"[TEST] Found {len(memories)} memories about pizza")
    
    # Save everything
    emm.save_all_profiles()
    
    # Get stats
    stats = emm.get_system_stats()
    print(f"[TEST] System stats: {stats}")
    
    print("[TEST] Emotional Memory System test completed successfully!")
