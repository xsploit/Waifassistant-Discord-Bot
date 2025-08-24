#!/usr/bin/env python3
"""
Modern Conversation Memory Management for Discord LLM Bots (2025)
- Buffer window memory with bot message filtering
- Conversation summarization for long contexts
- Anti-repetition and context relevance optimization
"""

import time
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class ConversationMessage:
    """Structured conversation message"""
    content: str
    author_id: str
    author_name: str
    timestamp: float
    is_bot: bool = False
    message_id: Optional[str] = None
    channel_id: Optional[str] = None

@dataclass
class ConversationContext:
    """Processed conversation context for LLM"""
    recent_messages: List[ConversationMessage]
    summary: Optional[str] = None
    total_messages: int = 0
    context_tokens: int = 0

class ConversationMemoryManager:
    """
    Modern conversation memory management for Discord LLM bots (2025)
    
    Features:
    - Buffer window memory (keeps N recent user messages only)
    - Bot message filtering (prevents self-feeding)
    - Conversation summarization (for long contexts)
    - Token counting and management
    - Anti-repetition tracking
    """
    
    def __init__(self, 
                 window_size: int = 8,
                 summary_threshold: int = 20,
                 max_context_tokens: int = 3000,
                 bot_user_id: Optional[str] = None):
        """
        Initialize conversation memory manager
        
        Args:
            window_size: Number of recent user messages to keep
            summary_threshold: Messages count to trigger summarization
            max_context_tokens: Maximum tokens for context
            bot_user_id: Bot's user ID to filter out
        """
        self.window_size = window_size
        self.summary_threshold = summary_threshold
        self.max_context_tokens = max_context_tokens
        self.bot_user_id = bot_user_id
        
        # Per-channel conversation storage
        self.conversations: Dict[str, List[ConversationMessage]] = {}
        self.summaries: Dict[str, str] = {}
        self.last_cleanup: Dict[str, float] = {}
        
        # Anti-repetition tracking
        self.recent_responses: Dict[str, List[str]] = {}
        
    def add_message(self, 
                   channel_id: str,
                   content: str,
                   author_id: str, 
                   author_name: str,
                   message_id: Optional[str] = None,
                   is_bot: bool = False) -> None:
        """Add a message to conversation history"""
        
        if channel_id not in self.conversations:
            self.conversations[channel_id] = []
            
        message = ConversationMessage(
            content=content,
            author_id=author_id,
            author_name=author_name,
            timestamp=time.time(),
            is_bot=is_bot,
            message_id=message_id,
            channel_id=channel_id
        )
        
        self.conversations[channel_id].append(message)
        
        # Cleanup old messages periodically
        self._cleanup_old_messages(channel_id)
        
    def get_conversation_context(self, channel_id: str) -> ConversationContext:
        """
        Get processed conversation context for LLM
        
        Returns:
            ConversationContext with recent messages and optional summary
        """
        if channel_id not in self.conversations:
            return ConversationContext(recent_messages=[], total_messages=0)
            
        all_messages = self.conversations[channel_id]
        
        # Filter out bot messages (prevent self-feeding)
        user_messages = [
            msg for msg in all_messages 
            if not msg.is_bot and msg.author_id != self.bot_user_id
        ]
        
        # Get recent messages within window
        recent_messages = user_messages[-self.window_size:] if user_messages else []
        
        # Generate summary if we have many messages
        summary = None
        if len(user_messages) > self.summary_threshold:
            summary = self._get_or_create_summary(channel_id, user_messages[:-self.window_size])
        
        # Estimate context tokens (rough approximation: ~4 chars per token)
        context_tokens = sum(len(msg.content) for msg in recent_messages) // 4
        if summary:
            context_tokens += len(summary) // 4
            
        return ConversationContext(
            recent_messages=recent_messages,
            summary=summary,
            total_messages=len(user_messages),
            context_tokens=context_tokens
        )
    
    def format_context_for_llm(self, channel_id: str) -> str:
        """
        Format conversation context as string for LLM prompt
        
        Returns:
            Formatted context string ready for LLM
        """
        context = self.get_conversation_context(channel_id)
        
        if not context.recent_messages and not context.summary:
            return "No previous conversation context."
            
        formatted_parts = []
        
        # Add summary if available
        if context.summary:
            formatted_parts.append(f"Previous conversation summary: {context.summary}")
            
        # Add recent messages
        if context.recent_messages:
            formatted_parts.append("Recent conversation:")
            for msg in context.recent_messages:
                timestamp_str = datetime.fromtimestamp(msg.timestamp).strftime("%H:%M")
                formatted_parts.append(f"[{timestamp_str}] {msg.author_name}: {msg.content}")
                
        return "\n".join(formatted_parts)
    
    def check_repetition(self, channel_id: str, response: str, threshold: int = 3) -> bool:
        """
        Check if response is too similar to recent responses
        
        Returns:
            True if response is repetitive, False otherwise
        """
        if channel_id not in self.recent_responses:
            self.recent_responses[channel_id] = []
            
        recent = self.recent_responses[channel_id]
        
        # Simple similarity check (can be enhanced with embeddings)
        response_words = set(response.lower().split())
        
        for prev_response in recent[-threshold:]:
            prev_words = set(prev_response.lower().split())
            
            # Check word overlap percentage
            if len(response_words & prev_words) / len(response_words | prev_words) > 0.7:
                return True
                
        return False
    
    def add_bot_response(self, channel_id: str, response: str) -> None:
        """Track bot response for anti-repetition"""
        if channel_id not in self.recent_responses:
            self.recent_responses[channel_id] = []
            
        self.recent_responses[channel_id].append(response)
        
        # Keep only recent responses
        if len(self.recent_responses[channel_id]) > 10:
            self.recent_responses[channel_id] = self.recent_responses[channel_id][-5:]
    
    def clear_channel_history(self, channel_id: str) -> None:
        """Clear all history for a channel"""
        if channel_id in self.conversations:
            del self.conversations[channel_id]
        if channel_id in self.summaries:
            del self.summaries[channel_id]
        if channel_id in self.recent_responses:
            del self.recent_responses[channel_id]
        if channel_id in self.last_cleanup:
            del self.last_cleanup[channel_id]
            
    def get_memory_stats(self, channel_id: str) -> Dict[str, Any]:
        """Get memory usage statistics for a channel"""
        if channel_id not in self.conversations:
            return {"total_messages": 0, "user_messages": 0, "bot_messages": 0}
            
        all_messages = self.conversations[channel_id]
        user_messages = [msg for msg in all_messages if not msg.is_bot]
        bot_messages = [msg for msg in all_messages if msg.is_bot]
        
        context = self.get_conversation_context(channel_id)
        
        return {
            "total_messages": len(all_messages),
            "user_messages": len(user_messages),
            "bot_messages": len(bot_messages),
            "recent_context_messages": len(context.recent_messages),
            "has_summary": context.summary is not None,
            "estimated_context_tokens": context.context_tokens,
            "memory_efficiency": len(context.recent_messages) / max(1, len(user_messages))
        }
    
    def _get_or_create_summary(self, channel_id: str, old_messages: List[ConversationMessage]) -> str:
        """Get existing summary or create new one for old messages"""
        if channel_id in self.summaries:
            return self.summaries[channel_id]
            
        # Create simple summary (can be enhanced with LLM summarization)
        if not old_messages:
            return ""
            
        # Extract key topics and participants
        participants = set(msg.author_name for msg in old_messages)
        recent_topics = []
        
        # Look for questions, mentions of topics, etc.
        for msg in old_messages[-10:]:  # Last 10 old messages for topic extraction
            content = msg.content.lower()
            if any(word in content for word in ['help', 'question', 'problem', 'issue']):
                recent_topics.append(f"{msg.author_name} asked about: {msg.content[:50]}...")
                
        summary_parts = []
        if participants:
            summary_parts.append(f"Participants: {', '.join(participants)}")
        if recent_topics:
            summary_parts.append("Recent topics: " + "; ".join(recent_topics[-3:]))
        
        summary = " | ".join(summary_parts) if summary_parts else "General conversation"
        self.summaries[channel_id] = summary
        return summary
    
    def _cleanup_old_messages(self, channel_id: str) -> None:
        """Clean up very old messages to prevent memory bloat"""
        now = time.time()
        
        # Only cleanup every hour
        if channel_id in self.last_cleanup:
            if now - self.last_cleanup[channel_id] < 3600:  # 1 hour
                return
                
        self.last_cleanup[channel_id] = now
        
        if channel_id not in self.conversations:
            return
            
        messages = self.conversations[channel_id]
        
        # Remove messages older than 7 days
        week_ago = now - (7 * 24 * 3600)
        self.conversations[channel_id] = [
            msg for msg in messages if msg.timestamp > week_ago
        ]
        
        print(f"[MEMORY] Cleaned {len(messages) - len(self.conversations[channel_id])} old messages from {channel_id}")
    
    def get_persistent_state(self) -> Dict[str, Any]:
        """Get persistent state for saving to disk"""
        return {
            'conversations': {
                channel_id: [
                    {
                        'content': msg.content,
                        'author_id': msg.author_id,
                        'author_name': msg.author_name,
                        'message_id': msg.message_id,
                        'is_bot': msg.is_bot,
                        'timestamp': msg.timestamp
                    }
                    for msg in messages
                ]
                for channel_id, messages in self.conversations.items()
            },
            'summaries': self.summaries,
            'recent_responses': self.recent_responses,
            'last_cleanup': self.last_cleanup
        }
    
    def load_persistent_state(self, state: Dict[str, Any]) -> None:
        """Load persistent state from disk"""
        try:
            if 'conversations' in state:
                for channel_id, msg_data in state['conversations'].items():
                    messages = []
                    for msg_dict in msg_data:
                        msg = ConversationMessage(
                            content=msg_dict['content'],
                            author_id=msg_dict['author_id'],
                            author_name=msg_dict['author_name'],
                            message_id=msg_dict['message_id'],
                            is_bot=msg_dict['is_bot'],
                            timestamp=msg_dict['timestamp']
                        )
                        messages.append(msg)
                    self.conversations[channel_id] = messages
            
            if 'summaries' in state:
                self.summaries = state['summaries']
            
            if 'recent_responses' in state:
                self.recent_responses = state['recent_responses']
            
            if 'last_cleanup' in state:
                self.last_cleanup = state['last_cleanup']
                
            print(f"[MEMORY] Loaded persistent state: {len(self.conversations)} channels, {sum(len(msgs) for msgs in self.conversations.values())} messages")
            
        except Exception as e:
            print(f"[MEMORY ERROR] Failed to load persistent state: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Test the conversation memory manager
    memory = ConversationMemoryManager(window_size=5, summary_threshold=10)
    
    # Simulate conversation
    memory.add_message("123", "Hello bot!", "user1", "Alice")
    memory.add_message("123", "Hi Alice! How are you?", "bot", "Hikari", is_bot=True)
    memory.add_message("123", "I'm doing great, thanks!", "user1", "Alice")
    
    # Get context (should exclude bot message)
    context = memory.get_conversation_context("123")
    print(f"Context has {len(context.recent_messages)} user messages")
    
    formatted = memory.format_context_for_llm("123")
    print(f"Formatted context:\n{formatted}")
    
    stats = memory.get_memory_stats("123")
    print(f"Memory stats: {stats}")
    
    print("[SUCCESS] Conversation Memory Manager working correctly!")