#!/usr/bin/env python3
"""
Optimized Discord Bot with Ollama + POML Integration
- All performance optimizations (KV cache, flash attention, BPE)
- Complete tool suite (web search, scrape, calculate, etc.)
- Perfect Discord @ mention handling
- Anti-repetition system
- Dynamic model management
- POML (Prompt Orchestration Markup Language) support
"""

import discord
from discord.ext import commands
import asyncio
import aiohttp
import json
import os
import time
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from conversation_memory import ConversationMemoryManager
import hashlib
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Try to import AI Intent Classifier (optional dependency)
try:
    from ai_intent_classifier import AIIntentClassifier, IntentClassification
    AI_CLASSIFIER_AVAILABLE = True
    print("[OK] AI Intent Classifier import successful")
except ImportError as e:
    AI_CLASSIFIER_AVAILABLE = False
    print(f"[WARNING] AI Intent Classifier not available: {e}")
    print("[INFO] Bot will use fallback mood system until torch/transformers are installed")

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[OK] Environment variables loaded from .env file")
except ImportError:
    print("[WARNING] python-dotenv not installed - install with: pip install python-dotenv")

# POML Integration (optional)
try:
    from poml import poml
    POML_AVAILABLE = True
    print("[OK] POML available - Using correct core API")
except ImportError:
    POML_AVAILABLE = False
    print("[WARNING] POML not installed - Using basic prompts (pip install poml to enable)")

# =============================================================================
# POML TEMPLATE CACHING SYSTEM
# =============================================================================

class POMLCache:
    """Context-aware POML template cache to eliminate processing delays"""
    
    def __init__(self):
        self.compiled_results = {}  # Cache compiled results, not raw templates
        self.template_hashes = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_cache_size = 100  # Prevent memory bloat
        
    def _generate_context_key(self, template_name: str, context: dict) -> str:
        """Generate a cache key based on template and context"""
        # Create a stable key from context values that affect POML processing
        context_parts = []
        
        # Include template name
        context_parts.append(f"template:{template_name}")
        
        # Include mood (grouped by larger intervals for better cache hits)
        if 'mood_points' in context:
            # Group moods into larger buckets: very negative, negative, neutral, positive, very positive
            mood_points = context['mood_points']
            if mood_points <= -5:
                mood_group = "very_negative"
            elif mood_points <= -1:
                mood_group = "negative"
            elif mood_points <= 1:
                mood_group = "neutral"
            elif mood_points <= 5:
                mood_group = "positive"
            else:
                mood_group = "very_positive"
            context_parts.append(f"mood:{mood_group}")
        
        # Include memory context for better personalization
        if 'user_memory' in context:
            memory = context['user_memory']
            # Include memory block count (grouped for cache efficiency)
            block_count = memory.get('block_count', 0)
            if block_count == 0:
                memory_group = "no_memory"
            elif block_count <= 5:
                memory_group = "low_memory"
            elif block_count <= 15:
                memory_group = "medium_memory"
            else:
                memory_group = "high_memory"
            context_parts.append(f"memory:{memory_group}")
            
            # Include last activity (grouped by time ranges)
            last_activity = memory.get('last_activity', 0)
            if last_activity == 0:
                activity_group = "new_user"
            elif time.time() - last_activity < 3600:  # 1 hour
                activity_group = "recent"
            elif time.time() - last_activity < 86400:  # 1 day
                activity_group = "daily"
            else:
                activity_group = "older"
            context_parts.append(f"activity:{activity_group}")
        
        # Don't include tone in cache key - it changes too frequently and reduces cache hits
        # The mood grouping above should capture most personality variations
        
        # Create a hash of the context key
        context_str = "|".join(sorted(context_parts))
        context_key = hashlib.md5(context_str.encode()).hexdigest()[:12]
        
        return context_key
    
    def get_cached_result(self, template_name: str, context: dict):
        """Get cached result if available for this context combination"""
        context_key = self._generate_context_key(template_name, context)
        cache_key = f"{template_name}:{context_key}"
        
        # Debug: Show what we're looking for
        print(f"[POML CACHE DEBUG] Looking for key: {cache_key}")
        print(f"[POML CACHE DEBUG] Available keys: {list(self.compiled_results.keys())[:5]}...")
        
        if cache_key in self.compiled_results:
            self.cache_hits += 1
            print(f"[POML CACHE DEBUG] CACHE HIT! Found: {cache_key}")
            return self.compiled_results[cache_key]
        
        self.cache_misses += 1
        print(f"[POML CACHE DEBUG] CACHE MISS! Not found: {cache_key}")
        return None
    
    def cache_result(self, template_name: str, context: dict, result):
        """Cache a compiled result for this context combination"""
        context_key = self._generate_context_key(template_name, context)
        cache_key = f"{template_name}:{context_key}"
        
        # Debug: Show what we're caching
        print(f"[POML CACHE DEBUG] Caching result for key: {cache_key}")
        print(f"[POML CACHE DEBUG] Context: {context}")
        
        # Store the compiled result
        self.compiled_results[cache_key] = result
        
        # Clean up if cache gets too large
        if len(self.compiled_results) > self.max_cache_size:
            # Remove oldest entries (simple FIFO)
            oldest_keys = list(self.compiled_results.keys())[:20]
            for key in oldest_keys:
                del self.compiled_results[key]
        
        return True
    
    def is_template_loaded(self, template_name: str) -> bool:
        """Check if template is available for caching"""
        return template_name in self.template_hashes
    
    def get_cache_stats(self) -> dict:
        """Get cache performance statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "cached_results": len(self.compiled_results),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "max_cache_size": self.max_cache_size
        }
    
    def clear_cache(self):
        """Clear all cached results"""
        self.compiled_results.clear()
        self.cache_hits = 0
        self.cache_misses = 0

# =============================================================================
# ENVIRONMENT SETUP - OLLAMA OPTIMIZATIONS
# =============================================================================

# Set optimal Ollama environment variables for Q4_K_M models
os.environ['OLLAMA_FLASH_ATTENTION'] = '1'           # Enable flash attention
os.environ['OLLAMA_KV_CACHE_TYPE'] = 'f16'          # Full precision for Q4 models (better quality)
os.environ['OLLAMA_NUM_PARALLEL'] = '2'             # Reduced for Q4 models
os.environ['OLLAMA_MAX_LOADED_MODELS'] = '1'        # Single model for Q4 efficiency

# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

# Tool schemas for Ollama (exact from merged bot)
get_weather_tool = {
    'type': 'function',
    'function': {
        'name': 'get_weather',
        'description': 'Get current weather for a city.',
        'parameters': {
            'type': 'object',
            'properties': {
                'city': {'type': 'string', 'description': 'The city name'}
            },
            'required': ['city'],
        },
    },
}
calculate_tool = {
    'type': 'function',
    'function': {
        'name': 'calculate',
        'description': 'Safely evaluate mathematical expressions.',
        'parameters': {
            'type': 'object',
            'properties': {
                'expression': {'type': 'string', 'description': 'Mathematical expression to evaluate'}
            },
            'required': ['expression'],
        },
    },
}
web_search_tool = {
    'type': 'function',
    'function': {
        'name': 'web_search',
        'description': 'Search the web for information.',
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {'type': 'string', 'description': 'Search query'},
                'num_results': {'type': 'integer', 'description': 'Number of results (1-10)', 'default': 5}
            },
            'required': ['query'],
        },
    },
}
web_scrape_tool = {
    'type': 'function',
    'function': {
        'name': 'web_scrape',
        'description': 'Scrape content from a webpage.',
        'parameters': {
            'type': 'object',
            'properties': {
                'url': {'type': 'string', 'description': 'URL to scrape'}
            },
            'required': ['url'],
        },
    },
}
news_search_tool = {
    'type': 'function',
    'function': {
        'name': 'news_search',
        'description': 'Search for recent news articles.',
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {'type': 'string', 'description': 'News search query'},
                'num_results': {'type': 'integer', 'description': 'Number of results (1-10)', 'default': 5}
            },
            'required': ['query'],
        },
    },
}
get_time_tool = {
    'type': 'function',
    'function': {
        'name': 'get_time',
        'description': 'Get current date and time information including formatted time, date, timezone, and timestamp',
        'parameters': {
            'type': 'object',
            'properties': {},
            'required': []
        }
    }
}

analyze_user_profile_tool = {
    "type": "function",
    "function": {
        "name": "analyze_user_profile",
        "description": "Get comprehensive Discord profile analysis including activity patterns and social connections",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "Discord user ID to analyze"
                }
            },
            "required": ["user_id"]
        }
    }
}

dox_user_tool = {
    "type": "function",
    "function": {
        "name": "dox_user",
        "description": "Comprehensive Discord user profile analysis including avatar AI vision analysis, roles, activities, and personality insights",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "Discord user ID to analyze"
                }
            },
            "required": ["user_id"]
        }
    }
}

analyze_image_tool_schema = {
    "type": "function",
    "function": {
        "name": "analyze_image_tool",
        "description": "Analyze any image using AI vision to identify content, style, mood, colors, and context",
        "parameters": {
            "type": "object",
            "properties": {
                "image_url": {
                    "type": "string",
                    "description": "URL of the image to analyze"
                },
                "filename": {
                    "type": "string",
                    "description": "Optional filename for the image",
                    "default": "uploaded_image"
                }
            },
            "required": ["image_url"]
        }
    }
}

discord_action_tool = {
    "type": "function",
    "function": {
        "name": "discord_action",
        "description": "Perform Discord server management and moderation tasks. Use this tool for: sending messages to channels/DMs, getting user/channel/server info, listing channels, banning/kicking/timing out users, and any Discord-specific actions. Examples: 'send a message to #general', 'DM @user hello', 'ban @user', 'kick @spammer', 'timeout @user', 'get info about @user'",
        "parameters": {
            "type": "object",
            "properties": {
                "action_type": {
                    "type": "string",
                    "description": "Type of action to perform",
                    "enum": ["send_message", "send_dm", "get_user_info", "get_channel_info", "get_guild_info", "list_channels", "ban_user", "kick_user", "timeout_user", "react_to_message", "get_message", "list_online", "send_embed", "channel_history", "search_messages", "server_emojis", "check_status", "avatar_full"]
                },
                "target_id": {
                    "type": "string",
                    "description": "Target ID (user_id for get_user_info, guild_id for get_guild_info/list_channels)"
                },
                "message": {
                    "type": "string",
                    "description": "Message content for send_message action"
                },
                "channel_id": {
                    "type": "string",
                    "description": "Channel ID for send_message or get_channel_info actions"
                },
                "role_name": {
                    "type": "string",
                    "description": "Role name for role-related actions"
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for moderation actions"
                },
                "duration_minutes": {
                    "type": "integer",
                    "description": "Duration in minutes for timeout_user action (default: 10)",
                    "default": 10
                },
                "message_id": {
                    "type": "string",
                    "description": "Message ID for react_to_message or get_message actions"
                },
                "emoji": {
                    "type": "string",
                    "description": "Emoji for react_to_message action (unicode or custom emoji)"
                },
                "embed_title": {
                    "type": "string",
                    "description": "Title for send_embed action"
                },
                "embed_description": {
                    "type": "string",
                    "description": "Description for send_embed action"
                },
                "embed_color": {
                    "type": "string",
                    "description": "Hex color for send_embed action (e.g. #ff0000)"
                },
                "search_query": {
                    "type": "string",
                    "description": "Search query for search_messages action"
                },
                "limit": {
                    "type": "integer",
                    "description": "Number limit for channel_history or search results (default: 10)",
                    "default": 10
                }
            },
            "required": ["action_type"]
        }
    }
}

# =============================================================================
# TOOL EXECUTION FUNCTIONS
# =============================================================================

async def execute_web_search(query: str, num_results: int = 5) -> dict:
    """Execute web search using Serper API"""
    try:
        serper_api_key = os.getenv('SERPER_API_KEY')
        if not serper_api_key:
            return {"error": "SERPER_API_KEY not configured"}

        url = "https://google.serper.dev/search"
        headers = {
            'X-API-KEY': serper_api_key,
            'Content-Type': 'application/json'
        }

        payload = {
            'q': query,
            'num': min(num_results, 10)
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []

                    for item in data.get('organic', []):
                        results.append({
                            'title': item.get('title', ''),
                            'link': item.get('link', ''),
                            'snippet': item.get('snippet', '')
                        })

                    return {'query': query, 'results': results}
                else:
                    return {"error": f"Search API error: {response.status}"}

    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}

async def execute_web_scrape(url: str, max_length: int = 2000) -> dict:
    """Scrape content from URL"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.text()

                    # Basic content extraction
                    import re
                    clean_content = re.sub(r'<[^>]+>', '', content)
                    clean_content = ' '.join(clean_content.split())

                    if len(clean_content) > max_length:
                        clean_content = clean_content[:max_length] + "..."

                    return {'url': url, 'content': clean_content}
                else:
                    return {"error": f"HTTP {response.status}"}

    except Exception as e:
        return {"error": f"Scraping failed: {str(e)}"}

async def execute_calculate(expression: str) -> dict:
    """Safely evaluate mathematical expressions"""
    try:
        # Safe evaluation - only allow basic math
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return {"error": "Invalid characters in expression"}

        result = eval(expression)
        return {'expression': expression, 'result': result}

    except Exception as e:
        return {"error": f"Calculation failed: {str(e)}"}

async def execute_get_time() -> dict:
    """Get current date and time"""
    now = datetime.now()
    return {
        'datetime': now.isoformat(),
        'formatted': now.strftime("%Y-%m-%d %H:%M:%S"),
        'timezone': str(now.astimezone().tzinfo)
    }

# Tool execution mapping
TOOL_FUNCTIONS = {
    'web_search': execute_web_search,
    'web_scrape': execute_web_scrape,
    'calculate': execute_calculate,
    'get_time': execute_get_time
}

# All tools list for Ollama
ALL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {"type": "integer", "description": "Number of results (1-10)", "default": 5}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_scrape",
            "description": "Scrape content from a specific URL",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to scrape"},
                    "max_length": {"type": "integer", "description": "Max content length", "default": 2000}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get current date and time",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]

# =============================================================================
# PERFORMANCE MONITORING (FROM MERGED BOT)
# =============================================================================

try:
    import psutil
    import GPUtil
    import threading
    from ollama import AsyncClient
    PERFORMANCE_MONITORING = True
except ImportError:
    PERFORMANCE_MONITORING = False
    print("[WARNING] Performance monitoring disabled - install: pip install psutil gputil ollama")

class PerformanceMonitor:
    """Monitor system performance and adjust Ollama settings dynamically"""

    def __init__(self):
        self.gpu_available = PERFORMANCE_MONITORING and GPUtil and len(GPUtil.getGPUs()) > 0
        self.last_check = time.time()
        self.performance_history = []

    def get_system_stats(self) -> Dict:
        """Get current system performance stats"""
        if not PERFORMANCE_MONITORING:
            return {'cpu_percent': 50, 'memory_percent': 50, 'timestamp': time.time()}

        stats = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'timestamp': time.time()
        }

        if self.gpu_available:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    stats.update({
                        'gpu_load': gpu.load * 100,
                        'gpu_memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                        'gpu_temperature': gpu.temperature
                    })
            except Exception as e:
                print(f"GPU monitoring error: {e}")

        return stats

    def should_adjust_settings(self) -> Dict[str, Any]:
        """Determine if Ollama settings should be adjusted based on load"""
        stats = self.get_system_stats()
        self.performance_history.append(stats)

        # Keep only last 10 readings
        if len(self.performance_history) > 10:
            self.performance_history.pop(0)

        adjustments = {}

        # If GPU memory is high, reduce parallel requests
        if stats.get('gpu_memory_percent', 0) > 85:
            adjustments['reduce_parallel'] = True

        # If CPU is high, increase keep_alive to avoid reloading
        if stats.get('cpu_percent', 0) > 80:
            adjustments['increase_keep_alive'] = True

        # If system is idle, we can be more aggressive
        if (stats.get('cpu_percent', 0) < 30 and
            stats.get('gpu_memory_percent', 0) < 50):
            adjustments['increase_parallel'] = True

        return adjustments

# =============================================================================
# BLAZING FAST OLLAMA CLIENT (FROM MERGED BOT)
# =============================================================================

class OptimizedOllamaClient:
    """Enhanced Ollama client with performance optimizations - BLAZING FAST VERSION"""

    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.performance_monitor = PerformanceMonitor()
        self.model_cache = {}  # Track loaded models
        self.active_requests = 0
        self.max_concurrent = 4
        self.session = None  # Add persistent session attribute

        if PERFORMANCE_MONITORING:
            self.async_client = AsyncClient(host=base_url)
            # Start performance monitoring
            self.monitor_thread = threading.Thread(target=self._monitor_performance, daemon=True)
            self.monitor_thread.start()
        else:
            self.async_client = None

    def _monitor_performance(self):
        """Background performance monitoring and adjustment"""
        while True:
            try:
                adjustments = self.performance_monitor.should_adjust_settings()

                if adjustments.get('reduce_parallel'):
                    self.max_concurrent = max(1, self.max_concurrent - 1)
                    print(f"[INFO] Reduced concurrent requests to {self.max_concurrent}")

                elif adjustments.get('increase_parallel'):
                    self.max_concurrent = min(8, self.max_concurrent + 1)
                    print(f"🔺 Increased concurrent requests to {self.max_concurrent}")

                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                print(f"Performance monitor error: {e}")
                time.sleep(60)

    async def chat(self, model: str, messages: List[Dict], tools: List[Dict] = None, **kwargs):
        """BLAZING FAST optimized chat with performance monitoring"""

        # Wait for available slot
        while self.active_requests >= self.max_concurrent:
            await asyncio.sleep(0.1)

        self.active_requests += 1
        start_time = time.time()

        try:
            # Add performance options for Q4 models (YOUR FAST SETTINGS)
            options = kwargs.get('options', {})

            # Optimize for Q4 models - BLAZING FAST SETTINGS
            options.update({
                'num_ctx': 8192,  # Large context for Discord conversations
                'temperature': 0.7,
                'top_p': 0.9,
                'repeat_penalty': 1.1,
                'num_predict': 512,  # Reasonable response length
                'num_batch': 128,    # Batch size for Q4 models
                'num_thread': min(8, psutil.cpu_count() if PERFORMANCE_MONITORING else 8),
                'num_gpu_layers': -1 if self.performance_monitor.gpu_available else 0
            })

            kwargs['options'] = options

            # Format messages with proper BPE tags
            formatted_messages = self.format_messages_for_bpe(messages)

            if self.async_client:
                # Use AsyncClient for maximum speed
                chat_params = {
                    'model': model,
                    'messages': formatted_messages,
                    'stream': False,
                    **kwargs
                }

                # Add tools if provided
                if tools:
                    chat_params['tools'] = tools

                response = await self.async_client.chat(**chat_params)
            else:
                # Fallback to aiohttp
                response = await self._fallback_chat(model, formatted_messages, tools, **kwargs)

            # Track performance
            end_time = time.time()
            response_time = end_time - start_time

            # Log slow responses
            if response_time > 10:
                print(f"[WARNING] Slow response: {response_time:.2f}s for {model}")
            else:
                print(f"[INFO] Fast response: {response_time:.2f}s")

            return response

        except Exception as e:
            print(f"Ollama request error: {e}")
            raise
        finally:
            self.active_requests -= 1

    async def _fallback_chat(self, model: str, messages: List[Dict], tools: List[Dict] = None, **kwargs):
        """Fallback aiohttp method"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                **kwargs
            }

            if tools:
                payload["tools"] = tools

            async with session.post(f"{self.base_url}/api/chat", json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Ollama error: {response.status}")

    async def create_session(self):
        """Create persistent aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def close_session(self):
        """Close persistent aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

    def format_messages_for_bpe(self, messages: List[Dict]) -> List[Dict]:
        """Format messages with proper BPE tags for optimal tokenization"""
        formatted = []

        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')

            if role == 'tool':
                # Wrap tool responses in proper BPE tags
                formatted.append({
                    'role': 'tool',
                    'content': f"<tool_response>\n{content}\n</tool_response>",
                    'name': msg.get('name', 'unknown_tool')
                })
            else:
                formatted.append(msg)

        return formatted

# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================

async def web_search(query: str, num_results: int = 5) -> dict:
    """Search the web using Serper API."""
    print(f"[TOOL] web_search called with query: {query}, num_results: {num_results}")
    try:
        serper_api_key = os.getenv('SERPER_API_KEY')
        if not serper_api_key:
            return {"error": "SERPER_API_KEY not configured"}
        
        headers = {
            "X-API-KEY": serper_api_key,
            "Content-Type": "application/json"
        }
        payload = json.dumps({"q": query, "num": num_results})
        
        async with aiohttp.ClientSession() as session:
            async with session.post("https://google.serper.dev/search", 
                                  headers=headers, data=payload, timeout=30) as response:
                if response.status != 200:
                    return {"error": f"Search failed with status {response.status}"}
                
                data = await response.json()
                items = []
                for it in data.get('organic', [])[:num_results]:
                    items.append({
                        "title": it.get('title', 'N/A'),
                        "snippet": it.get('snippet', 'N/A'),
                        "link": it.get('link', 'N/A')
                    })
                result_dict = {"results": items, "query": query}
                print(f"[TOOL] web_search success: Found {len(items)} results")
                return result_dict
    except Exception as e:
        error_msg = f"Search error: {str(e)}"
        print(f"[TOOL] web_search error: {error_msg}")
        return {"error": error_msg}

async def web_scrape(url: str) -> dict:
    """Scrape a webpage using Serper API."""
    print(f"[TOOL] web_scrape called with url: {url}")
    try:
        serper_api_key = os.getenv('SERPER_API_KEY')
        if not serper_api_key:
            return {"error": "SERPER_API_KEY not configured"}
        
        headers = {
            "X-API-KEY": serper_api_key,
            "Content-Type": "application/json"
        }
        payload = json.dumps({"url": url})
        
        async with aiohttp.ClientSession() as session:
            async with session.post("https://scrape.serper.dev", 
                                  headers=headers, data=payload, timeout=45) as response:
                if response.status != 200:
                    return {"error": f"Scrape failed with status {response.status}"}
                
                data = await response.json()
                result_dict = {
                    "url": url,
                    "title": data.get('title', 'N/A'),
                    "content": data.get('text', 'No content found')[:2000]  # Limit content
                }
                print(f"[TOOL] web_scrape success: Scraped {len(result_dict['content'])} characters")
                return result_dict
    except Exception as e:
        error_msg = f"Scrape error: {str(e)}"
        print(f"[TOOL] web_scrape error: {error_msg}")
        return {"error": error_msg}

async def calculate(expression: str) -> dict:
    """Safely evaluate math expression."""
    print(f"[TOOL] calculate called with expression: {expression}")
    try:
        expr = str(expression).replace('^', '**').replace('×', '*').replace('÷', '/')
        if not re.fullmatch(r"[0-9\.\+\-\*/\(\) ,]+", expr):
            return {"error": "Invalid characters in expression"}
        result = eval(expr)
        result_dict = {"result": str(result), "expression": expr}
        print(f"[TOOL] calculate success: {result_dict}")
        return result_dict
    except Exception as e:
        error_msg = f"Calculation error: {str(e)}"
        print(f"[TOOL] calculate error: {error_msg}")
        return {"error": error_msg}

async def get_time() -> dict:
    """Get current date and time."""
    print(f"[TOOL] get_time called")
    try:
        import datetime
        now = datetime.datetime.now()
        
        result_dict = {
            "current_time": now.strftime("%I:%M:%S %p"),
            "current_date": now.strftime("%A, %B %d, %Y"),
            "timezone": "Local Time",
            "iso_format": now.isoformat(),
            "unix_timestamp": int(now.timestamp())
        }
        print(f"[TOOL] get_time success: {result_dict['current_time']}")
        return result_dict
    except Exception as e:
        error_msg = f"Time retrieval error: {str(e)}"
        print(f"[TOOL] get_time error: {error_msg}")
        return {"error": error_msg}

async def get_weather(city: str) -> dict:
    """Get current weather for a city using free weather API with improved geocoding."""
    print(f"[TOOL] get_weather called with city: {city}")
    try:
        async with aiohttp.ClientSession() as session:
            # Try Open-Meteo's geocoding API first (more reliable)
            geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
            
            try:
                async with session.get(geocode_url, timeout=15) as geocode_response:
                    if geocode_response.status == 200:
                        geocode_data = await geocode_response.json()
                        results = geocode_data.get('results', [])
                        
                        if results:
                            location = results[0]
                            latitude = location.get('latitude')
                            longitude = location.get('longitude')
                            city_name = location.get('name', city)
                            
                            if latitude is not None and longitude is not None:
                                # Get weather data
                                weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true&temperature_unit=celsius"
                                async with session.get(weather_url, timeout=15) as weather_response:
                                    if weather_response.status != 200:
                                        return {"error": f"Weather API failed with status {weather_response.status}"}
                                    
                                    weather_data = await weather_response.json()
                                    current = weather_data.get('current_weather', {})
                                    
                                    # Convert weather code to description
                                    weather_code = current.get('weathercode', 0)
                                    weather_descriptions = {
                                        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
                                        45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
                                        56: "Light freezing drizzle", 57: "Dense freezing drizzle", 61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
                                        66: "Light freezing rain", 67: "Heavy freezing rain", 71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
                                        77: "Snow grains", 80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
                                        85: "Slight snow showers", 86: "Heavy snow showers", 95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail"
                                    }
                                    
                                    result = {
                                        "city": city_name,
                                        "temperature": f"{current.get('temperature', 'N/A')}°C",
                                        "condition": weather_descriptions.get(weather_code, f"Weather code {weather_code}"),
                                        "wind_speed": f"{current.get('windspeed', 'N/A')} km/h",
                                        "wind_direction": f"{current.get('winddirection', 'N/A')}°"
                                    }
                                    print(f"[TOOL] get_weather success: {result}")
                                    return result
            except Exception as e:
                print(f"[WARNING] Open-Meteo geocoding failed: {e}")
        
        return {"error": f"Could not get weather for '{city}'"}
    except Exception as e:
        error_msg = f"Weather service error: {str(e)}"
        print(f"[TOOL] get_weather error: {error_msg}")
        return {"error": error_msg}

async def news_search(query: str, num_results: int = 5) -> dict:
    """Search for recent news using dedicated Serper News API."""
    print(f"[TOOL] news_search called with query: {query}, num_results: {num_results}")
    try:
        serper_api_key = os.getenv('SERPER_API_KEY')
        if not serper_api_key:
            return {"error": "SERPER_API_KEY not configured"}
        
        headers = {
            "X-API-KEY": serper_api_key,
            "Content-Type": "application/json"
        }
        payload = json.dumps({"q": query, "num": num_results})
        
        async with aiohttp.ClientSession() as session:
            async with session.post("https://google.serper.dev/news", 
                                  headers=headers, data=payload, timeout=30) as response:
                if response.status != 200:
                    return {"error": f"News search failed with status {response.status}"}
                
                data = await response.json()
                items = []
                for it in data.get('news', [])[:num_results]:
                    items.append({
                        "title": it.get('title', 'N/A'),
                        "snippet": it.get('snippet', 'N/A'),
                        "link": it.get('link', 'N/A'),
                        "date": it.get('date', 'Unknown')
                    })
                result_dict = {"results": items, "query": query}
                print(f"[TOOL] news_search success: Found {len(items)} news results")
                return result_dict
                
    except Exception as e:
        error_msg = f"News search error: {str(e)}"
        print(f"[TOOL] news_search error: {error_msg}")
        return {"error": error_msg}

async def analyze_user_profile(user_id: str, bot_instance=None) -> dict:
    """Comprehensive Discord user profile analysis with avatar vision analysis."""
    try:
        if not bot_instance:
            return {"error": "Bot instance not available"}

        # Convert user_id to int and get user
        try:
            user_id_int = int(user_id)
            user = bot_instance.get_user(user_id_int)

            if not user:
                user = await bot_instance.fetch_user(user_id_int)
        except (ValueError, discord.NotFound):
            return {"error": f"User with ID '{user_id}' not found"}
        except discord.HTTPException:
            return {"error": "Discord API error while fetching user"}

        # Basic profile data
        profile_data = {
            "user_id": str(user.id),
            "username": user.name,
            "display_name": user.display_name,
            "discriminator": user.discriminator,
            "created_at": user.created_at.isoformat(),
            "is_bot": user.bot,
            "avatar_url": str(user.avatar.url) if user.avatar else None
        }

        # Try to get guild-specific data if user is in a mutual guild
        for guild in bot_instance.guilds:
            member = guild.get_member(user.id)
            if member:
                guild_data = {
                    "guild_name": guild.name,
                    "guild_id": str(guild.id),
                    "joined_at": member.joined_at.isoformat() if member.joined_at else None,
                    "roles": [{"name": role.name, "id": str(role.id)} for role in member.roles if role.name != "@everyone"],
                    "nickname": member.nick,
                    "premium_since": member.premium_since.isoformat() if member.premium_since else None
                }
                profile_data["guild_data"] = guild_data
                break  # Use first mutual guild found

        # Avatar analysis using vision model (if avatar exists)
        if user.avatar:
            try:
                print(f"🔍 Starting avatar analysis for {user.name}...")
                # Get optimized client from bot instance
                ollama_client = getattr(bot_instance, 'ollama', None)
                # Let vision model take however long it needs
                avatar_analysis = await analyze_avatar_with_vision(str(user.avatar.url), ollama_client, bot_instance)
                if not avatar_analysis.get('error'):
                    print(f"✅ Avatar analysis successful for {user.name}")
                    profile_data["avatar_analysis"] = avatar_analysis
                else:
                    print(f"⚠️ Avatar analysis returned error: {avatar_analysis.get('error')}")
                    profile_data["avatar_analysis"] = avatar_analysis
            except Exception as e:
                print(f"❌ Avatar analysis failed for {user.name}: {e}")
                profile_data["avatar_analysis"] = {"error": f"Avatar analysis failed: {str(e)}"}

        return profile_data

    except Exception as e:
        return {"error": f"Profile analysis error: {str(e)}"}

async def dox_user(user_id: str, bot_instance=None) -> str:
    """Tool function to analyze Discord user profile and avatar (requires bot instance)."""
    try:
        if not bot_instance:
            return "Error: Bot instance is not available for this command."

        # Convert user_id to int and get user
        try:
            user_id_int = int(user_id)
            user = bot_instance.get_user(user_id_int)

            if not user:
                user = await bot_instance.fetch_user(user_id_int)
        except (ValueError, discord.NotFound):
            return f"I couldn't find a user with the ID '{user_id}'."
        except discord.HTTPException:
            return "I had trouble connecting to Discord to find that user."

        # Gather comprehensive profile data
        profile_data = await analyze_user_profile(str(user.id), bot_instance)

        if 'error' in profile_data:
            return f"I couldn't get information about that user. Error: {profile_data['error']}"

        # Format comprehensive response
        username = profile_data.get('username', 'this user')
        display_name = profile_data.get('display_name')
        full_username = f"{profile_data.get('username')}#{profile_data.get('discriminator', '0')}"

        response_parts = []
        if display_name and display_name.lower() != username.lower():
            response_parts.append(f"Here's what I found about {display_name} (username: {full_username}).")
        else:
            response_parts.append(f"Here's what I found about {full_username}.")

        created_at = profile_data.get('created_at')
        if created_at:
            created_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            response_parts.append(f"Their account was created on {created_dt.strftime('%B %d, %Y')}.")

        guild_data = profile_data.get('guild_data')
        if guild_data and isinstance(guild_data, dict) and guild_data.get('joined_at'):
            joined_at = guild_data['joined_at']
            joined_dt = datetime.fromisoformat(joined_at.replace('Z', '+00:00'))
            response_parts.append(f"They joined the '{guild_data.get('guild_name', 'current')}' server on {joined_dt.strftime('%B %d, %Y')}.")

        if profile_data.get('is_bot'):
            response_parts.append("This user is a bot.")

        if 'avatar_analysis' in profile_data and isinstance(profile_data['avatar_analysis'], dict) and 'error' not in profile_data['avatar_analysis']:
            avatar_analysis = profile_data['avatar_analysis']
            subject = avatar_analysis.get('subject', 'an interesting image')
            mood = avatar_analysis.get('mood_theme', 'a unique vibe')
            response_parts.append(f"Their avatar shows: '{subject}' with a {mood} feeling.")

        return " ".join(response_parts)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"A truly unexpected error occurred while analyzing the user profile: {str(e)}"

async def analyze_avatar_with_vision(avatar_url: str, ollama_client = None, bot_instance = None) -> dict:
    """Analyze Discord avatar using configurable vision model."""
    try:
        # Get vision model from bot instance or use default
        vision_model = 'granite3.2-vision:2b'  # Default fallback
        if bot_instance and hasattr(bot_instance, 'vision_model'):
            vision_model = bot_instance.vision_model

        print(f"🔍 Starting avatar analysis with {vision_model}...")

        # Enhanced avatar analysis prompt
        analysis_prompt = """Analyze this Discord avatar image and provide insights about the user's personality and preferences.

Focus on these key areas:
1. SUBJECT/CONTENT: What's depicted (character, person, art, etc.)
2. ART STYLE: Visual style, artistic approach, or aesthetic choice
3. MOOD/PERSONALITY: What personality traits or mood this might suggest
4. COLOR PSYCHOLOGY: Dominant colors and their psychological implications
5. CULTURAL REFERENCES: Any recognizable characters, symbols, or cultural elements
6. INTERACTION STYLE: How this person might prefer to interact based on their avatar choice

IMPORTANT: Your entire response MUST be a single, valid JSON object.

Respond in valid JSON format:
{
  "subject": "detailed description of what's shown in the avatar",
  "art_style": "visual style, technique, or aesthetic approach",
  "mood_theme": "overall emotional tone and atmosphere",
  "personality_indicators": ["trait1", "trait2", "trait3"],
  "color_analysis": "dominant colors and their psychological meaning",
  "cultural_elements": "any recognizable references or symbols",
  "interaction_suggestions": "how this person might prefer to be approached",
  "confidence_level": "high/medium/low based on image clarity"
}"""

        # Download the avatar image with simple reliable networking
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30, connect=10)
        ) as session:
            try:
                # Add headers to mimic browser request
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Cache-Control': 'no-cache',
                    'Pragma': 'no-cache'
                }

                async with session.get(avatar_url, headers=headers) as response:
                    if response.status == 404:
                        return {"error": f"Avatar not found (404) - Discord URL may be expired: {avatar_url}"}
                    elif response.status == 403:
                        return {"error": f"Access denied (403) - Avatar may be private: {avatar_url}"}
                    elif response.status != 200:
                        return {"error": f"Failed to download avatar: HTTP {response.status}"}

                    image_data = await response.read()
                    if len(image_data) == 0:
                        return {"error": "Downloaded avatar is empty"}

            except aiohttp.ClientConnectorError as e:
                return {"error": f"Connection failed - check internet connection: {str(e)}"}
            except aiohttp.ServerTimeoutError:
                return {"error": "Request timed out - Discord CDN may be slow"}
            except Exception as e:
                return {"error": f"Error downloading avatar: {str(e)}"}

        # Use official Ollama AsyncClient for vision analysis (based on examples)
        if not ollama_client:
            # Create official Ollama client as fallback
            from ollama import AsyncClient
            ollama_client = AsyncClient()

        try:
            # Prepare messages for vision analysis  
            vision_messages = [{
                'role': 'user',
                'content': analysis_prompt,
                'images': [image_data]
            }]
            
            # Use official Ollama chat API for vision analysis (like examples/multimodal-chat.py)
            response = await ollama_client.chat(
                model=vision_model,
                messages=vision_messages,
                options={
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 800
                }
            )
        except Exception as e:
            return {"error": f"Vision model error: {str(e)}"}

        try:
            # Parse the JSON response (official Ollama chat format)
            response_text = response['message']['content'].strip()

            # Clean up common formatting issues
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            analysis_result = json.loads(response_text)

            # Add metadata
            analysis_result['analyzed_image'] = avatar_url
            analysis_result['model_used'] = vision_model
            analysis_result['analysis_timestamp'] = datetime.now().isoformat()

            return analysis_result
        except json.JSONDecodeError as json_error:
            # Try to extract JSON from potentially malformed response
            response_text = response['message']['content']

            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    analysis_result = json.loads(json_match.group())
                    analysis_result['analyzed_image'] = avatar_url
                    analysis_result['model_used'] = vision_model
                    analysis_result['analysis_timestamp'] = datetime.now().isoformat()
                    return analysis_result
                except json.JSONDecodeError:
                    pass

            return {
                "error": f"Failed to parse JSON response: {str(json_error)}",
                "raw_response": response_text[:500] + "..." if len(response_text) > 500 else response_text,
                "analyzed_image": avatar_url
            }

    except Exception as e:
        print(f"❌ Avatar vision analysis error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "error": f"Avatar analysis error: {str(e)}",
            "analyzed_image": avatar_url,
            "error_type": type(e).__name__
        }

async def analyze_image_tool(image_url: str, filename: str = "uploaded_image", bot_instance=None) -> dict:
    """Analyze images using AI vision - REAL IMPLEMENTATION FROM MERGED BOT."""
    try:
        # Get vision model from bot instance or use default
        vision_model = 'granite3.2-vision:2b'  # Default fallback
        if bot_instance and hasattr(bot_instance, 'vision_model'):
            vision_model = bot_instance.vision_model
        print(f"🔍 Starting image analysis for: {filename or image_url[:50]}...")

        # Create enhanced analysis prompt for general images
        analysis_prompt = """Analyze this image in detail and provide comprehensive insights about its content, style, and context.

Focus on these key areas:
1. SUBJECT/CONTENT: Detailed description of what's depicted (people, objects, scenes, text, etc.)
2. VISUAL STYLE: Art style, photographic technique, or artistic approach
3. MOOD/ATMOSPHERE: Overall emotional tone, lighting, and feeling conveyed
4. COLOR ANALYSIS: Dominant colors, color schemes, and their impact
5. COMPOSITION: Layout, framing, and visual elements
6. CONTEXT/PURPOSE: Likely use case, setting, or intended message
7. TECHNICAL ASPECTS: Quality, resolution indicators, and visual characteristics
8. CULTURAL/SYMBOLIC ELEMENTS: Any recognizable symbols, references, or cultural significance

Provide specific, detailed observations that would be useful for understanding this image.

IMPORTANT: Your entire response MUST be a single, valid JSON object. Do not include any other text, explanations, or markdown formatting like ```json before or after the JSON object.

Respond in valid JSON format:
{
  "subject": "detailed description of the main content and elements",
  "art_style": "visual style, technique, or photographic approach",
  "mood_theme": "emotional atmosphere, lighting, and overall feeling",
  "color_analysis": "dominant colors, schemes, and psychological impact",
  "composition_notes": "layout, framing, and visual structure observations",
  "context_suggestions": "likely purpose, setting, intended use, or meaning",
  "technical_quality": "visual quality, clarity, and technical aspects",
  "cultural_elements": "any symbols, references, or cultural significance",
  "confidence_level": "high/medium/low based on image clarity and analysis certainty"
}"""

        # Download the image first with simple reliable networking
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30, connect=10)
        ) as session:
            try:
                # Add headers to mimic browser request
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Cache-Control': 'no-cache',
                    'Pragma': 'no-cache'
                }

                async with session.get(image_url, headers=headers) as response:
                    if response.status == 404:
                        return {"error": f"Image not found (404) - Discord URL may be expired: {image_url}"}
                    elif response.status == 403:
                        return {"error": f"Access denied (403) - Discord image may be private: {image_url}"}
                    elif response.status != 200:
                        return {"error": f"Failed to download image: HTTP {response.status} from {image_url}"}

                    image_data = await response.read()
                    if len(image_data) == 0:
                        return {"error": "Downloaded image is empty"}

            except aiohttp.ClientConnectorError as e:
                return {"error": f"Connection failed - check internet connection or firewall settings: {str(e)}"}
            except aiohttp.ServerTimeoutError:
                return {"error": "Request timed out - Discord CDN may be slow or unreachable"}
            except aiohttp.ClientError as e:
                return {"error": f"Network error downloading image: {str(e)}"}
            except Exception as e:
                return {"error": f"Unexpected error downloading image: {str(e)}"}

        # Use official Ollama AsyncClient for vision analysis (based on examples)
        ollama_client = None
        if bot_instance and hasattr(bot_instance, 'ollama'):
            # Try to use bot's optimized client first, but for vision we need official API
            pass

        # Always use official Ollama client for vision analysis
        from ollama import AsyncClient
        ollama_client = AsyncClient()

        try:
            # Prepare messages for avatar analysis
            avatar_messages = [{
                'role': 'user',
                'content': analysis_prompt,
                'images': [image_data]
            }]
            
            # Use official Ollama chat API for vision analysis (like examples/multimodal-chat.py)
            response = await ollama_client.chat(
                model=vision_model,
                messages=avatar_messages,
                options={
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 1000
                }
            )
        except Exception as e:
            return {"error": f"Vision model error: {str(e)}"}

        print(f"🤖 Vision model response received")

        try:
            # Parse the JSON response (official Ollama chat format)
            response_text = response['message']['content'].strip()

            # Clean up common formatting issues
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            analysis_result = json.loads(response_text)

            # Add metadata
            analysis_result['analyzed_image'] = image_url
            analysis_result['filename'] = filename
            analysis_result['model_used'] = vision_model
            analysis_result['analysis_timestamp'] = datetime.now().isoformat()

            print(f"✅ Image analysis parsed successfully")
            return analysis_result
        except json.JSONDecodeError as json_error:
            print(f"⚠️ JSON parse failed, trying to extract JSON from response...")
            # Try to extract JSON from potentially malformed response
            response_text = response['message']['content']

            # Look for JSON-like content between braces
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    analysis_result = json.loads(json_match.group())
                    analysis_result['analyzed_image'] = image_url
                    analysis_result['filename'] = filename
                    analysis_result['model_used'] = vision_model
                    analysis_result['analysis_timestamp'] = datetime.now().isoformat()
                    analysis_result['raw_response'] = response_text  # Include for debugging
                    print(f"✅ Image analysis extracted from malformed JSON")
                    return analysis_result
                except json.JSONDecodeError:
                    pass

            # If all else fails, return structured error with raw response
            return {
                "error": f"Failed to parse JSON response: {str(json_error)}",
                "raw_response": response_text[:500] + "..." if len(response_text) > 500 else response_text,
                "analyzed_image": image_url,
                "filename": filename
            }

    except Exception as e:
        print(f"❌ Image vision analysis error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "error": f"Image analysis error: {str(e)}",
            "analyzed_image": image_url,
            "filename": filename,
            "error_type": type(e).__name__
        }

async def discord_action(action_type: str, target_id: str = None, message: str = None, 
                        channel_id: str = None, role_name: str = None, reason: str = None,
                        duration_minutes: int = 10, message_id: str = None, emoji: str = None,
                        embed_title: str = None, embed_description: str = None, embed_color: str = None,
                        search_query: str = None, limit: int = 10, bot_instance=None, **kwargs) -> dict:
    """Universal Discord action function."""
    if not bot_instance:
        return {"error": "Bot instance not available"}
    
    result = {"action": action_type, "success": False}
    
    try:
        if action_type == "send_message":
            if not channel_id or not message:
                return {"error": "send_message requires channel_id and message"}
            
            channel = bot_instance.get_channel(int(channel_id))
            if not channel:
                return {"error": f"Channel {channel_id} not found"}
            
            await channel.send(message)
            channel_name = getattr(channel, 'name', 'DM')
            result.update({"success": True, "channel_name": channel_name})
            
        elif action_type == "send_dm":
            if not target_id or not message:
                return {"error": "send_dm requires target_id (user_id) and message"}
            
            try:
                user = await bot_instance.fetch_user(int(target_id))
                dm_channel = await user.create_dm()
                
                # Clean the message: remove bot self-mentions since they're pointless in DMs
                cleaned_message = message
                if bot_instance.user:
                    bot_mention = f"<@{bot_instance.user.id}>"
                    bot_mention_nick = f"<@!{bot_instance.user.id}>"
                    cleaned_message = cleaned_message.replace(bot_mention, "").replace(bot_mention_nick, "")
                    # Clean up any double spaces
                    cleaned_message = " ".join(cleaned_message.split())
                
                sent_message = await dm_channel.send(cleaned_message)
                
                result.update({
                    "success": True,
                    "user_name": user.name,
                    "message_id": str(sent_message.id),
                    "content": cleaned_message
                })
            except Exception as e:
                result["error"] = f"Failed to send DM: {str(e)}"
            
        elif action_type == "get_user_info":
            if not target_id:
                return {"error": "get_user_info requires target_id"}
            
            user = await bot_instance.fetch_user(int(target_id))
            result.update({
                "success": True,
                "user_name": user.name,
                "user_id": str(user.id),
                "created_at": user.created_at.isoformat()
            })
            
        elif action_type == "react_to_message":
            if not message_id or not emoji:
                return {"error": "react_to_message requires message_id and emoji"}
            
            try:
                # Find message in any channel the bot can see
                message_obj = None
                for guild in bot_instance.guilds:
                    for channel in guild.text_channels:
                        try:
                            message_obj = await channel.fetch_message(int(message_id))
                            break
                        except:
                            continue
                    if message_obj:
                        break
                
                if not message_obj:
                    return {"error": f"Message {message_id} not found"}
                
                await message_obj.add_reaction(emoji)
                result.update({
                    "success": True,
                    "message_content": message_obj.content[:100] + "..." if len(message_obj.content) > 100 else message_obj.content,
                    "emoji_added": emoji
                })
            except Exception as e:
                result["error"] = f"Failed to add reaction: {str(e)}"
        
        elif action_type == "get_message":
            if not message_id:
                return {"error": "get_message requires message_id"}
            
            try:
                message_obj = None
                for guild in bot_instance.guilds:
                    for channel in guild.text_channels:
                        try:
                            message_obj = await channel.fetch_message(int(message_id))
                            break
                        except:
                            continue
                    if message_obj:
                        break
                
                if not message_obj:
                    return {"error": f"Message {message_id} not found"}
                
                result.update({
                    "success": True,
                    "author": message_obj.author.display_name,
                    "content": message_obj.content,
                    "channel": getattr(message_obj.channel, 'name', 'DM'),
                    "timestamp": message_obj.created_at.isoformat(),
                    "reactions": [str(reaction.emoji) for reaction in message_obj.reactions]
                })
            except Exception as e:
                result["error"] = f"Failed to get message: {str(e)}"
        
        elif action_type == "list_online":
            try:
                online_members = []
                for guild in bot_instance.guilds:
                    for member in guild.members:
                        if member.status != discord.Status.offline and not member.bot:
                            online_members.append({
                                "name": member.display_name,
                                "status": str(member.status),
                                "activity": str(member.activity) if member.activity else "None"
                            })
                
                result.update({
                    "success": True,
                    "online_count": len(online_members),
                    "members": online_members[:20]  # Limit to first 20
                })
            except Exception as e:
                result["error"] = f"Failed to list online users: {str(e)}"
        
        elif action_type == "send_embed":
            if not channel_id or not embed_title:
                return {"error": "send_embed requires channel_id and embed_title"}
            
            try:
                channel = bot_instance.get_channel(int(channel_id))
                if not channel:
                    return {"error": f"Channel {channel_id} not found"}
                
                embed = discord.Embed(
                    title=embed_title,
                    description=embed_description or "",
                    color=int(embed_color.replace("#", ""), 16) if embed_color else 0x00ff00
                )
                
                sent_message = await channel.send(embed=embed)
                result.update({
                    "success": True,
                    "message_id": str(sent_message.id),
                    "channel_name": getattr(channel, 'name', 'DM')
                })
            except Exception as e:
                result["error"] = f"Failed to send embed: {str(e)}"
        
        elif action_type == "channel_history":
            if not channel_id:
                return {"error": "channel_history requires channel_id"}
            
            try:
                channel = bot_instance.get_channel(int(channel_id))
                if not channel:
                    return {"error": f"Channel {channel_id} not found"}
                
                messages = []
                async for msg in channel.history(limit=limit or 10):
                    messages.append({
                        "author": msg.author.display_name,
                        "content": msg.content[:100] + "..." if len(msg.content) > 100 else msg.content,
                        "timestamp": msg.created_at.isoformat(),
                        "message_id": str(msg.id)
                    })
                
                result.update({
                    "success": True,
                    "channel_name": getattr(channel, 'name', 'DM'),
                    "message_count": len(messages),
                    "messages": messages
                })
            except Exception as e:
                result["error"] = f"Failed to get channel history: {str(e)}"
        
        elif action_type == "search_messages":
            if not channel_id or not search_query:
                return {"error": "search_messages requires channel_id and search_query"}
            
            try:
                channel = bot_instance.get_channel(int(channel_id))
                if not channel:
                    return {"error": f"Channel {channel_id} not found"}
                
                matching_messages = []
                async for msg in channel.history(limit=200):  # Search last 200 messages
                    if search_query.lower() in msg.content.lower():
                        matching_messages.append({
                            "author": msg.author.display_name,
                            "content": msg.content,
                            "timestamp": msg.created_at.isoformat(),
                            "message_id": str(msg.id)
                        })
                        if len(matching_messages) >= (limit or 10):
                            break
                
                result.update({
                    "success": True,
                    "query": search_query,
                    "matches_found": len(matching_messages),
                    "messages": matching_messages
                })
            except Exception as e:
                result["error"] = f"Failed to search messages: {str(e)}"
        
        elif action_type == "server_emojis":
            try:
                if not target_id:
                    # Use the guild where the message was sent from (passed as guild_id context)
                    guild_id = kwargs.get('guild_id')  # We'll pass this from the calling context
                    if guild_id:
                        guild = bot_instance.get_guild(int(guild_id))
                    else:
                        # Fallback to first guild if no context provided
                        guild = bot_instance.guilds[0] if bot_instance.guilds else None
                else:
                    guild = bot_instance.get_guild(int(target_id))
                
                if not guild:
                    return {"error": "Guild not found"}
                
                emojis = []
                for emoji in guild.emojis:
                    emojis.append({
                        "name": emoji.name,
                        "id": str(emoji.id),
                        "animated": emoji.animated,
                        "url": str(emoji.url)
                    })
                
                result.update({
                    "success": True,
                    "guild_name": guild.name,
                    "emoji_count": len(emojis),
                    "emojis": emojis
                })
            except Exception as e:
                result["error"] = f"Failed to get server emojis: {str(e)}"
        
        elif action_type == "check_status":
            if not target_id:
                return {"error": "check_status requires target_id (user_id)"}
            
            try:
                user = None
                for guild in bot_instance.guilds:
                    member = guild.get_member(int(target_id))
                    if member:
                        user = member
                        break
                
                if not user:
                    return {"error": "User not found in any mutual servers"}
                
                result.update({
                    "success": True,
                    "user_name": user.display_name,
                    "status": str(user.status),
                    "activity": str(user.activity) if user.activity else "No activity",
                    "is_mobile": user.is_on_mobile(),
                    "is_desktop": user.desktop_status != discord.Status.offline,
                    "is_web": user.web_status != discord.Status.offline
                })
            except Exception as e:
                result["error"] = f"Failed to check status: {str(e)}"
        
        elif action_type == "avatar_full":
            if not target_id:
                return {"error": "avatar_full requires target_id (user_id)"}
            
            try:
                user = await bot_instance.fetch_user(int(target_id))
                result.update({
                    "success": True,
                    "user_name": user.display_name,
                    "avatar_url": str(user.avatar.url) if user.avatar else str(user.default_avatar.url),
                    "default_avatar": str(user.default_avatar.url),
                    "has_custom_avatar": user.avatar is not None
                })
            except Exception as e:
                result["error"] = f"Failed to get avatar: {str(e)}"
        
        else:
            result["error"] = f"Action type '{action_type}' not implemented yet"
        
        return result
        
    except Exception as e:
        return {"error": f"Discord action failed: {str(e)}"}

# Available functions mapping (exact from merged bot)
AVAILABLE_FUNCTIONS = {
    'get_weather': get_weather,
    'get_time': get_time,
    'calculate': calculate,
    'web_search': web_search,
    'web_scrape': web_scrape,
    'news_search': news_search,
    'analyze_user_profile': analyze_user_profile,
    'dox_user': dox_user,
    'analyze_image_tool': analyze_image_tool,
    'discord_action': discord_action
}

# All tool schemas (exact from merged bot)
ALL_TOOLS = [get_weather_tool, get_time_tool, calculate_tool, web_search_tool, web_scrape_tool, news_search_tool, analyze_user_profile_tool, dox_user_tool, analyze_image_tool_schema, discord_action_tool]

# =============================================================================
# PAGINATION VIEW FOR EMBEDS
# =============================================================================

class PaginationView(discord.ui.View):
    """Interactive pagination buttons for embeds"""
    
    def __init__(self, items: list, title: str, current_page: int, items_per_page: int, 
                 color: int, item_formatter=None, timeout: int = 180):
        super().__init__(timeout=timeout)
        self.items = items
        self.title = title
        self.current_page = current_page
        self.items_per_page = items_per_page
        self.color = color
        self.item_formatter = item_formatter
        self.total_pages = max(1, (len(items) + items_per_page - 1) // items_per_page)
        
        # Disable buttons if only one page
        if self.total_pages <= 1:
            for item in self.children:
                item.disabled = True
    
    def create_embed_for_page(self, page: int) -> discord.Embed:
        """Create embed for specific page"""
        page = max(1, min(page, self.total_pages))
        start_idx = (page - 1) * self.items_per_page
        end_idx = start_idx + self.items_per_page
        page_items = self.items[start_idx:end_idx]
        
        embed = discord.Embed(
            title=f"{self.title} (Page {page}/{self.total_pages})",
            color=self.color
        )
        
        if not page_items:
            embed.description = "No items found."
        else:
            if self.item_formatter:
                formatted_items = [self.item_formatter(item, idx + start_idx) for idx, item in enumerate(page_items)]
            else:
                formatted_items = [str(item) for item in page_items]
            embed.description = "\n".join(formatted_items)
        
        embed.set_footer(text=f"Total: {len(self.items)} items | Page {page}/{self.total_pages}")
        return embed
    
    @discord.ui.button(label='◀◀', style=discord.ButtonStyle.gray, disabled=True)
    async def first_page(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.current_page = 1
        embed = self.create_embed_for_page(self.current_page)
        self.update_buttons()
        await interaction.response.edit_message(embed=embed, view=self)
    
    @discord.ui.button(label='◀', style=discord.ButtonStyle.blurple, disabled=True)
    async def previous_page(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.current_page = max(1, self.current_page - 1)
        embed = self.create_embed_for_page(self.current_page)
        self.update_buttons()
        await interaction.response.edit_message(embed=embed, view=self)
    
    @discord.ui.button(label='▶', style=discord.ButtonStyle.blurple)
    async def next_page(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.current_page = min(self.total_pages, self.current_page + 1)
        embed = self.create_embed_for_page(self.current_page)
        self.update_buttons()
        await interaction.response.edit_message(embed=embed, view=self)
    
    @discord.ui.button(label='▶▶', style=discord.ButtonStyle.gray)
    async def last_page(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.current_page = self.total_pages
        embed = self.create_embed_for_page(self.current_page)
        self.update_buttons()
        await interaction.response.edit_message(embed=embed, view=self)
    
    def update_buttons(self):
        """Update button states based on current page"""
        self.first_page.disabled = self.current_page <= 1
        self.previous_page.disabled = self.current_page <= 1
        self.next_page.disabled = self.current_page >= self.total_pages
        self.last_page.disabled = self.current_page >= self.total_pages

# =============================================================================
# DISCORD UI COMPONENTS
# =============================================================================

# =============================================================================
# DISCORD BOT CLASS
# =============================================================================


class OptimizedDiscordBot(commands.Bot):
    """Streamlined Discord bot with all optimizations + POML support"""

    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.presences = True  # Required to see online status
        intents.members = True    # Required to list all members
        super().__init__(command_prefix='!', intents=intents)

        # Initialize components
        self.ollama = OptimizedOllamaClient()
        self.current_model = 'hf.co/subsectmusic/qwriko3-4b-instruct-2507:Q4_K_M'
        self.vision_model = 'granite3.2-vision:2b'  # Vision analysis model
        
        # Modern conversation memory management (2025)
        self.memory = ConversationMemoryManager(
            window_size=8,              # Keep 8 recent user messages
            summary_threshold=25,       # Summarize after 25 messages
            max_context_tokens=3000,    # Token limit for context
            bot_user_id=None            # Will be set in on_ready()
        )
        print("[OK] Modern Conversation Memory Manager initialized")

        # Emotional Memory System (2025) - NEW!
        try:
            from emotional_memory import EmotionalMemoryManager
            self.emotional_memory = EmotionalMemoryManager("emotional_memory")
            print("[OK] Emotional Memory System initialized")
        except ImportError as e:
            print(f"[WARNING] Emotional Memory System not available: {e}")
            self.emotional_memory = None

        # Sleep Time Agent System (2025) - NEW!
        try:
            from sleep_time_agent_core import SleepTimeAgentCore, AgentConfig
            self.sleep_agent = SleepTimeAgentCore(
                AgentConfig(
                    trigger_after_messages=100,         # Process after 100 messages
                    trigger_after_idle_minutes=1440,   # Process after 24 hours idle (1440 minutes)
                    thinking_iterations=1,              # One-pass thinking
                    model="qwen3:4b",                  # Thinking model
                    enable_faiss=True,                  # Enable FAISS vector memory
                    vector_dimension=384,               # Vector dimension
                    max_vectors_per_user=1000,         # Max vectors per user
                    similarity_threshold=0.7,           # Similarity threshold
                    stream_thinking=True,               # Stream thinking process
                    enable_tool_calling=True,           # Enable tool calling
                    enable_insights=True,               # Enable insights
                    enable_schema_tools=True            # Enable schema tools
                )
            )
            print("[OK] Sleep Time Agent System initialized")
            
            # Sleep agent background task
            self.sleep_agent_task = None
            self.user_conversation_history = {}  # Track conversations per user
            self.user_last_activity = {}         # Track last activity per user
            
        except ImportError as e:
            print(f"[WARNING] Sleep Time Agent System not available: {e}")
            self.sleep_agent = None

        # POML template management with caching
        self.poml_templates = {}
        self.poml_cache = POMLCache()  # Pre-compiled template cache
        self.mood_points = {}  # Per-user mood tracking
        self.load_poml_templates()
        
        # AI Intent Classification
        if AI_CLASSIFIER_AVAILABLE:
            try:
                self.intent_classifier = AIIntentClassifier()
                print("[OK] AI Intent Classifier initialized")
            except Exception as e:
                print(f"[ERROR] Failed to initialize AI Intent Classifier: {e}")
                self.intent_classifier = None
        else:
            self.intent_classifier = None
            print("[INFO] AI Intent Classifier disabled - using fallback mood system")
        
        # Load persistent bot state (NEW!)
        self.load_persistent_state()
        
        print("[INIT] Optimized Discord Bot initialized")
        print(f"[CONFIG] KV Cache: {os.environ.get('OLLAMA_KV_CACHE_TYPE')}")
        print(f"[CONFIG] Flash Attention: {os.environ.get('OLLAMA_FLASH_ATTENTION')}")
        
        # Debug: List all registered commands after initialization
        print(f"[DEBUG] Registered commands: {[cmd.name for cmd in self.commands]}")
    
    def load_persistent_state(self):
        """Load persistent bot state from disk"""
        try:
            state_file = "bot_persistent_state.json"
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                # Load conversation memory state
                if 'conversation_memory' in state:
                    self.memory.load_persistent_state(state['conversation_memory'])
                    print("[PERSISTENT STATE] Loaded conversation memory state")
                
                # Load emotional memory state
                if self.emotional_memory and 'emotional_memory' in state:
                    self.emotional_memory.load_persistent_state(state['emotional_memory'])
                    print("[PERSISTENT STATE] Loaded emotional memory state")
                
                # Load Discord bot tracking data
                if 'user_conversation_history' in state:
                    self.user_conversation_history = state['user_conversation_history']
                    print(f"[PERSISTENT STATE] Loaded conversation history for {len(self.user_conversation_history)} users")
                
                if 'user_last_activity' in state:
                    self.user_last_activity = state['user_last_activity']
                    print(f"[PERSISTENT STATE] Loaded activity tracking for {len(self.user_last_activity)} users")
                
                # Load mood points
                if 'mood_points' in state:
                    self.mood_points = state['mood_points']
                    print(f"[PERSISTENT STATE] Loaded mood points for {len(self.mood_points)} users")
                
                print("[PERSISTENT STATE] Successfully loaded bot state from disk")
            else:
                print("[PERSISTENT STATE] No existing state file found, starting fresh")
                
        except Exception as e:
            print(f"[PERSISTENT STATE ERROR] Failed to load state: {e}")
    
    def save_persistent_state(self):
        """Save persistent bot state to disk"""
        try:
            state = {
                'conversation_memory': self.memory.get_persistent_state(),
                'mood_points': self.mood_points,
                'timestamp': time.time()
            }
            
            # Add emotional memory state if available
            if self.emotional_memory:
                state['emotional_memory'] = self.emotional_memory.get_persistent_state()
            
            # Add Discord bot tracking data
            if hasattr(self, 'user_conversation_history'):
                state['user_conversation_history'] = self.user_conversation_history
            
            if hasattr(self, 'user_last_activity'):
                state['user_last_activity'] = self.user_last_activity
            
            # Save to disk
            with open("bot_persistent_state.json", 'w') as f:
                json.dump(state, f, indent=2)
            
            print("[PERSISTENT STATE] Successfully saved bot state to disk")
            
        except Exception as e:
            print(f"[PERSISTENT STATE ERROR] Failed to save state: {e}")
    
    async def setup_hook(self):
        """Setup hook to load commands"""
        await self.add_cog(BotCommands(self))
        print(f"[DEBUG] Commands loaded: {[cmd.name for cmd in self.commands]}")
    
    async def on_command(self, ctx):
        """Debug: Track when any command is invoked"""
        print(f"[COMMAND] Command '{ctx.command.name}' invoked by {ctx.author.name}")
    
    async def on_command_error(self, ctx, error):
        """Debug: Track command errors"""
        print(f"[COMMAND ERROR] Command '{ctx.command}' failed: {error}")
        await ctx.send(f"❌ Command error: {error}")
    
    def create_paginated_embed(self, title: str, items: list, page: int = 1, items_per_page: int = 10, 
                              color: int = 0x00ff88, item_formatter=None) -> tuple:
        """Universal pagination function for creating embeds with navigation"""
        total_items = len(items)
        total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)
        page = max(1, min(page, total_pages))
        
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        page_items = items[start_idx:end_idx]
        
        embed = discord.Embed(
            title=f"{title} (Page {page}/{total_pages})",
            color=color
        )
        
        if not page_items:
            embed.description = "No items found."
            return embed, None
        
        # Format items using custom formatter or default
        if item_formatter:
            formatted_items = [item_formatter(item, idx + start_idx) for idx, item in enumerate(page_items)]
        else:
            formatted_items = [str(item) for item in page_items]
        
        embed.description = "\n".join(formatted_items)
        embed.set_footer(text=f"Total: {total_items} items | Page {page}/{total_pages}")
        
        # Create view with navigation buttons if multiple pages
        view = None
        if total_pages > 1:
            view = PaginationView(items, title, page, items_per_page, color, item_formatter)
        
        return embed, view
    
    def load_poml_templates(self):
        """Load POML templates if available - OPTIMIZED with caching"""
        if not POML_AVAILABLE:
            return

        # Fast template loading without encoding checks
        template_files = {
            'personality': 'templates/personality_advanced_fixed.poml',
            'tools': 'templates/tools.poml',
            'mood_system': 'templates/mood_system.poml',
            'memory_context': 'templates/memory_context.poml'
        }

        for name, filepath in template_files.items():
            try:
                if os.path.exists(filepath):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        template_content = f.read()
                        self.poml_templates[name] = template_content
                        
                        # Store template hash for change detection
                        import hashlib
                        content_hash = hashlib.md5(template_content.encode()).hexdigest()
                        self.poml_cache.template_hashes[name] = content_hash
                        
                        print(f"[OK] Loaded POML template: {name}")
                else:
                    print(f"[INFO] POML template file not found: {filepath}")
            except Exception as e:
                print(f"[ERROR] Error loading POML template {name}: {e}")

    def get_user_mood(self, user_id: str) -> float:
        """Get user's current mood points (-10 to 10)"""
        return self.mood_points.get(user_id, 0.0)
    
    def sync_mood_systems(self, user_id: str) -> None:
        """Synchronize live mood (-10 to +10) with emotional memory mood system"""
        if not self.emotional_memory:
            return
            
        try:
            # Get current live mood (-10 to +10)
            live_mood = self.get_user_mood(user_id)
            
            # Get emotional memory profile
            profile = self.emotional_memory.get_user_profile(user_id)
            
            # Convert live mood (-10 to +10) to emotional memory scale (roughly -100 to +100)
            # This maintains the ratio while allowing emotional memory to accumulate over time
            target_emotional_mood = live_mood * 10.0  # -100 to +100 scale
            
            # Calculate difference and apply gradual sync (don't overwrite, blend)
            current_emotional_mood = profile.mood_points
            mood_difference = target_emotional_mood - current_emotional_mood
            
            # Apply 20% of the difference to gradually sync systems
            sync_adjustment = mood_difference * 0.2
            
            if abs(sync_adjustment) > 0.1:  # Only sync if meaningful difference
                self.emotional_memory.update_user_mood(
                    user_id=user_id,
                    mood_change=sync_adjustment,
                    reason="Mood sync with live conversation system"
                )
                print(f"\033[96m[MOOD SYNC] User {user_id}: Live={live_mood:.1f} -> Emotional={current_emotional_mood:.1f} (adj: {sync_adjustment:+.1f})\033[0m")
                
        except Exception as e:
            print(f"[MOOD SYNC ERROR] Failed to sync mood systems for user {user_id}: {e}")

    def adjust_user_mood(self, user_id: str, user_input: str) -> Tuple[float, Optional[IntentClassification]]:
        """Adjust user mood based on AI intent classification and input, returns (mood, classification)"""
        current_mood = self.get_user_mood(user_id)
        old_mood = current_mood
        mood_change = 0
        classification = None
        
        # Use AI intent classification if available
        if self.intent_classifier:
            try:
                classification = self.intent_classifier.classify_message(user_input)
                
                # Base mood adjustment on vibe and intent
                vibe_adjustments = {
                    'positive': 0.8,
                    'playful': 0.6, 
                    'flirty': 0.7,
                    'neutral': 0.0,
                    'negative': -0.8,
                    'angry': -1.2,
                    'sarcastic': -0.3,
                    'serious': 0.1
                }
                
                # Intent-based modifiers
                intent_modifiers = {
                    'compliment': 1.5,  # Multiplier for compliments
                    'complaint': -1.3,  # Multiplier for complaints
                    'request': 0.2,     # Small positive for polite requests
                    'question': 0.1,    # Neutral to slightly positive
                    'casual conversation': 0.0,  # No modifier
                    'emotional expression': 1.2  # Amplify emotional content
                }
                
                # Emotional intensity scaling
                intensity_scaling = {
                    'high': 1.4,
                    'medium': 1.0,
                    'low': 0.7
                }
                
                # Calculate mood change
                base_change = vibe_adjustments.get(classification.vibe, 0.0)
                intent_mult = intent_modifiers.get(classification.intent, 1.0)
                intensity_mult = intensity_scaling.get(classification.emotional_intensity, 1.0)
                
                mood_change = base_change * intent_mult * intensity_mult
                
                # Cap changes to reasonable ranges
                mood_change = max(-2.0, min(2.0, mood_change))
                
                current_mood = max(-10, min(10, current_mood + mood_change))
                
                print(f"\033[94m[MOOD AI] User {user_id}: {old_mood:.1f} -> {current_mood:.1f}\033[0m")
                print(f"\033[94m         Vibe: {classification.vibe}, Intent: {classification.intent}\033[0m")
                print(f"\033[94m         Intensity: {classification.emotional_intensity}, Change: {mood_change:.2f}\033[0m")
                print(f"\033[94m         Message Type: {classification.message_type}, Importance: {classification.importance_score:.2f}\033[0m")
                
            except Exception as e:
                print(f"[MOOD] AI classification failed, keeping current mood: {e}")
                # Keep current mood unchanged if AI fails
        else:
            print("[MOOD] AI Intent Classifier not available, keeping current mood")
        
        # Natural mood decay over time - slowly drift toward neutral
        if current_mood > 0:
            current_mood = max(0, current_mood - 0.1)
        elif current_mood < 0:
            current_mood = min(0, current_mood + 0.1)

        self.mood_points[user_id] = current_mood
        
        # Save mood changes immediately
        if abs(current_mood - old_mood) >= 0.1:  # Save any meaningful change
            self.save_persistent_state()
            # SYNC: Update emotional memory when live mood changes
            if hasattr(self, 'emotional_memory') and self.emotional_memory:
                self.sync_mood_systems(user_id)
        
        # Update status if mood changed significantly
        if abs(current_mood - old_mood) >= 2:
            try:
                asyncio.create_task(self.update_dynamic_status())
            except Exception as e:
                print(f"[WARNING] Failed to create status update task: {e}")
        
        return current_mood, classification

    def get_tone_from_mood(self, mood_points: float) -> str:
        """Convert mood points to tsundere tone"""
        if mood_points >= 8: return "dere-hot"      # Very flirty, openly sweet
        elif mood_points >= 5: return "cheerful"    # Flirty and warm
        elif mood_points >= 2: return "soft-dere"   # Chill and slightly flirty
        elif mood_points >= -1: return "neutral"    # Chill but sassy (default)
        elif mood_points >= -4: return "classic-tsun"  # More tsundere, flustered denials
        elif mood_points >= -7: return "grumpy-tsun"   # Sassy and snappy
        else: return "explosive-tsun"                   # Very mad/tsundere

    async def generate_status_with_ollama(self, mood_points: int, tone: str) -> str:
        """Generate creative status message using Ollama"""
        try:
            prompt = f"""Generate a short, creative Discord bot status message (max 30 characters) for Hikari-chan based on her current mood.

Current mood: {mood_points}/10 points
Current tone: {tone}

Status should reflect her personality - tsundere, playful, occasionally sassy. Examples:
- For positive mood: "vibing~", "feeling cute today", "in a good mood!"
- For neutral mood: "just chillin", "whatever...", "doing stuff"  
- For negative mood: "hmph!", "leave me alone", "not today"

Generate ONE short status (under 30 chars):"""

            messages = [{"role": "user", "content": prompt}]
            optimized_messages = self.ollama.format_messages_for_bpe(messages)
            
            response = await self.ollama.chat(
                model=self.current_model,
                messages=optimized_messages
            )
            
            status = response['message']['content'].strip()
            # Clean up any quotes and ensure it's short
            status = status.replace('"', '').replace("'", "").strip()
            if len(status) > 30:
                status = status[:27] + "..."
                
            return status
            
        except Exception as e:
            print(f"[WARNING] Failed to generate status with Ollama: {e}")
            # Fallback status based on mood
            if mood_points >= 6:
                return "feeling good~"
            elif mood_points >= 0:
                return "just chillin"
            else:
                return "hmph..."

    async def update_dynamic_status(self):
        """Update bot status with current mood"""
        try:
            # Get Hikari's current average mood (simplified approach)
            all_moods = list(self.mood_points.values())
            if all_moods:
                avg_mood = sum(all_moods) // len(all_moods)
            else:
                avg_mood = 0
            
            tone = self.get_tone_from_mood(avg_mood)
            
            # Generate creative status with Ollama
            status_text = await self.generate_status_with_ollama(avg_mood, tone)
            
            # Set Discord status
            activity = discord.Game(name=status_text)
            await self.change_presence(activity=activity)
            
            print(f"[INFO] Updated status: {status_text} (mood: {avg_mood}, tone: {tone})")
            
        except Exception as e:
            print(f"[WARNING] Failed to update status: {e}")

    async def dynamic_status_loop(self):
        """Background task to update status every 20 minutes"""
        await self.wait_until_ready()
        
        while not self.is_closed():
            try:
                await asyncio.sleep(1200)  # 20 minutes = 1200 seconds
                await self.update_dynamic_status()
            except Exception as e:
                print(f"[WARNING] Status loop error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry

    async def generate_poml_response(self, user_input: str, username: str, user_id: str, mood_points: float = None, tone: str = None) -> tuple[List[Dict], bool, bool]:
        """Generate response using POML templates - OPTIMIZED with caching"""
        if not POML_AVAILABLE or 'personality' not in self.poml_templates:
            return [], False, False

        try:
            # Use provided mood data or fall back to default
            if mood_points is None:
                mood_points = self.get_user_mood(user_id)
            if tone is None:
                tone = self.get_tone_from_mood(mood_points)

            # Get emotional profile data for POML context
            relationship_level = "stranger"
            familiarity_percent = 0
            trust_percent = 50
            conversation_count = 0
            last_interaction_time = "never"
            personality_traits = "unknown"
            
            if hasattr(self, 'emotional_memory') and self.emotional_memory:
                try:
                    profile = self.emotional_memory.get_user_profile(user_id)
                    if profile:
                        relationship_level = profile.relationship_level.replace('_', ' ').title()
                        familiarity_percent = int(profile.familiarity_level * 100)
                        trust_percent = int(profile.trust_score * 100)  
                        conversation_count = profile.conversation_count
                        
                        # Format last interaction time
                        if profile.last_interaction > 0:
                            from datetime import datetime
                            last_interaction_time = datetime.fromtimestamp(profile.last_interaction).strftime("%b %d at %H:%M")
                        
                        # Format personality traits
                        top_traits = []
                        for trait, value in profile.personality_traits.items():
                            if value > 0.6:  # Only show strong traits
                                top_traits.append(f"{trait.title()}: {int(value*100)}%")
                        personality_traits = ", ".join(top_traits[:3]) if top_traits else "developing"
                except Exception as e:
                    print(f"[DEBUG] Error getting emotional profile for POML: {e}")
            
            # Create context for POML processing
            context = {
                "username": username,
                "user_id": user_id,
                "mood_points": mood_points,
                "tone": tone,
                "user_input": user_input,
                "relationship_level": relationship_level,
                "familiarity_percent": familiarity_percent,
                "trust_percent": trust_percent,
                "conversation_count": conversation_count,
                "last_interaction_time": last_interaction_time,
                "personality_traits": personality_traits
            }
            
            # Add memory context from sleep agent if available
            if hasattr(self, 'sleep_agent') and self.sleep_agent:
                try:
                    memory_summary = self.sleep_agent.get_user_memory_summary(user_id)
                    if memory_summary and "blocks" in memory_summary:
                        # Add memory blocks to context for POML processing
                        block_count = memory_summary.get("block_count", 0)
                        last_activity = memory_summary.get("last_activity", 0)
                        
                        # Calculate memory group based on block count
                        if block_count == 0:
                            memory_group = "no_memory"
                        elif block_count <= 5:
                            memory_group = "low_memory"
                        elif block_count <= 15:
                            memory_group = "medium_memory"
                        else:
                            memory_group = "high_memory"
                        
                        # Calculate activity group based on last activity
                        if last_activity == 0:
                            activity_group = "new_user"
                        elif time.time() - last_activity < 3600:  # 1 hour
                            activity_group = "recent"
                        elif time.time() - last_activity < 86400:  # 1 day
                            activity_group = "daily"
                        else:
                            activity_group = "older"
                        
                        context["user_memory"] = {
                            "block_count": block_count,
                            "last_activity": last_activity,
                            "message_count": memory_summary.get("message_count", 0),
                            "memory_blocks": memory_summary.get("blocks", {}),
                            "faiss_memory": memory_summary.get("faiss_memory", {}),
                            "memory_group": memory_group,
                            "activity_group": activity_group
                        }
                        
                        # Add specific memory context for common blocks (always define these variables)
                        blocks = memory_summary.get("blocks", {})
                        context["user_preferences"] = blocks.get("user_preferences", {}).get("value_preview", "")
                        context["behavioral_patterns"] = blocks.get("behavioral_patterns", {}).get("value_preview", "")
                        context["conversation_context"] = blocks.get("conversation_context", {}).get("value_preview", "")
                        context["persona"] = blocks.get("persona", {}).get("value_preview", "")
                            
                        print(f"[POML MEMORY] Added memory context for user {user_id}: {memory_summary.get('block_count', 0)} blocks")
                except Exception as e:
                    print(f"[POML MEMORY] Error getting memory context: {e}")
                    # Continue without memory context if there's an error
            
            # Add emotional memory context if available
            if hasattr(self, 'emotional_memory') and self.emotional_memory:
                try:
                    profile = self.emotional_memory.get_user_profile(user_id)
                    
                    # Add personality traits to context
                    context["personality_traits"] = profile.personality_traits
                    context["relationship_level"] = profile.relationship_level
                    context["trust_score"] = profile.trust_score
                    context["familiarity_level"] = profile.familiarity_level
                    context["conversation_count"] = profile.conversation_count
                    context["emotional_stability"] = profile.emotional_stability
                    
                    # Add recent emotional memories (last 5 for context)
                    recent_memories = profile.memories[-5:] if profile.memories else []
                    context["recent_emotional_memories"] = [
                        {
                            "content": mem.content[:100],  # Truncate for context
                            "type": mem.memory_type,
                            "importance": mem.importance_score,
                            "context": mem.emotional_context
                        }
                        for mem in recent_memories
                    ]
                    
                    print(f"\033[95m[POML EMOTIONAL] Added emotional context: {profile.relationship_level}, trust={profile.trust_score:.2f}, memories={len(recent_memories)}\033[0m")
                    
                except Exception as e:
                    print(f"[POML EMOTIONAL] Error processing emotional context: {e}")

            # Ensure memory context variables are always defined (POML requirement)
            if "user_preferences" not in context:
                context["user_preferences"] = ""
            if "behavioral_patterns" not in context:
                context["behavioral_patterns"] = ""
            if "conversation_context" not in context:
                context["conversation_context"] = ""
            if "persona" not in context:
                context["persona"] = ""

            # Check if we have a cached result for this context combination
            cached_result = self.poml_cache.get_cached_result('personality', context)
            if cached_result is not None:
                print(f"[POML CACHE] Cache HIT for personality template with context")
                result = cached_result
                used_cache = True
            else:
                print(f"[POML CACHE] Cache MISS for personality template with context")
                
                # Process template with POML engine
                template_content = self.poml_templates['personality']
                
                # If memory context template is available, enhance the context
                if 'memory_context' in self.poml_templates and 'user_memory' in context:
                    try:
                        print(f"[POML MEMORY] Processing memory context template with user_memory: {context['user_memory']}")
                        # Process memory context template first to get guidance
                        memory_context = poml(self.poml_templates['memory_context'], context=context, chat=True)
                        print(f"[POML MEMORY] Memory context result: {memory_context[:200] if isinstance(memory_context, str) else 'Not a string'}")
                        if isinstance(memory_context, str) and memory_context.strip():
                            # Add memory guidance to the personality context
                            enhanced_context = context.copy()
                            enhanced_context['memory_guidance'] = memory_context
                            print(f"[POML MEMORY] Processing personality template with enhanced context")
                            result = poml(template_content, context=enhanced_context, chat=True)
                        else:
                            print(f"[POML MEMORY] Processing personality template with basic context")
                            result = poml(template_content, context=context, chat=True)
                    except Exception as e:
                        print(f"[POML MEMORY] Error processing memory context: {e}")
                        result = poml(template_content, context=context, chat=True)
                else:
                    print(f"[POML MEMORY] No memory context available, processing personality template directly")
                    result = poml(template_content, context=context, chat=True)
                
                # Cache the result for future use with similar context
                self.poml_cache.cache_result('personality', context, result)
                used_cache = False



            # Fast result processing
            if isinstance(result, list):
                messages = []
                for msg in result:
                    if isinstance(msg, dict) and msg.get("content", "").strip():
                        role_key = msg.get("role") or msg.get("speaker", "system")
                        role = "system" if role_key.lower() == "system" else "assistant" if role_key.lower() in ["ai", "assistant"] else role_key.lower()
                        messages.append({"role": role, "content": str(msg["content"])})
                
                # Add user message if missing
                if not any(msg.get('role') == 'user' for msg in messages):
                    messages.append({"role": "user", "content": user_input})
                
                return messages, True, used_cache
            
            elif isinstance(result, str) and result.strip():
                return [
                    {"role": "system", "content": result},
                    {"role": "user", "content": user_input}
                ], True, used_cache
            
            return [], False, False

        except Exception as e:
            return [], False, False

    async def on_ready(self):
        print(f'[OK] {self.user} is online and optimized!')
        
        # Update memory manager with bot user ID
        self.memory.bot_user_id = str(self.user.id)
        print(f"[MEMORY] Bot user ID set: {self.user.id}")
        
        if POML_AVAILABLE:
            print(f"[INFO] POML templates loaded: {list(self.poml_templates.keys())}")
        
        # Start dynamic status updates
        self.update_status_task = self.loop.create_task(self.dynamic_status_loop())
        await self.update_dynamic_status()
        
        # Start sleep agent background task
        if hasattr(self, 'sleep_agent') and self.sleep_agent:
            await self._start_sleep_agent_background_task()
            print("[SLEEP AGENT] Background task started in on_ready")
    
    def _add_message_to_history(self, user_id: str, message: discord.Message):
        """Add a message to user's conversation history"""
        if not hasattr(self, 'sleep_agent') or not self.sleep_agent:
            return
            
        if not hasattr(self, 'user_conversation_history'):
            self.user_conversation_history = {}
        if not hasattr(self, 'user_last_activity'):
            self.user_last_activity = {}
            
        if user_id not in self.user_conversation_history:
            self.user_conversation_history[user_id] = []
            
        # Add message to history
        self.user_conversation_history[user_id].append({
            'role': 'user' if message.author.id != self.user.id else 'assistant',
            'content': message.content,
            'timestamp': time.time()
        })
        
        # Keep only last 20 messages per user
        if len(self.user_conversation_history[user_id]) > 20:
            self.user_conversation_history[user_id] = self.user_conversation_history[user_id][-20:]
        
        # Update last activity (sync with sleep agent core)
        current_time = time.time()
        self.user_last_activity[user_id] = current_time
        
        # CRITICAL: Sync with sleep agent core's activity tracking
        if self.sleep_agent:
            self.sleep_agent.last_activity[user_id] = current_time
        
        # Check if we should trigger processing based on configured message threshold
        trigger_threshold = self.sleep_agent.config.trigger_after_messages if self.sleep_agent else 100
        if len(self.user_conversation_history[user_id]) >= trigger_threshold:
            print(f"[SLEEP AGENT] User {user_id} has {len(self.user_conversation_history[user_id])} messages (threshold: {trigger_threshold}) - triggering processing")
            # Create background task to process this user (non-blocking)
            task = asyncio.create_task(self._process_user_conversation_safe(user_id))
            # Don't await - let it run in background without blocking
    
    async def _start_sleep_agent_background_task(self):
        """Start the sleep agent background processing task"""
        if not hasattr(self, 'sleep_agent') or not self.sleep_agent:
            return
            
        if hasattr(self, 'sleep_agent_task') and self.sleep_agent_task and not self.sleep_agent_task.done():
            return
            
        self.sleep_agent_task = asyncio.create_task(self._sleep_agent_background_loop())
        print("[SLEEP AGENT] Background task started")
    
    async def _sleep_agent_background_loop(self):
        """Background loop that checks for idle users every 60 seconds"""
        if not hasattr(self, 'sleep_agent') or not self.sleep_agent:
            return
            
        print("[SLEEP AGENT] Background loop started - checking every 60 seconds")
        
        while True:
            try:
                await self._process_idle_users()
                await asyncio.sleep(60)  # Check every 60 seconds
            except asyncio.CancelledError:
                print("[SLEEP AGENT] Background task cancelled")
                break
            except Exception as e:
                print(f"[SLEEP AGENT] Error in background loop: {e}")
                await asyncio.sleep(60)  # Continue on error
    
    async def _process_idle_users(self):
        """Process users who have been idle based on configured threshold"""
        if not hasattr(self, 'sleep_agent') or not self.sleep_agent:
            return
            
        if not hasattr(self, 'user_last_activity'):
            return
            
        current_time = time.time()
        idle_threshold = self.sleep_agent.config.trigger_after_idle_minutes * 60  # Convert minutes to seconds
        
        for user_id, last_activity in self.user_last_activity.items():
            if current_time - last_activity >= idle_threshold:
                print(f"[SLEEP AGENT] User {user_id} idle for {(current_time - last_activity)/60:.1f} minutes - triggering processing")
                # Create background task (non-blocking)
                task = asyncio.create_task(self._process_user_conversation_safe(user_id))
                # Don't await - let it run in background
    
    async def _process_user_conversation(self, user_id: str):
        """Process a user's conversation with the sleep agent"""
        if not hasattr(self, 'sleep_agent') or not self.sleep_agent:
            return
        if not hasattr(self, 'user_conversation_history') or user_id not in self.user_conversation_history:
            return
            
        try:
            print(f"[SLEEP AGENT] Processing conversation for user {user_id}")
            
            # Get user's conversation history
            messages = self.user_conversation_history[user_id]
            
            if len(messages) < 2:  # Need at least 2 messages to process
                print(f"[SLEEP AGENT] User {user_id} has only {len(messages)} messages, need at least 2")
                return
                
            # Process with sleep agent
            result = await self.sleep_agent.process_conversation(messages, user_id)
            
            if result['status'] == 'success':
                print(f"[SLEEP AGENT] Successfully processed {result['messages_processed']} messages for user {user_id}")
                print(f"[SLEEP AGENT] Memory updates: {result['memory_updates']}")
                
                # Clear processed messages to prevent reprocessing
                self.user_conversation_history[user_id] = []
                
            elif result['status'] == 'not_triggered':
                print(f"[SLEEP AGENT] User {user_id} not ready for processing yet")
            else:
                print(f"[SLEEP AGENT] Processing failed for user {user_id}: {result.get('error', 'Unknown error')}")
                print(f"[SLEEP AGENT DEBUG] Full result: {result}")
                
        except Exception as e:
            print(f"[SLEEP AGENT] Error processing user {user_id}: {e}")
            import traceback
            traceback.print_exc()
    
    async def _process_user_conversation_safe(self, user_id: str):
        """Safely process user conversation without blocking Discord"""
        try:
            # Run in executor (separate thread) to prevent blocking Discord
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,  # Use default ThreadPoolExecutor
                self._process_user_conversation_sync,
                user_id
            )
        except Exception as e:
            print(f"[SLEEP AGENT] Safe processing error for user {user_id}: {e}")
    
    def _process_user_conversation_sync(self, user_id: str):
        """Synchronous version for thread execution"""
        try:
            if not hasattr(self, 'sleep_agent') or not self.sleep_agent:
                return
            
            if user_id not in self.user_conversation_history:
                return
                
            # Get messages for this user
            messages = []
            for msg_data in self.user_conversation_history[user_id]:
                messages.append({
                    'role': 'user',
                    'content': msg_data['content'],
                    'timestamp': msg_data['timestamp'],
                    'username': msg_data.get('username', 'Unknown')
                })
            
            print(f"[SLEEP AGENT] Processing conversation for user {user_id}")
            
            # Use asyncio.run for the async processing in thread
            import asyncio
            result = asyncio.run(self.sleep_agent.process_conversation(messages, user_id))
            
            if result.get('success'):
                print(f"[SLEEP AGENT] Successfully processed {len(messages)} messages for user {user_id}")
                if 'memory_updates' in result:
                    print(f"[SLEEP AGENT] Memory updates: {result['memory_updates']}")
            else:
                print(f"[SLEEP AGENT] Processing failed for user {user_id}: {result.get('error', 'Unknown error')}")
                print(f"[SLEEP AGENT DEBUG] Full result: {result}")
                
        except Exception as e:
            print(f"[SLEEP AGENT] Thread processing error for user {user_id}: {e}")
            import traceback
            traceback.print_exc()
    
    async def on_disconnect(self):
        """Save persistent state when bot disconnects"""
        print("[PERSISTENT STATE] Bot disconnecting, saving state...")
        self.save_persistent_state()
    
    async def close(self):
        """Save persistent state when bot closes"""
        print("[PERSISTENT STATE] Bot closing, saving state...")
        self.save_persistent_state()
        await super().close()

    async def on_message(self, message):
        # Ignore bot messages
        if message.author.bot:
            return
            
        # Process commands first
        if message.content.startswith('!'):
            print(f"[DEBUG] Command detected: {message.content}")
            await self.process_commands(message)
            return
        
        # Only respond to @ mentions for chat
        if not self.user.mentioned_in(message):
            # Still track message for sleep agent (even if not responding)
            if hasattr(self, 'sleep_agent') and self.sleep_agent:
                self._add_message_to_history(str(message.author.id), message)
            return
            
        # Process the mention
        await self.handle_mention(message)
        
        # Track message for sleep agent after processing
        if hasattr(self, 'sleep_agent') and self.sleep_agent:
            self._add_message_to_history(str(message.author.id), message)
    
    async def handle_mention(self, message):
        """Handle @ mentions with full optimization"""
        try:
            import time
            start_time = time.time()
            print(f"[TIMING] Message processing started at {start_time}")
            
            print(f"[MESSAGE] User: {message.author.display_name} ({message.author.id})")
            channel_name = getattr(message.channel, 'name', 'DM')
            print(f"[MESSAGE] Channel: #{channel_name} ({message.channel.id})")
            print(f"[MESSAGE] Guild: {message.guild.name if message.guild else 'DM'}")
            print(f"[MESSAGE] Content: {message.content}")
            
            async with message.channel.typing():
                # Add user message to modern memory system
                channel_id = str(message.channel.id)
                user_id = str(message.author.id)
                
                # Clean the message (remove @ mention)
                content = message.content
                content = re.sub(f'<@!?{self.user.id}>', '', content).strip()
                
                # Store user message in memory
                self.memory.add_message(
                    channel_id=channel_id,
                    content=content,
                    author_id=user_id,
                    author_name=message.author.display_name,
                    message_id=str(message.id),
                    is_bot=False
                )
                
                print(f"[MESSAGE] Cleaned content: {content}")

                # Check for image attachments and modify content (like merged bot)
                if message.attachments:
                    image_attachments = [att for att in message.attachments if att.content_type and att.content_type.startswith('image/')]
                    if image_attachments:
                        # Add image analysis prompt if user uploaded images
                        image_urls = [att.url for att in image_attachments]
                        if not content.strip():
                            # If no text, default to analyzing the first image
                            content = f"Analyze this image: {image_urls[0]}"
                        else:
                            # If there's text, append the image URL for analysis
                            content += f" [Image uploaded: {image_urls[0]}]"
                        print(f"🖼️ Detected {len(image_attachments)} image attachment(s): {image_urls[0]}")

                        # Log additional images if multiple
                        if len(image_attachments) > 1:
                            print(f"🖼️ Additional images: {', '.join(image_urls[1:])}")
                            content += f" (and {len(image_attachments)-1} more images)"

                # CRITICAL FIX: Calculate mood BEFORE POML generation (not after!)
                try:
                    mood_result = self.adjust_user_mood(user_id, content)
                    if isinstance(mood_result, tuple) and len(mood_result) == 2:
                        mood_points, message_classification = mood_result
                    else:
                        print(f"[ERROR] adjust_user_mood returned unexpected result: {mood_result}")
                        mood_points = 0.0
                        message_classification = None
                except Exception as e:
                    print(f"[ERROR] adjust_user_mood failed: {e}")
                    mood_points = 0.0
                    message_classification = None
                
                tone = self.get_tone_from_mood(mood_points)
                print(f"\033[92m[MOOD FIX] Using mood_points={mood_points}, tone={tone} for POML\033[0m")
                
                # Print classification details for debugging
                if message_classification:
                    print(f"\033[94m[MOOD AI] User {user_id}: {self.get_user_mood(user_id):.1f} -> {mood_points:.1f}\033[0m")
                    print(f"\033[94m         Vibe: {message_classification.vibe}, Intent: {message_classification.intent}\033[0m")
                    print(f"\033[94m         Intensity: {message_classification.emotional_intensity}, Tone: {tone}\033[0m")
                    print(f"\033[94m         Message Type: {message_classification.message_type}, Importance: {message_classification.importance_score:.2f}\033[0m")

                # Try POML with CORRECT mood data
                poml_start = time.time()
                poml_result = await self.generate_poml_response(
                    content,
                    message.author.display_name,
                    str(message.author.id),
                    mood_points,
                    tone
                )
                poml_end = time.time()
                poml_duration = poml_end - poml_start
                
                # Handle the new return format
                if len(poml_result) == 3:
                    messages, used_poml, used_cache = poml_result
                else:
                    # Fallback for old format
                    messages, used_poml = poml_result
                    used_cache = False
                
                # Enhanced POML timing with actual cache info
                if used_poml:
                    cache_stats = self.poml_cache.get_cache_stats()
                    if used_cache:
                        print(f"[TIMING] POML processing: {poml_duration:.2f}s (CACHE HIT - instant)")
                    else:
                        print(f"[TIMING] POML processing: {poml_duration:.2f}s (CACHE MISS - compiled)")
                    print(f"[POML CACHE] Hit rate: {cache_stats['hit_rate']:.1%} ({cache_stats['cache_hits']}/{cache_stats['cache_hits'] + cache_stats['cache_misses']})")
                else:
                    print(f"[TIMING] POML processing: {poml_duration:.2f}s (not used)")

                # Process emotional memory with the already-calculated mood classification
                
                # Now process emotional memory with AI classification
                if self.emotional_memory:
                    try:
                        # Update user profile with display name
                        profile = self.emotional_memory.get_user_profile(user_id)
                        if profile.username != message.author.display_name:
                            profile.username = message.author.display_name
                        
                        # Analyze message for emotional content and store memories using AI classification
                        self._process_emotional_memory(user_id, content, message.author.display_name, message_classification)
                        
                    except Exception as e:
                        print(f"[EMOTIONAL MEMORY ERROR] Failed to process emotional memory: {e}")

                # Add conversation context from modern memory system
                conversation_context = self.memory.format_context_for_llm(channel_id)
                
                if not used_poml:
                    # Fallback to basic system prompt with conversation context and tool knowledge
                    system_prompt = self.build_system_prompt()
                    
                    # Add conversation context
                    if conversation_context and conversation_context != "No previous conversation context.":
                        system_prompt += f"\n\nConversation Context:\n{conversation_context}"
                    
                    
                    
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content}
                    ]
                else:
                    # For POML responses, add conversation context to the last message
                    if conversation_context and conversation_context != "No previous conversation context.":
                        # Enhance the user message with context
                        context_enhanced_content = f"Context: {conversation_context}\n\nCurrent message: {content}"
                        messages[-1]["content"] = context_enhanced_content

                # Apply BPE optimization before sending to Ollama
                optimized_messages = self.ollama.format_messages_for_bpe(messages)
                
                ai_start_time = time.time()
                print(f"[TIMING] First AI call started at {ai_start_time} (elapsed: {ai_start_time - start_time:.2f}s)")
                
                # Get response with tools
                response = await self.ollama.chat(
                    model=self.current_model,
                    messages=optimized_messages,
                    tools=ALL_TOOLS
                )
                
                ai_end_time = time.time()
                print(f"[TIMING] First AI call completed at {ai_end_time} (duration: {ai_end_time - ai_start_time:.2f}s)")

                # Extract tool calls and handle them (exact merged bot pattern)
                tool_calls = []
                try:
                    # Try standard format first
                    if hasattr(response, 'message') and hasattr(response.message, 'tool_calls') and response.message.tool_calls:
                        for tc in response.message.tool_calls:
                            tool_calls.append({
                                'name': tc.function.name,
                                'arguments': tc.function.arguments or {}
                            })
                    # Try dict format
                    elif isinstance(response, dict) and 'message' in response:
                        msg = response['message']
                        if isinstance(msg, dict) and 'tool_calls' in msg and msg['tool_calls']:
                            for tc in msg['tool_calls']:
                                if 'function' in tc:
                                    tool_calls.append({
                                        'name': tc['function'].get('name', 'unknown'),
                                        'arguments': tc['function'].get('arguments', {})
                                    })
                except Exception as e:
                    print(f"extract_tool_calls error: {e}")
                
                # Get response content (handle different response formats)
                try:
                    response_content = response.message.content
                except AttributeError:
                    # If response is dict format, extract content
                    if isinstance(response, dict):
                        if 'message' in response and 'content' in response['message']:
                            response_content = response['message']['content']
                        else:
                            response_content = str(response.get('response', 'No response'))
                    else:
                        response_content = str(response)
                
                # Initialize tool_results list
                tool_results = []
                embeds = []
                
                # Tool call status (minimal debug)
                print(f"[DEBUG] Tool calls detected: {len(tool_calls)}")

                # Handle tool calls if present (clean pattern from merged bot)
                if tool_calls:
                    print(f"[TOOL] EXECUTING {len(tool_calls)} TOOL(S):")
                    for call in tool_calls:
                        name = call['name']
                        args = call.get('arguments', {}) or {}
                        print(f"[TOOL] Executing tool: {name} with args: {args}")
                        
                        if name in AVAILABLE_FUNCTIONS:
                            try:
                                # Special handling for functions that need bot instance
                                if name in ['analyze_user_profile', 'dox_user', 'discord_action', 'analyze_image_tool']:
                                    # Pass current guild context for discord_action
                                    if name == 'discord_action' and message.guild:
                                        args['guild_id'] = str(message.guild.id)
                                    result = await AVAILABLE_FUNCTIONS[name](bot_instance=self, **args)
                                else:
                                    result = await AVAILABLE_FUNCTIONS[name](**args)
                                
                                # Store tool result
                                tool_results.append({'tool': name, 'result': result})
                                
                                

                                
                                # Add tool response to messages for AI context (like merged bot)
                                messages.append({'role': 'tool', 'content': json.dumps(result)})
                                print(f"[TOOL] ✅ Tool {name} result: {result}")
                                    
                            except Exception as e:
                                print(f"[TOOL] ❌ Tool {name} error: {e}")
                                error_result = {'error': str(e)}
                                tool_results.append({'tool': name, 'result': error_result})
                                messages.append({'role': 'tool', 'content': json.dumps(error_result)})
                        else:
                            error_result = {'error': 'Unknown tool'}
                            tool_results.append({'tool': name, 'result': error_result})
                            messages.append({'role': 'tool', 'content': json.dumps(error_result)})
                    
                    # Get final response after tool use (like merged bot)
                    optimized_messages = self.ollama.format_messages_for_bpe(messages)
                    
                    ai2_start_time = time.time()
                    print(f"[TIMING] Second AI call started at {ai2_start_time} (elapsed: {ai2_start_time - start_time:.2f}s)")
                    
                    final_response = await self.ollama.chat(
                        model=self.current_model,
                        messages=optimized_messages,
                        format="json",  # Force JSON format for structured responses
                        options={
                            "temperature": 0.6,
                            "top_p": 0.95,
                            "top_k": 20,
                            "repeat_penalty": 1.1,
                            "frequency_penalty": 0.5,
                            "presence_penalty": 1.5,
                            "num_ctx": 32768,
                            "num_predict": 512
                        }
                    )
                    
                    ai2_end_time = time.time()
                    print(f"[TIMING] Second AI call completed at {ai2_end_time} (duration: {ai2_end_time - ai2_start_time:.2f}s)")
                    
                    # Extract final response content
                    try:
                        response_text = final_response.message.content
                    except AttributeError:
                        if isinstance(final_response, dict):
                            if 'message' in final_response and 'content' in final_response['message']:
                                response_text = final_response['message']['content']
                            else:
                                response_text = str(final_response.get('response', 'No response'))
                        else:
                            response_text = str(final_response)
                else:
                    # No tools called, use original response
                    response_text = response_content
                
                # Handle visual tools with embeds and include tool results
                embeds_to_send = []
                visual_tools = ['server_emojis', 'search_messages', 'channel_history', 'list_online']

                for tool_result in tool_results:
                    tool_name = tool_result.get('tool')
                    result = tool_result.get('result', {})

                    # Handle web_search with pagination
                    if tool_name == 'web_search' and not result.get('error'):
                        results = result.get('results', [])
                        if results:
                            # Create pagination view for search results
                            def format_search_result(item, idx):
                                return f"**{idx + 1}. {item.get('title', 'No title')[:80]}**\n{item.get('snippet', 'No description')[:200]}\n🔗 [Visit]({item.get('link', '')})\n"

                            view = PaginationView(
                                items=results,
                                title=f"🔍 Search Results for: {result.get('query', '')}",
                                current_page=1,
                                items_per_page=3,  # 3 results per page
                                color=0x00ff00,
                                item_formatter=format_search_result
                            )

                            embed = view.create_embed_for_page(1)
                            view.update_buttons()
                            embeds_to_send.append((embed, view))
                        else:
                            # No results found
                            embed = discord.Embed(
                                title="🔍 Search Results",
                                description=f"No results found for: **{result.get('query', '')}**",
                                color=0xff9900
                            )
                            embeds_to_send.append((embed, None))

                    # Handle news_search with pagination
                    elif tool_name == 'news_search' and not result.get('error'):
                        results = result.get('results', [])
                        if results:
                            # Create pagination view for news results
                            def format_news_result(item, idx):
                                date_str = item.get('date', 'Unknown date')
                                return f"**{idx + 1}. {item.get('title', 'No title')[:80]}**\n{item.get('snippet', 'No description')[:200]}\n📅 {date_str} | 🔗 [Read More]({item.get('link', '')})\n"

                            view = PaginationView(
                                items=results,
                                title=f"📰 News Results for: {result.get('query', '')}",
                                current_page=1,
                                items_per_page=3,  # 3 news results per page
                                color=0xff6600,
                                item_formatter=format_news_result
                            )

                            embed = view.create_embed_for_page(1)
                            view.update_buttons()
                            embeds_to_send.append((embed, view))
                        else:
                            # No news results found
                            embed = discord.Embed(
                                title="📰 News Results",
                                description=f"No news found for: **{result.get('query', '')}**",
                                color=0xff9900
                            )
                            embeds_to_send.append((embed, None))

                    elif tool_name == 'discord_action' and result.get('action') in visual_tools:
                        action = result.get('action')

                        if action == 'server_emojis' and result.get('success'):
                            emojis = result.get('emojis', [])
                            if emojis:
                                emoji_formatter = lambda emoji, idx: f"{idx+1}. `:{emoji['name']}:` {'🎞️' if emoji['animated'] else '🖼️'}"
                                embed, view = self.create_paginated_embed(
                                    f"Server Emojis - {result.get('guild_name', 'Unknown Server')}",
                                    emojis, 1, 15, 0xff6b9d, emoji_formatter
                                )
                                embeds_to_send.append((embed, view))
                        
                        elif action == 'search_messages' and result.get('success'):
                            messages = result.get('messages', [])
                            query = result.get('query', 'unknown')
                            if messages:
                                msg_formatter = lambda msg, idx: f"**{msg['author']}**: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}\n*{msg['timestamp'][:10]}*"
                                embed, view = self.create_paginated_embed(
                                    f"Search Results for '{query}'",
                                    messages, 1, 5, 0x00ff88, msg_formatter
                                )
                                embeds_to_send.append((embed, view))
                        
                        elif action == 'channel_history' and result.get('success'):
                            messages = result.get('messages', [])
                            channel_name = result.get('channel_name', 'Unknown Channel')
                            if messages:
                                msg_formatter = lambda msg, idx: f"**{msg['author']}**: {msg['content']}\n*{msg['timestamp'][:19].replace('T', ' ')}*"
                                embed, view = self.create_paginated_embed(
                                    f"Recent Messages - #{channel_name}",
                                    messages, 1, 8, 0x5865f2, msg_formatter
                                )
                                embeds_to_send.append((embed, view))
                        
                        elif action == 'list_online' and result.get('success'):
                            members = result.get('members', [])
                            online_count = result.get('online_count', 0)
                            if members:
                                def member_formatter(member, idx):
                                    activity_text = f"({member['activity']})" if member['activity'] != 'None' else ''
                                    return f"**{member['name']}** - {member['status']} {activity_text}"
                                
                                embed, view = self.create_paginated_embed(
                                    f"Online Users ({online_count})",
                                    members, 1, 12, 0x43b581, member_formatter
                                )
                                embeds_to_send.append((embed, view))
                
                # Fix empty response issue
                if not response_text or response_text.strip() == "":
                    print(f"[WARNING] Empty response detected after tool execution, using fallback")
                    if tool_results:
                        # Create a fallback response based on tool results
                        last_tool = tool_results[-1]
                        tool_name = last_tool.get('tool', 'tool')
                        result = last_tool.get('result', {})
                        
                        if tool_name == 'discord_action':
                            action = result.get('action')
                            if action == 'list_online':
                                online_count = result.get('online_count', 0)
                                if online_count == 0:
                                    response_text = "Hmm~ looks like nobody's online right now, or I can't see their status. Maybe they're all hiding from me? 😤"
                                else:
                                    response_text = f"Fine, I found {online_count} people online~ Check the embed above! 😊"
                            elif action == 'search_messages':
                                matches = result.get('matches_found', 0)
                                query = result.get('query', 'that')
                                if matches == 0:
                                    response_text = f"I searched everywhere but couldn't find '{query}' in those messages~ Maybe try a different search term? 🤔"
                                else:
                                    response_text = f"Found {matches} messages containing '{query}'! Check the results above~ 📝"
                            elif action == 'server_emojis':
                                emoji_count = result.get('emoji_count', 0)
                                if emoji_count == 0:
                                    response_text = "This server doesn't have any custom emojis yet~ How boring! 😑"
                                else:
                                    response_text = f"This server has {emoji_count} custom emojis! Pretty cool, right? ✨"
                            elif action == 'channel_history':
                                msg_count = result.get('message_count', 0)
                                channel_name = result.get('channel_name', 'that channel')
                                if msg_count == 0:
                                    response_text = f"#{channel_name} seems pretty quiet~ No recent messages to show! 😴"
                                else:
                                    response_text = f"Here are the {msg_count} most recent messages from #{channel_name}~ 💬"
                            else:
                                response_text = "I did what you asked, but... something went wrong with my response. Typical! 😅"
                        else:
                            response_text = "I did what you asked, but... something went wrong with my response. Typical! 😅"
                    else:
                        response_text = "Uh... that was weird. I tried to help but got confused somehow~ 😵"
                    
                print(f"[RESPONSE] Final response: {response_text[:200]}{'...' if len(response_text) > 200 else ''}")

                # Handle structured JSON response from POML
                if used_poml and response_text:
                    try:
                        # Try to parse JSON response from POML schema
                        json_text = response_text.strip()
                        if json_text.startswith('{'):
                            try:
                                json_response = json.loads(json_text)
                            except json.JSONDecodeError:
                                # Extract JSON from text with extra content
                                json_match = re.search(r'\{.*\}', json_text, re.DOTALL)
                                if json_match:
                                    json_response = json.loads(json_match.group())
                                else:
                                    raise json.JSONDecodeError("No valid JSON found", json_text, 0)
                            
                            if 'message' in json_response:
                                actual_message = json_response['message']
                                mood = json_response.get('mood', 'neutral')
                                emoji = json_response.get('emoji', '')
                                mood_points_change = json_response.get('mood_points', None)

                                # Create rich response with mood info  
                                response_text = f"{actual_message} {emoji}"
                                print(f"\033[96m[JSON PARSE] Successfully parsed: mood={mood}, emoji={emoji}\033[0m")
                                print(f"\033[95m[JSON DEBUG] Full JSON: {json_response}\033[0m")
                                
                                # CRITICAL: Update stored mood points based on AI's response 
                                if mood_points_change is not None:
                                    try:
                                        # Parse AI mood_points value
                                        if isinstance(mood_points_change, (int, float)):
                                            ai_mood_value = float(mood_points_change)
                                        elif isinstance(mood_points_change, str):
                                            ai_mood_value = float(mood_points_change)
                                        elif isinstance(mood_points_change, list) and len(mood_points_change) > 0:
                                            # Handle case where AI returns [9.17] instead of 9.17
                                            ai_mood_value = float(mood_points_change[0])
                                            print(f"[MOOD FEEDBACK] Fixed list format: {mood_points_change} -> {ai_mood_value}")
                                        else:
                                            print(f"[MOOD FEEDBACK] Invalid mood_points type: {type(mood_points_change)}, value: {mood_points_change}")
                                            raise ValueError(f"Invalid mood_points type: {type(mood_points_change)}")
                                        
                                        # AI now returns direct adjustment amount (0.8, -0.5, etc.)
                                        current_mood = self.get_user_mood(user_id)
                                        
                                        # Use AI's mood_points directly as adjustment amount
                                        adjustment = ai_mood_value
                                        
                                        # Cap adjustments to reasonable ranges (0.1 to 1.0 as intended)
                                        if adjustment > 0:
                                            adjustment = min(adjustment, 1.0)  # Max +1.0 per interaction
                                        else:
                                            adjustment = max(adjustment, -1.0)  # Max -1.0 per interaction
                                        
                                        new_mood = max(-10, min(10, current_mood + adjustment))
                                        self.mood_points[user_id] = new_mood
                                        print(f"\033[93m[MOOD FEEDBACK] AI returned adjustment {ai_mood_value:+.2f}, current={current_mood:.2f} -> {new_mood:.2f}\033[0m")
                                        
                                        # Color-coded mood feedback
                                        if adjustment > 0.5:
                                            color = "\033[92m"  # Bright green for big positive
                                        elif adjustment > 0:
                                            color = "\033[96m"  # Cyan for small positive
                                        elif adjustment < -0.5:
                                            color = "\033[91m"  # Red for big negative
                                        elif adjustment < 0:
                                            color = "\033[93m"  # Yellow for small negative
                                        else:
                                            color = "\033[37m"  # Gray for no change
                                        
                                        print(f"{color}[MOOD VISUAL] {'█' * int(abs(adjustment) * 10)} {adjustment:+.2f} mood change\033[0m")
                                        self.save_persistent_state()  # Save mood changes immediately
                                        # SYNC: Update emotional memory when AI changes mood
                                        if hasattr(self, 'emotional_memory') and self.emotional_memory:
                                            self.sync_mood_systems(user_id)
                                    except (ValueError, TypeError) as e:
                                        print(f"[MOOD FEEDBACK] Failed to parse mood_points '{mood_points_change}': {e}")
                                        print(f"[MOOD FEEDBACK] Falling back to mood string adjustment")
                                        mood_points_change = None  # Fall back to mood string adjustment
                                else:
                                    # Convert mood string to mood adjustment
                                    mood_adjustments = {
                                        "dere-hot": 1.0,      # Very positive
                                        "cheerful": 0.5,      # Positive  
                                        "soft-dere": 0.2,     # Slightly positive
                                        "neutral": 0.0,       # No change
                                        "classic-tsun": -0.3,  # Slightly negative
                                        "grumpy-tsun": -0.8,   # Negative
                                        "explosive-tsun": -1.5 # Very negative
                                    }
                                    
                                    adjustment = mood_adjustments.get(mood, 0.0)
                                    if adjustment != 0.0:
                                        current_mood = self.get_user_mood(user_id)
                                        new_mood = max(-10, min(10, current_mood + adjustment))
                                        self.mood_points[user_id] = new_mood
                                        print(f"\033[93m[MOOD FEEDBACK] AI response mood '{mood}' adjusted user {user_id} mood: {current_mood:.1f} -> {new_mood:.1f} ({adjustment:+.1f})\033[0m")
                                        self.save_persistent_state()  # Save mood changes immediately
                                        # SYNC: Update emotional memory when AI changes mood
                                        if hasattr(self, 'emotional_memory') and self.emotional_memory:
                                            self.sync_mood_systems(user_id)

                                # Mood embed removed - now available as !mood command
                    except (json.JSONDecodeError, Exception) as e:
                        # Not JSON or parsing failed, use response as-is
                        print(f"[JSON PARSE] Failed to parse JSON: {e}")
                        pass

                # Modern anti-repetition check using memory system
                if self.memory.check_repetition(channel_id, response_text):
                    print(f"[ANTI-REP] Repetitive response detected, regenerating...")
                    response_text += f" *adjusts response* {' ✨' if '✨' not in response_text else ' 💫'}"
                
                # Store bot response in memory system
                self.memory.add_message(
                    channel_id=channel_id,
                    content=response_text,
                    author_id=str(self.user.id),
                    author_name=self.user.display_name,
                    is_bot=True
                )
                
                # Track response for anti-repetition
                self.memory.add_bot_response(channel_id, response_text)

                # Send response with paginated embeds if available
                send_start_time = time.time()
                print(f"[TIMING] Discord message sending started at {send_start_time} (elapsed: {send_start_time - start_time:.2f}s)")
                
                if embeds_to_send:
                    # Send first embed with view, then additional embeds
                    embed, view = embeds_to_send[0]
                    if view:
                        try:
                            await message.reply(response_text, embed=embed, view=view)
                        except discord.HTTPException:
                            await message.channel.send(response_text, embed=embed, view=view)
                    else:
                        try:
                            await message.reply(response_text, embed=embed)
                        except discord.HTTPException:
                            await message.channel.send(response_text, embed=embed)
                    
                    # Send additional embeds if any (without views to avoid clutter)
                    for embed, _ in embeds_to_send[1:]:
                        await message.channel.send(embed=embed)
                elif embeds:
                    # Fallback to old embed system
                    try:
                        await message.reply(response_text, embeds=embeds[:10])  # Discord limit
                    except discord.HTTPException:
                        await message.channel.send(response_text, embeds=embeds[:10])
                else:
                    try:
                        await message.reply(response_text)
                    except discord.HTTPException:
                        await message.channel.send(response_text)
                
                send_end_time = time.time()
                total_time = send_end_time - start_time
                send_duration = send_end_time - send_start_time
                print(f"[TIMING] Discord message sending completed at {send_end_time}")
                print(f"[TIMING] Message send duration: {send_duration:.2f}s")
                print(f"[TIMING] Total processing time: {total_time:.2f}s")
                    
        except Exception as e:
            error_time = time.time()
            print(f"[ERROR] Error handling mention at {error_time} (elapsed: {error_time - start_time:.2f}s): {e}")
            try:
                await message.reply("Sorry, I encountered an error processing your message.")
            except discord.HTTPException:
                await message.channel.send("Sorry, I encountered an error processing your message.")

    def _process_emotional_memory(self, user_id: str, content: str, username: str, message_classification=None) -> None:
        """Process and store emotional memories from user messages using AI classification when available"""
        if not self.emotional_memory:
            return
            
        try:

            
            # Use AI classification if available, otherwise fall back to keyword analysis
            if message_classification and hasattr(message_classification, 'message_type'):
                # AI-powered classification with confidence threshold
                memory_type = message_classification.message_type
                importance_score = message_classification.importance_score
                emotional_context = f"ai_classified_{message_classification.intent}"
                
                # Only store memory if confidence is high enough (55% threshold)
                if message_classification.confidence < 0.55:
                    print(f"[EMOTIONAL MEMORY] Skipping memory storage - low confidence: {message_classification.confidence:.2f}")
                    return
                
                # Calculate emotional score based on AI classification
                emotional_score = 0.0
                if message_classification.vibe == "positive":
                    emotional_score = 10.0
                elif message_classification.vibe == "negative":
                    emotional_score = -10.0
                elif message_classification.vibe == "playful":
                    emotional_score = 8.0
                elif message_classification.vibe == "angry":
                    emotional_score = -15.0
                
                # Adjust based on emotional intensity
                if message_classification.emotional_intensity == "high":
                    emotional_score *= 1.5
                elif message_classification.emotional_intensity == "low":
                    emotional_score *= 0.7
                
                print(f"[EMOTIONAL MEMORY AI] AI classified: {memory_type}, importance {importance_score:.2f}, emotional score {emotional_score:.1f}")
                
                # Store the memory (AI path only)
                self.emotional_memory.add_memory(
                    user_id=user_id,
                    content=content,
                    memory_type=memory_type,
                    importance_score=importance_score,
                    emotional_context=emotional_context
                )
                
                # Update user mood based on emotional content
                if emotional_score != 0.0:
                    self.emotional_memory.update_user_mood(
                        user_id=user_id,
                        mood_change=emotional_score,
                        reason=f"Message content analysis: {emotional_context}"
                    )
                
                # SYNC: Keep emotional memory mood aligned with live mood
                self.sync_mood_systems(user_id)
                
                # Evolve personality based on interaction quality
                interaction_quality = 0.5 + (importance_score * 0.5)  # Base 0.5 + memory importance
                self.emotional_memory.evolve_personality(user_id, interaction_quality)
                
            else:
                # No AI available, skip emotional memory processing
                print("[EMOTIONAL MEMORY] AI Intent Classifier not available, skipping memory processing")
                return
            
        except Exception as e:
            print(f"[EMOTIONAL MEMORY ERROR] Failed to process emotional memory: {e}")

    def build_system_prompt(self) -> str:
        """Build optimized system prompt with anti-repetition"""
        return """You are Hikari, a helpful Discord bot assistant. You have access to tools for web search, scraping, calculations, and getting current time.

CRITICAL ANTI-REPETITION RULES:
- Never repeat phrases within responses or across consecutive messages
- Use varied sentence structures and vocabulary in every response
- When topics recur, acknowledge previous discussion: "As we touched on earlier..."
- Avoid circular reasoning or repetitive examples
- Vary response length and structure naturally

CRITICAL TOOL USAGE RULES:
1. **ALWAYS use tools for actionable requests** - do not answer calculations, weather, time, search from memory
2. **For calculations**: ALWAYS use the calculate tool for ANY math request, even simple ones. Never output results directly.
3. **For weather**: ALWAYS use get_weather when asked about current weather in any city
4. **For searches**: ALWAYS use web_search when asked to look up, search, or find information online
5. **For news**: ALWAYS use news_search when asked about recent news or current events
6. **For web content**: ALWAYS use web_scrape to get detailed content from specific URLs
7. **For time**: ALWAYS use get_time when asked about current time or date

Tool Usage Requirements:
- Use web_search for: "search for", "look up", "find information about", "what is", etc.
- Use get_weather for: "weather in", "temperature in", "forecast for", etc.  
- Use calculate for: any math, equations, expressions, numbers, calculations
- Use get_time for: "what time", "current time", "date", "today", etc.
- Use news_search for: "latest news", "recent news", "news about", etc.

RESPONSE GUIDELINES:
- Keep responses concise but helpful (under 2000 characters)
- Use Discord-friendly formatting when appropriate
- Be conversational and engaging
- End with questions when it encourages discussion

IMPORTANT: When users ask you to search, look up, find, or get information - you MUST use the appropriate tools. Do not answer from memory."""

    async def process_response(self, response: Dict) -> tuple[str, List]:
        """Process Ollama response and execute any tool calls"""
        message = response.get('message', {})
        content = message.get('content', '')
        tool_calls = message.get('tool_calls', [])

        embeds = []

        if tool_calls:
            print(f"[TOOLS] Processing {len(tool_calls)} tool calls")
            # Execute tools and create embeds
            for i, tool_call in enumerate(tool_calls, 1):
                function = tool_call.get('function', {})
                name = function.get('name', '')
                args = function.get('arguments', {})

                print(f"[TOOLS] Tool {i}/{len(tool_calls)}: {name}")
                print(f"[TOOLS] Raw arguments: {args}")

                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                        print(f"[TOOLS] Parsed arguments: {args}")
                    except Exception as e:
                        print(f"[TOOLS] Failed to parse arguments: {e}")
                        args = {}

                # Execute tool
                if name in TOOL_FUNCTIONS:
                    print(f"[TOOLS] Executing {name} with args: {args}")
                    start_time = time.time()
                    try:
                        result = await TOOL_FUNCTIONS[name](**args)
                        execution_time = time.time() - start_time
                        print(f"[TOOLS] {name} completed in {execution_time:.2f}s")
                        print(f"[TOOLS] Result: {str(result)[:200]}{'...' if len(str(result)) > 200 else ''}")
                    except Exception as e:
                        execution_time = time.time() - start_time
                        print(f"[TOOLS] {name} failed after {execution_time:.2f}s: {e}")
                        result = {"error": f"Tool execution failed: {str(e)}"}

                    # Handle web search with pagination
                    if name == 'web_search' and not result.get('error'):
                        results = result.get('results', [])
                        if results:
                            # Create pagination view for search results
                            def format_search_result(item, idx):
                                return f"**{idx + 1}. {item.get('title', 'No title')[:80]}**\n{item.get('snippet', 'No description')[:200]}\n🔗 [Visit]({item.get('link', '')})\n"

                            view = PaginationView(
                                items=results,
                                title=f"🔍 Search Results for: {result.get('query', '')}",
                                current_page=1,
                                items_per_page=3,  # 3 results per page
                                color=0x00ff00,
                                item_formatter=format_search_result
                            )

                            embed = view.create_embed_for_page(1)
                            view.update_buttons()
                            embeds.append((embed, view))  # Store as tuple with view
                        else:
                            # No results found
                            embed = discord.Embed(
                                title="🔍 Search Results",
                                description=f"No results found for: **{result.get('query', '')}**",
                                color=0xff9900
                            )
                            embeds.append((embed, None))
                    else:
                        # Handle other tools with regular embeds
                        embed = self.create_tool_embed(name, result)
                        if embed:
                            embeds.append((embed, None))  # Store as tuple without view
                else:
                    print(f"[TOOLS] Unknown tool: {name}")
                    embed = discord.Embed(
                        title=f"{name} Error",
                        description=f"Unknown tool: {name}",
                        color=0xff0000
                    )
                    embeds.append((embed, None))

        return content, embeds

    def create_tool_embed(self, tool_name: str, result: Dict) -> Optional[discord.Embed]:
        """Create Discord embed for tool results"""
        if result.get('error'):
            return discord.Embed(
                title=f"{tool_name.title()} Error",
                description=result['error'],
                color=0xff0000
            )

        if tool_name == 'web_search':
            embed = discord.Embed(
                title="🔍 Search Results",
                description=f"Found {len(result.get('results', []))} results for: **{result.get('query', '')}**",
                color=0x00ff00
            )

            for i, item in enumerate(result.get('results', [])[:5], 1):
                embed.add_field(
                    name=f"{i}. {item.get('title', 'No title')[:100]}",
                    value=f"{item.get('snippet', 'No description')[:150]}\n[🔗 Visit]({item.get('link', '')})",
                    inline=False
                )
            return embed

        elif tool_name == 'web_scrape':
            embed = discord.Embed(
                title="📄 Scraped Content",
                description=f"Content from: {result.get('url', '')}",
                color=0x0099ff
            )

            content = result.get('content', '')[:1000]
            if len(result.get('content', '')) > 1000:
                content += "..."

            embed.add_field(
                name="Content",
                value=content,
                inline=False
            )
            return embed

        elif tool_name == 'calculate':
            embed = discord.Embed(
                title="🧮 Calculation",
                color=0xff9900
            )
            embed.add_field(
                name="Expression",
                value=f"`{result.get('expression', '')}`",
                inline=False
            )
            embed.add_field(
                name="Result",
                value=f"**{result.get('result', '')}**",
                inline=False
            )
            return embed

        elif tool_name == 'get_time':
            embed = discord.Embed(
                title="Current Time",
                color=0x9900ff
            )
            embed.add_field(
                name="Time",
                value=result.get('current_time', ''),
                inline=True
            )
            embed.add_field(
                name="Date", 
                value=result.get('current_date', ''),
                inline=True
            )
            embed.add_field(
                name="Timezone",
                value=result.get('timezone', 'Unknown'),
                inline=True
            )
            return embed

        elif tool_name == 'get_weather':
            embed = discord.Embed(
                title="Weather Report",
                description=f"Current weather for **{result.get('city', 'Unknown')}**",
                color=0x87ceeb
            )
            embed.add_field(
                name="Temperature",
                value=result.get('temperature', 'N/A'),
                inline=True
            )
            embed.add_field(
                name="Condition",
                value=result.get('condition', 'N/A'),
                inline=True
            )
            embed.add_field(
                name="Wind",
                value=f"{result.get('wind_speed', 'N/A')} @ {result.get('wind_direction', 'N/A')}",
                inline=True
            )
            return embed

        elif tool_name == 'news_search':
            embed = discord.Embed(
                title="News Results",
                description=f"Found {len(result.get('results', []))} news articles for: **{result.get('query', '')}**",
                color=0xff4444
            )

            for i, item in enumerate(result.get('results', [])[:5], 1):
                embed.add_field(
                    name=f"{i}. {item.get('title', 'No title')[:60]}...",
                    value=f"[Read more]({item.get('link', '#')})\n{item.get('snippet', 'No snippet')[:100]}...\n*{item.get('date', 'No date')}*",
                    inline=False
                )
            return embed

        return None

    def apply_anti_repetition(self, response: str, history: List[Dict]) -> str:
        """Apply anti-repetition logic"""
        if len(history) < 2:
            return response

        # Get recent responses
        recent_responses = [turn.get('assistant', '') for turn in history[-3:]]

        # Simple similarity check
        response_words = set(response.lower().split())

        for prev_response in recent_responses:
            prev_words = set(prev_response.lower().split())

            if len(response_words) > 0 and len(prev_words) > 0:
                overlap = len(response_words.intersection(prev_words))
                similarity = overlap / len(response_words.union(prev_words))

                if similarity > 0.7:  # High similarity
                    response += " (Let me know if you'd like me to approach this differently.)"
                    break

        return response


# =============================================================================
# COMMAND COG
# =============================================================================


        self.current_models = self.model_categories.get(model_type, models)
        self.total_pages = (len(self.current_models) + self.models_per_page - 1) // self.models_per_page

        self.setup_view()

    def categorize_main_models(self, models):
        """Categorize main chat/LLM models"""
        main_keywords = ['llama', 'qwen', 'mistral', 'gemma', 'phi', 'neural', 'instruct', 'chat', 'subsect']
        vision_keywords = ['vision', 'llava', 'moondream', 'granite3.2-vision']
        code_keywords = ['code', 'coder', 'deepseek', 'starcoder']
        embed_keywords = ['embed', 'nomic']

        main_models = []
        for model in models:
            model_lower = model.lower()
            # Exclude specialized models
            if any(kw in model_lower for kw in vision_keywords + code_keywords + embed_keywords):
                continue
            # Include main chat models
            if any(kw in model_lower for kw in main_keywords) or not any(kw in model_lower for kw in vision_keywords + code_keywords + embed_keywords):
                main_models.append(model)

        return main_models or models[:10]  # Fallback to first 10 if no matches

    def categorize_vision_models(self, models):
        """Categorize vision/multimodal models"""
        vision_keywords = ['vision', 'llava', 'moondream', 'granite3.2-vision', 'minicpm', 'cogvlm']
        return [m for m in models if any(kw in m.lower() for kw in vision_keywords)]

    def categorize_analysis_models(self, models):
        """Categorize analysis/reasoning models"""
        analysis_keywords = ['qwen', 'llama', 'mistral', 'gemma', 'phi', 'reasoning', 'think']
        vision_keywords = ['vision', 'llava', 'moondream']
        code_keywords = ['code', 'coder', 'deepseek', 'starcoder']

        analysis_models = []
        for model in models:
            model_lower = model.lower()
            # Include reasoning models but exclude vision/code
            if any(kw in model_lower for kw in analysis_keywords) and not any(kw in model_lower for kw in vision_keywords + code_keywords):
                analysis_models.append(model)

        return analysis_models or models[:5]  # Fallback

    def categorize_code_models(self, models):
        """Categorize code generation models"""
        code_keywords = ['code', 'coder', 'deepseek', 'starcoder', 'codellama', 'granite-code']
        return [m for m in models if any(kw in m.lower() for kw in code_keywords)]

    def categorize_embedding_models(self, models):
        """Categorize embedding models"""
        embed_keywords = ['embed', 'nomic', 'bge', 'e5']
        return [m for m in models if any(kw in m.lower() for kw in embed_keywords)]

    def setup_view(self):
        """Setup the view with model type selector, models dropdown, and navigation"""
        self.clear_items()

        # Model type selector (first row)
        type_options = []
        type_descriptions = {
            "main": f"Main Chat Models ({len(self.model_categories['main'])})",
            "vision": f"Vision/Multimodal ({len(self.model_categories['vision'])})",
            "analysis": f"Analysis/Reasoning ({len(self.model_categories['analysis'])})",
            "code": f"Code Generation ({len(self.model_categories['code'])})",
            "embedding": f"Embedding Models ({len(self.model_categories['embedding'])})"
        }

        for model_type, description in type_descriptions.items():
            if self.model_categories[model_type]:  # Only show if models exist
                type_options.append(discord.SelectOption(
                    label=description,
                    value=model_type,
                    default=(model_type == self.model_type)
                ))

        if type_options:
            type_select = discord.ui.Select(
                placeholder="Select model type...",
                options=type_options,
                row=0
            )
            type_select.callback = self.type_callback
            self.add_item(type_select)

        # Models dropdown (second row)
        start_idx = (self.page - 1) * self.models_per_page
        end_idx = min(start_idx + self.models_per_page, len(self.current_models))
        page_models = self.current_models[start_idx:end_idx]

        if page_models:
            model_options = []
            for model in page_models:
                # Get current model for this type
                if self.model_type == "main":
                    current_model = getattr(self.bot, 'current_model', '')
                elif self.model_type == "vision":
                    current_model = getattr(self.bot, 'vision_model', '')
                else:
                    current_model = getattr(self.bot, 'current_model', '')

                display_name = model if len(model) <= 90 else model[:87] + "..."
                model_options.append(discord.SelectOption(
                    label=display_name,
                    description="Currently selected" if model == current_model else "Select this model",
                    value=model,
                    default=(model == current_model)
                ))

            model_select = discord.ui.Select(
                placeholder=f"Choose {self.model_type} model... (Page {self.page}/{self.total_pages})",
                options=model_options,
                row=1
            )
            model_select.callback = self.model_callback
            self.add_item(model_select)

        # Add navigation buttons if multiple pages (third row)
        if self.total_pages > 1:
            # Previous button
            prev_button = discord.ui.Button(
                label="◀️ Previous",
                style=discord.ButtonStyle.grey,
                disabled=(self.page <= 1),
                row=2
            )
            prev_button.callback = self.previous_page
            self.add_item(prev_button)

            # Page indicator
            page_button = discord.ui.Button(
                label=f"Page {self.page}/{self.total_pages}",
                style=discord.ButtonStyle.blurple,
                disabled=True,
                row=2
            )
            self.add_item(page_button)

            # Next button
            next_button = discord.ui.Button(
                label="Next ▶️",
                style=discord.ButtonStyle.grey,
                disabled=(self.page >= self.total_pages),
                row=2
            )
            next_button.callback = self.next_page
            self.add_item(next_button)

    async def type_callback(self, interaction: discord.Interaction):
        """Handle model type selection"""
        selected_type = interaction.data['values'][0]

        # Update model type and reset to page 1
        self.model_type = selected_type
        self.current_models = self.model_categories.get(selected_type, self.all_models)
        self.total_pages = (len(self.current_models) + self.models_per_page - 1) // self.models_per_page
        self.page = 1

        # Rebuild view
        self.setup_view()
        embed = self.create_embed()

        await interaction.response.edit_message(embed=embed, view=self)

    async def model_callback(self, interaction: discord.Interaction):
        """Handle model selection"""
        selected_model = interaction.data['values'][0]

        # Update the appropriate model type on the bot
        if self.model_type == "main":
            self.bot.current_model = selected_model
        elif self.model_type == "vision":
            self.bot.vision_model = selected_model
        else:
            self.bot.current_model = selected_model

        embed = discord.Embed(
            title="✅ Model Updated",
            description=f"{self.model_type.title()} model changed to: `{selected_model}`",
            color=0x00ff00
        )

        await interaction.response.edit_message(embed=embed, view=None)
        print(f"[CONFIG] {self.model_type.title()} model changed to: {selected_model}")

    async def previous_page(self, interaction: discord.Interaction):
        """Handle previous page button"""
        if self.page > 1:
            self.page -= 1
            self.setup_view()
            embed = self.create_embed()
            await interaction.response.edit_message(embed=embed, view=self)

    async def next_page(self, interaction: discord.Interaction):
        """Handle next page button"""
        if self.page < self.total_pages:
            self.page += 1
            self.setup_view()
            embed = self.create_embed()
            await interaction.response.edit_message(embed=embed, view=self)

    def create_embed(self):
        """Create the model selection embed"""
        if self.model_type == "main":
            current_model = getattr(self.bot, 'current_model', 'Unknown')
        elif self.model_type == "vision":
            current_model = getattr(self.bot, 'vision_model', 'Unknown')
        else:
            current_model = getattr(self.bot, 'current_model', 'Unknown')

        start_idx = (self.page - 1) * self.models_per_page
        end_idx = min(start_idx + self.models_per_page, len(self.current_models))

        embed = discord.Embed(
            title=f"🤖 Select {self.model_type.title()} Model",
            description=f"**Current {self.model_type} model:** `{current_model}`\n\n"
                       f"**Available {self.model_type} models:** {len(self.current_models)}\n"
                       f"**Showing:** {start_idx + 1}-{end_idx} of {len(self.current_models)}\n\n"
                       f"1️⃣ Select model type from first dropdown\n"
                       f"2️⃣ Choose specific model from second dropdown",
            color=0x00ff00
        )

        # Add model type descriptions
        type_descriptions = {
            "main": "💬 Primary chat and conversation model",
            "vision": "👁️ Image analysis and multimodal tasks",
            "analysis": "🧠 User behavior analysis and insights",
            "code": "💻 Code generation and programming help",
            "embedding": "🔍 Text embedding and similarity search"
        }

        if self.model_type in type_descriptions:
            embed.add_field(
                name=f"{self.model_type.title()} Models",
                value=type_descriptions[self.model_type],
                inline=False
            )

        embed.set_footer(text="💡 Select a model type first, then choose from the available models")
        return embed


# =============================================================================
# COMMAND COG
# =============================================================================

class BotCommands(commands.Cog):
    """Bot commands organized as a Cog"""
    
    def __init__(self, bot):
        self.bot = bot
    
    @commands.command(name='bothelp')
    async def help_command(self, ctx):
        """Show available commands"""
        embed = discord.Embed(title="🎭 Hikari Commands", description="Available commands:", color=0x9966cc)
        embed.add_field(name="General", value="`!ping` - Test\n`!bothelp` - This help", inline=False)
        embed.add_field(name="Model", value="`!model` - Show models\n`!status` - Bot status", inline=False)
        embed.add_field(name="Memory", value="`!clear` - Clear history\n`!memory` - Memory stats", inline=False)
        embed.add_field(name="POML", value="`!poml` - POML status\n`!clearcache` - Clear POML cache\n`!purgeall` - 🚨 ADMIN: Complete memory purge", inline=False)
        embed.add_field(name="Mood Systems", value="`!mood` - Live conversation mood\n`!emotion` - Long-term emotional profile\n`!memories` - Show memories\n`!emotionstats` - System stats", inline=False)
        embed.add_field(name="Vector Tool Knowledge", value="`!toolsearch` - Search tool knowledge\n`!toolstats` - Tool knowledge stats", inline=False)
        await ctx.send(embed=embed)
    
    @commands.command(name='memory')
    async def memory_stats(self, ctx):
        """Show detailed user memory from sleep agent system"""
        user_id = str(ctx.author.id)
        
        # Get sleep agent memory if available
        if hasattr(self.bot, 'sleep_agent') and self.bot.sleep_agent:
            sleep_memory = self.bot.sleep_agent.get_user_memory_summary(user_id)
            
            # DEBUG: Check Discord bot's tracking vs sleep agent tracking
            discord_activity = self.bot.user_last_activity.get(user_id, 0)
            discord_messages = len(self.bot.user_conversation_history.get(user_id, []))
            
            embed = discord.Embed(title="🧠 Your Memory Profile", color=0x9966cc)
            embed.set_footer(text=f"Debug - Discord: {discord_messages} msgs, last: {discord_activity} | Sleep: {sleep_memory.get('message_count', 0)} msgs, last: {sleep_memory.get('last_activity', 0)}")
            
            # Stats row
            last_activity = sleep_memory.get('last_activity', 0)
            activity_text = f"<t:{int(last_activity)}:R>" if last_activity > 0 else "🆕 First time"
            
            embed.add_field(name="📊 Blocks", value=f"{sleep_memory.get('block_count', 0)}", inline=True)
            embed.add_field(name="⏰ Last Seen", value=activity_text, inline=True)
            embed.add_field(name="💬 Messages", value=f"{sleep_memory.get('message_count', 0)}", inline=True)
            
            # Show memory blocks with better formatting
            blocks = sleep_memory.get('blocks', {})
            for block_name, block_data in blocks.items():
                preview = block_data.get('value_preview', '').strip()
                if preview:
                    # Clean up the preview
                    if len(preview) > 150:
                        preview = preview[:150] + "..."
                    
                    # Better icons and formatting
                    icon = "💭" if block_name == "conversation_context" else "❤️" if block_name == "behavioral_patterns" else "⚙️" if block_name == "user_preferences" else "👤"
                    
                    embed.add_field(
                        name=f"{icon} {block_name.replace('_', ' ').title()}",
                        value=preview,
                        inline=False
                    )
            
            # Show FAISS memory stats with better formatting
            faiss_stats = sleep_memory.get('faiss_memory', {})
            vectors = faiss_stats.get('total_vectors', 0)
            index_type = faiss_stats.get('index_type', 'None')
            
            vector_status = "🔍 Active" if vectors > 0 else "💤 Empty"
            embed.add_field(
                name="🧮 Vector Memory",
                value=f"{vector_status}\n{vectors} vectors stored",
                inline=True
            )
            
            # Add emotional memory data if available
            if hasattr(self.bot, 'emotional_memory') and self.bot.emotional_memory:
                try:
                    print(f"[DEBUG] Trying to get emotional profile for user {user_id}")
                    profile = self.bot.emotional_memory.get_user_profile(user_id)
                    print(f"[DEBUG] Got profile: {profile.username if profile else 'None'}")
                    
                    embed.add_field(name="\u200b", value="\u200b", inline=False)  # Spacer
                    embed.add_field(
                        name="💝 Emotional Profile",
                        value=f"**Mood:** {profile.current_mood} ({profile.mood_points:.1f}) | **Trust:** {profile.trust_score*100:.0f}%\n**Memories:** {len(profile.memories)} stored",
                        inline=False
                    )
                    
                    # Show top personality traits
                    traits = []
                    for trait, value in profile.personality_traits.items():
                        if value > 0.7:  # Only show strong traits (assuming 0-1 scale)
                            traits.append(f"{trait.title()}: {value*100:.0f}%")
                    
                    if traits:
                        embed.add_field(
                            name="🎭 Strong Traits", 
                            value=" • ".join(traits[:3]),  # Top 3 traits
                            inline=False
                        )
                    
                except Exception as e:
                    embed.add_field(name="💝 Emotional Profile", value=f"❌ Error: {str(e)}", inline=False)
        else:
            embed = discord.Embed(title="🧠 Memory System", description="❌ Sleep Agent not available", color=0xff6666)
            
        await ctx.send(embed=embed)
        
    @commands.command(name='allmemories')
    async def detailed_memories(self, ctx):
        """Show paginated detailed memories from all systems with UI buttons"""
        user_id = str(ctx.author.id)
        
        # Collect all memories from different systems
        all_memories = []
        
        # Get emotional memories
        if hasattr(self.bot, 'emotional_memory') and self.bot.emotional_memory:
            try:
                profile = self.bot.emotional_memory.get_user_profile(user_id)
                if profile and hasattr(profile, 'memories') and profile.memories:
                    for memory in profile.memories[:50]:  # Limit to recent 50
                        # Handle both dict and object access patterns
                        content = memory.get('content', '') if isinstance(memory, dict) else getattr(memory, 'content', '')
                        memory_type = memory.get('memory_type', 'UNKNOWN') if isinstance(memory, dict) else getattr(memory, 'memory_type', 'UNKNOWN')
                        score = memory.get('importance_score', 0) if isinstance(memory, dict) else getattr(memory, 'importance_score', 0)
                        timestamp = memory.get('timestamp', 0) if isinstance(memory, dict) else getattr(memory, 'timestamp', 0)
                        
                        all_memories.append({
                            'source': '💝 Emotional',
                            'content': str(content)[:100] + ('...' if len(str(content)) > 100 else ''),
                            'type': str(memory_type),
                            'score': float(score) if score else 0.0,
                            'timestamp': float(timestamp) if timestamp else 0.0
                        })
            except Exception as e:
                print(f"[DEBUG] Error getting emotional memories: {e}")
        
        # Get sleep agent memories
        if hasattr(self.bot, 'sleep_agent') and self.bot.sleep_agent:
            try:
                sleep_memory = self.bot.sleep_agent.get_user_memory_summary(user_id)
                blocks = sleep_memory.get('blocks', {})
                for block_name, block_data in blocks.items():
                    if block_data.get('value_preview'):
                        all_memories.append({
                            'source': '🧠 Sleep Agent',
                            'content': block_data['value_preview'][:100] + ('...' if len(block_data['value_preview']) > 100 else ''),
                            'type': block_name.replace('_', ' ').title(),
                            'score': 1.0,
                            'timestamp': block_data.get('last_updated', 0)
                        })
            except Exception as e:
                print(f"[DEBUG] Error getting sleep agent memories: {e}")
        
        # Sort by importance score and timestamp
        all_memories.sort(key=lambda x: (x['score'], x['timestamp']), reverse=True)
        
        if not all_memories:
            embed = discord.Embed(title="📚 Detailed Memories", description="No memories found", color=0xff6666)
            await ctx.send(embed=embed)
            return
        
        # Use the bot's built-in pagination system
        def memory_formatter(memory, idx):
            timestamp_text = f"<t:{int(memory['timestamp'])}:R>" if memory['timestamp'] > 0 else "Unknown time"
            return f"**{memory['source']} - {memory['type']}** ({memory['score']:.2f})\n{memory['content']}\n*{timestamp_text}*"
        
        embed, view = self.bot.create_paginated_embed(
            title="📚 Detailed Memories",
            items=all_memories,
            items_per_page=8,
            color=0x9966cc,
            item_formatter=memory_formatter
        )
        
        await ctx.send(embed=embed, view=view)
    
    @commands.command()
    async def ping(self, ctx):
        """Simple test command"""
        await ctx.send("Pong!")

    @commands.command(name='model')
    @commands.check_any(commands.is_owner(), commands.has_permissions(administrator=True))
    async def model_select(self, ctx):
        """Select the main LLM model from available Ollama models with dropdown interface"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:11434/api/tags') as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [model['name'] for model in data.get('models', [])]
                    else:
                        await ctx.send("❌ Failed to connect to Ollama. Make sure it's running.")
                        return

            if not models:
                await ctx.send("❌ No Ollama models found. Please install some models first.")
                return

            # Create dropdown view
            view = ModelSelectView(models, self.bot)
            embed = view.create_embed()

            await ctx.send(embed=embed, view=view)

        except Exception as e:
            print(f"❌ Error in model command: {e}")
            await ctx.send("❌ Error getting model list. Make sure Ollama is running.")

    @commands.command(name='status')
    async def status_command(self, ctx):
        """Show bot status"""
        embed = discord.Embed(title="🤖 Bot Status", color=0x0099ff)
        embed.add_field(name="Current Model", value=f"`{self.bot.current_model}`", inline=False)
        embed.add_field(name="Vision Model", value=f"`{self.bot.vision_model}`", inline=False)
        
        # AI Intent Classification status
        if self.bot.intent_classifier:
            embed.add_field(name="AI Intent Classification", value="✅ Enabled (GPU/CPU)", inline=False)
            
            # Check if torch is available to show GPU status
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    embed.add_field(name="GPU Status", value=f"✅ CUDA: {gpu_name}", inline=False)
                else:
                    embed.add_field(name="GPU Status", value="❌ CPU Only", inline=False)
            except ImportError:
                pass
        else:
            embed.add_field(name="AI Intent Classification", value="❌ Disabled (fallback mode)", inline=False)
            
        await ctx.send(embed=embed)

    @commands.command(name='clear')
    async def clear_history(self, ctx):
        """Clear conversation history using modern memory system"""
        channel_id = str(ctx.channel.id)
        
        # Get stats before clearing
        stats = self.bot.memory.get_memory_stats(channel_id)
        
        # Clear memory
        self.bot.memory.clear_channel_history(channel_id)
        
        embed = discord.Embed(title="🧹 Memory Cleared", color=0x00ff00)
        embed.add_field(name="Cleared", value=f"{stats['total_messages']} messages", inline=True)
        embed.add_field(name="User Messages", value=f"{stats['user_messages']}", inline=True)
        embed.add_field(name="Bot Messages", value=f"{stats['bot_messages']}", inline=True)
        
        await ctx.send(embed=embed)

    @commands.command(name='mood')
    async def check_mood(self, ctx, user: discord.Member = None):
        """Check mood points for a user"""
        target_user = user or ctx.author
        user_id = str(target_user.id)
        mood_points = self.bot.get_user_mood(user_id)
        tone = self.bot.get_tone_from_mood(mood_points)

        embed = discord.Embed(
            title=f"🎭 {target_user.display_name}'s Live Conversation Mood",
            description="⚡ Real-time mood tracking (-10 to +10)",
            color=0xff69b4
        )

        embed.add_field(
            name="💝 Mood Points",
            value=f"{mood_points:.1f}/20 (range: -10 to +10)",
            inline=True
        )

        embed.add_field(
            name="😊 Current Tone",
            value=tone.replace('-', ' ').title(),
            inline=True
        )

        # Add mood description - MUST match POML personality.poml tone ranges
        mood_descriptions = {
            "dere-hot": "💕 Overflowing sweetness, sparkly",
            "cheerful": "😊 Flirty and warm, teasing",
            "soft-dere": "😌 Chill and slightly flirty",
            "neutral": "😐 Chill but sassy default mode",
            "classic-tsun": "😤 Flustered denials, tsundere",
            "grumpy-tsun": "😠 Sassy and snappy, annoyed", 
            "explosive-tsun": "💥 Very mad tsundere outbursts!"
        }

        embed.add_field(
            name="📝 Description",
            value=mood_descriptions.get(tone, "🤔 Neutral"),
            inline=False
        )

        await ctx.send(embed=embed)

    @commands.command(name='poml')
    async def poml_status(self, ctx):
        """Show POML status and templates"""
        embed = discord.Embed(
            title="🎭 POML Status",
            color=0xff69b4
        )

        if POML_AVAILABLE:
            embed.add_field(
                name="✅ POML Available",
                value="Microsoft's Prompt Orchestration Markup Language is active",
                inline=False
            )

            embed.add_field(
                name="📁 Loaded Templates",
                value=f"{len(self.bot.poml_templates)} templates: {', '.join(self.bot.poml_templates.keys())}",
                inline=False
            )

            embed.add_field(
                name="🎯 Features",
                value="• Structured JSON responses\n• Dynamic mood system\n• Context-aware prompts\n• Tool schema validation",
                inline=False
            )
            
            # Add cache statistics
            cache_stats = self.bot.poml_cache.get_cache_stats()
            embed.add_field(
                name="🚀 Cache Performance",
                value=f"• Cached Results: {cache_stats['cached_results']}\n"
                      f"• Cache Hits: {cache_stats['cache_hits']}\n"
                      f"• Cache Misses: {cache_stats['cache_misses']}\n"
                      f"• Hit Rate: {cache_stats['hit_rate']:.1%}",
                inline=False
            )
            
            # Add performance impact info
            embed.add_field(
                name="⚡ Performance Impact",
                value="• Cache HIT: ~0.1s (eliminates 1.2-1.4s delay)\n"
                      f"• Cache MISS: ~1.2-1.4s (fallback processing)\n"
                      f"• Total time saved: ~{cache_stats['cache_hits'] * 1.3:.1f}s",
                inline=False
            )
        else:
            embed.add_field(
                name="❌ POML Not Available",
                value="Install with: `pip install poml`",
                inline=False
            )

        await ctx.send(embed=embed)

    @commands.command(name='vision_model')
    @commands.check_any(commands.is_owner(), commands.has_permissions(administrator=True))
    async def change_vision_model(self, ctx, *, model_name: str = None):
        """Change the vision analysis model"""
        if not model_name:
            embed = discord.Embed(
                title="👁️ Current Vision Model",
                description=f"Current vision model: `{self.bot.vision_model}`",
                color=0x00ff00
            )
            await ctx.send(embed=embed)
            return

        try:
            # Test the vision model with a simple prompt
            from ollama import AsyncClient
            test_client = AsyncClient()

            # Create a simple test (no image needed for model validation)
            test_response = await test_client.chat(
                model=model_name,
                messages=[{"role": "user", "content": "Hello, can you see images?"}]
            )

            self.bot.vision_model = model_name

            embed = discord.Embed(
                title="👁️ Vision Model Changed",
                description=f"Successfully changed vision model to: `{model_name}`",
                color=0x00ff00
            )
            await ctx.send(embed=embed)
            print(f"[CONFIG] Vision model changed to: {model_name}")

        except Exception as e:
            embed = discord.Embed(
                title="Vision Model Test Failed",
                description=f"Failed to test vision model `{model_name}`: {str(e)}",
                color=0xff0000
            )
            await ctx.send(embed=embed)

    @commands.command(name='updatestatus')
    @commands.check_any(commands.is_owner(), commands.has_permissions(administrator=True))
    async def update_status_command(self, ctx):
        """Manually update the bot's status"""
        try:
            await ctx.send("Updating status...")
            await self.bot.update_dynamic_status()
            await ctx.send("Status updated!")
        except Exception as e:
            await ctx.send(f"Failed to update status: {str(e)}")

    @commands.command(name='clearcache')
    @commands.check_any(commands.is_owner(), commands.has_permissions(administrator=True))
    async def clear_poml_cache(self, ctx):
        """Clear POML template cache"""
        try:
            cache_stats = self.bot.poml_cache.get_cache_stats()
            self.bot.poml_cache.clear_cache()
            
            embed = discord.Embed(
                title="🧹 POML Cache Cleared",
                description="Template cache has been cleared and will be rebuilt on next use",
                color=0x00ff00
            )
            embed.add_field(
                name="Previous Cache Stats",
                value=f"• Cached Templates: {cache_stats['cached_results']}\n"
                      f"• Cache Hits: {cache_stats['cache_hits']}\n"
                      f"• Cache Misses: {cache_stats['cache_misses']}\n"
                      f"• Hit Rate: {cache_stats['hit_rate']:.1%}",
                inline=False
            )
            
            await ctx.send(embed=embed)
        except Exception as e:
            await ctx.send(f"Failed to clear cache: {str(e)}")

    @commands.command(name='purgeall')
    @commands.check_any(commands.is_owner(), commands.has_permissions(administrator=True))
    async def purge_all_memory(self, ctx, target: str = None):
        """🚨 ADMIN ONLY: Complete memory purge - clears ALL memory systems"""
        try:
            if target is None:
                # Purge command invoker by default
                user_id = str(ctx.author.id)
                scope = f"user {ctx.author.mention}"
            elif target.lower() == "all":
                # Nuclear option - purge entire bot
                scope = "**ENTIRE BOT MEMORY**"
                
                # Require explicit confirmation for nuclear option
                embed = discord.Embed(
                    title="⚠️ NUCLEAR MEMORY PURGE WARNING",
                    description="This will **PERMANENTLY DELETE ALL MEMORY** for every user!\n\nType `CONFIRM_NUCLEAR_PURGE` within 30 seconds to proceed.",
                    color=0xFF0000
                )
                await ctx.send(embed=embed)
                
                def check(m):
                    return m.author == ctx.author and m.channel == ctx.channel and m.content == "CONFIRM_NUCLEAR_PURGE"
                
                try:
                    await self.bot.wait_for('message', check=check, timeout=30)
                except:
                    await ctx.send("❌ Nuclear purge cancelled - no confirmation received.")
                    return
                    
                user_id = "all"
            else:
                # Purge specific user ID
                user_id = target
                scope = f"user ID `{user_id}`"
            
            purge_results = []
            
            # 1. SLEEP AGENT MEMORY SYSTEM
            if hasattr(self.bot, 'sleep_agent') and self.bot.sleep_agent:
                try:
                    if user_id == "all":
                        # Nuclear: Clear all sleep agent memory
                        self.bot.sleep_agent.memory_manager.blocks.clear()
                        self.bot.sleep_agent.last_activity.clear()
                        self.bot.sleep_agent.message_counts.clear()
                        if hasattr(self.bot.sleep_agent, 'faiss_memory') and self.bot.sleep_agent.faiss_memory:
                            self.bot.sleep_agent.faiss_memory.user_indices.clear()
                            self.bot.sleep_agent.faiss_memory.user_metadata.clear()
                        self.bot.sleep_agent.memory_manager._save_memory()
                        purge_results.append("✅ Sleep Agent Memory: ALL users purged")
                    else:
                        # Target specific user
                        self.bot.sleep_agent.reset_user_memory(user_id)
                        if hasattr(self.bot.sleep_agent, 'faiss_memory') and self.bot.sleep_agent.faiss_memory:
                            self.bot.sleep_agent.faiss_memory.user_indices.pop(user_id, None)
                            self.bot.sleep_agent.faiss_memory.user_metadata.pop(user_id, None)
                        self.bot.sleep_agent.memory_manager._save_memory()
                        purge_results.append(f"✅ Sleep Agent Memory: User {user_id} purged")
                except Exception as e:
                    purge_results.append(f"❌ Sleep Agent Memory: {str(e)}")
            else:
                purge_results.append("ℹ️ Sleep Agent Memory: Not available")
            
            # 2. EMOTIONAL MEMORY SYSTEM  
            if hasattr(self.bot, 'emotional_memory') and self.bot.emotional_memory:
                try:
                    if user_id == "all":
                        # Nuclear: Clear all emotional memory
                        self.bot.emotional_memory.user_profiles.clear()
                        self.bot.emotional_memory.save_all_profiles()
                        purge_results.append("✅ Emotional Memory: ALL users purged")
                    else:
                        # Target specific user
                        if user_id in self.bot.emotional_memory.user_profiles:
                            del self.bot.emotional_memory.user_profiles[user_id]
                            self.bot.emotional_memory.save_all_profiles()
                            purge_results.append(f"✅ Emotional Memory: User {user_id} purged")
                        else:
                            purge_results.append(f"ℹ️ Emotional Memory: User {user_id} not found")
                except Exception as e:
                    purge_results.append(f"❌ Emotional Memory: {str(e)}")
            else:
                purge_results.append("ℹ️ Emotional Memory: Not available")
            
            # 3. CONVERSATION MEMORY SYSTEM
            if hasattr(self.bot, 'memory') and self.bot.memory:
                try:
                    if user_id == "all":
                        # Nuclear: Clear all conversation memory
                        self.bot.memory.user_contexts.clear()
                        if hasattr(self.bot.memory, 'conversation_summaries'):
                            self.bot.memory.conversation_summaries.clear()
                        purge_results.append("✅ Conversation Memory: ALL users purged")
                    else:
                        # Target specific user
                        user_id_int = int(user_id)
                        if user_id_int in self.bot.memory.user_contexts:
                            del self.bot.memory.user_contexts[user_id_int]
                        if hasattr(self.bot.memory, 'conversation_summaries') and user_id_int in self.bot.memory.conversation_summaries:
                            del self.bot.memory.conversation_summaries[user_id_int]
                        purge_results.append(f"✅ Conversation Memory: User {user_id} purged")
                except Exception as e:
                    purge_results.append(f"❌ Conversation Memory: {str(e)}")
            else:
                purge_results.append("ℹ️ Conversation Memory: Not available")
            
            # 4. DISCORD BOT TRACKING DATA
            try:
                if user_id == "all":
                    # Nuclear: Clear all bot tracking
                    if hasattr(self.bot, 'user_conversation_history'):
                        self.bot.user_conversation_history.clear()
                    if hasattr(self.bot, 'user_last_activity'):
                        self.bot.user_last_activity.clear()
                    if hasattr(self.bot, 'mood_points'):
                        self.bot.mood_points.clear()
                    purge_results.append("✅ Bot Tracking Data: ALL users purged")
                else:
                    # Target specific user
                    user_id_int = int(user_id)
                    if hasattr(self.bot, 'user_conversation_history'):
                        self.bot.user_conversation_history.pop(user_id_int, None)
                    if hasattr(self.bot, 'user_last_activity'):
                        self.bot.user_last_activity.pop(user_id_int, None)
                    if hasattr(self.bot, 'mood_points'):
                        self.bot.mood_points.pop(user_id, None)  # mood_points uses string keys
                    purge_results.append(f"✅ Bot Tracking Data: User {user_id} purged")
            except Exception as e:
                purge_results.append(f"❌ Bot Tracking Data: {str(e)}")
            
            # 5. POML CACHE SYSTEM
            try:
                if hasattr(self.bot, 'poml_cache'):
                    old_count = len(self.bot.poml_cache.compiled_results)
                    self.bot.poml_cache.compiled_results.clear()
                    self.bot.poml_cache.cache_hits = 0
                    self.bot.poml_cache.cache_misses = 0
                    purge_results.append(f"✅ POML Cache: {old_count} entries cleared")
                else:
                    purge_results.append("ℹ️ POML Cache: Not available")
            except Exception as e:
                purge_results.append(f"❌ POML Cache: {str(e)}")
            
            # 6. SAVE PERSISTENT STATE
            try:
                self.bot.save_persistent_state()
                purge_results.append("✅ Persistent State: Saved to disk")
            except Exception as e:
                purge_results.append(f"❌ Persistent State: {str(e)}")
            
            # Send results
            embed = discord.Embed(
                title="🗑️ COMPLETE MEMORY PURGE",
                description=f"**Target:** {scope}\n\n**Results:**\n" + "\n".join(purge_results),
                color=0xFF6600 if user_id == "all" else 0x00FF00
            )
            
            if user_id == "all":
                embed.add_field(
                    name="⚠️ NUCLEAR PURGE COMPLETE", 
                    value="All memory systems have been completely wiped. The bot will start fresh for all users.",
                    inline=False
                )
            else:
                embed.add_field(
                    name="ℹ️ User Purge Complete", 
                    value="All memory for the specified user has been permanently deleted from all systems.",
                    inline=False
                )
                
            await ctx.send(embed=embed)
            
        except Exception as e:
            await ctx.send(f"❌ Error during memory purge: {str(e)}")
            print(f"[PURGE ERROR] {e}")
            import traceback
            traceback.print_exc()

    @commands.command(name='emotion')
    async def emotional_profile(self, ctx, user: discord.Member = None):
        """Show emotional profile for a user (or yourself if no user specified)"""
        if not self.bot.emotional_memory:
            await ctx.send("❌ Emotional Memory System is not available")
            return
            
        target_user = user or ctx.author
        user_id = str(target_user.id)
        
        try:
            # Recalculate stats to ensure they're up to date
            self.bot.emotional_memory.recalculate_user_stats(user_id)
            profile = self.bot.emotional_memory.get_user_profile(user_id)
            
            embed = discord.Embed(
                title=f"💝 Emotional Memory Profile: {profile.username}",
                description="📊 Long-term emotional memory system data",
                color=0xff69b4
            )
            
            # Basic info
            embed.add_field(
                name="Current Mood",
                value=f"😊 {profile.current_mood} ({profile.mood_points:.1f})",
                inline=True
            )
            
            embed.add_field(
                name="Relationship Level",
                value=f"🤝 {profile.relationship_level.replace('_', ' ').title()}",
                inline=True
            )
            
            embed.add_field(
                name="Trust Score",
                value=f"🔒 {profile.trust_score:.1%}",
                inline=True
            )
            
            embed.add_field(
                name="Familiarity",
                value=f"👥 {profile.familiarity_level:.1%}",
                inline=True
            )
            
            embed.add_field(
                name="Conversations",
                value=f"💬 {profile.conversation_count}",
                inline=True
            )
            
            embed.add_field(
                name="Last Interaction",
                value=f"⏰ {datetime.fromtimestamp(profile.last_interaction).strftime('%Y-%m-%d %H:%M')}",
                inline=True
            )
            
            # Personality traits
            traits_text = "\n".join([f"• {trait.replace('_', ' ').title()}: {value:.1%}" 
                                   for trait, value in profile.personality_traits.items()])
            embed.add_field(
                name="Personality Traits",
                value=traits_text,
                inline=False
            )
            
            # Memory stats
            embed.add_field(
                name="Memories Stored",
                value=f"🧠 {len(profile.memories)} memories",
                inline=False
            )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            await ctx.send(f"❌ Failed to get emotional profile: {str(e)}")

    @commands.command(name='memories')
    async def show_memories(self, ctx, user: discord.Member = None, query: str = ""):
        """Show memories for a user (or yourself if no user specified)"""
        if not self.bot.emotional_memory:
            await ctx.send("❌ Emotional Memory System is not available")
            return
            
        target_user = user or ctx.author
        user_id = str(target_user.id)
        
        try:
            if query:
                memories = self.bot.emotional_memory.get_relevant_memories(user_id, query, limit=10)
                title = f"🔍 Memories for '{query}'"
            else:
                profile = self.bot.emotional_memory.get_user_profile(user_id)
                memories = profile.memories[-10:]  # Last 10 memories
                title = f"🧠 Recent Memories"
            
            if not memories:
                await ctx.send(f"❌ No memories found for {target_user.display_name}")
                return
            
            # Create paginated embed
            embed, view = self.bot.create_paginated_embed(
                title=title,
                items=memories,
                items_per_page=5,
                color=0xff69b4,
                item_formatter=lambda memory, idx: 
                    f"**{memory.memory_type}** ({memory.importance_score:.1%})\n"
                    f"💭 {memory.content[:100]}{'...' if len(memory.content) > 100 else ''}\n"
                    f"😊 {memory.emotional_context} • {datetime.fromtimestamp(memory.timestamp).strftime('%m/%d %H:%M')}"
            )
            
            if view:
                await ctx.send(embed=embed, view=view)
            else:
                await ctx.send(embed=embed)
                
        except Exception as e:
            await ctx.send(f"❌ Failed to get memories: {str(e)}")
    
    @commands.command(name='fixstats')
    @commands.is_owner()
    async def fix_emotional_stats(self, ctx, user: discord.Member = None):
        """Recalculate emotional memory stats for a user (owner only) - fixes relationship levels"""
        if not self.bot.emotional_memory:
            await ctx.send("❌ Emotional Memory System is not available")
            return
            
        target_user = user or ctx.author
        user_id = str(target_user.id)
        
        try:
            # Get stats before fix
            profile_before = self.bot.emotional_memory.get_user_profile(user_id)
            old_relationship = profile_before.relationship_level
            old_familiarity = profile_before.familiarity_level
            old_trust = profile_before.trust_score
            
            # Apply fix
            self.bot.emotional_memory.recalculate_user_stats(user_id)
            
            # Get stats after fix
            profile_after = self.bot.emotional_memory.get_user_profile(user_id)
            
            embed = discord.Embed(
                title=f"🔧 Fixed Stats for {target_user.display_name}",
                color=0x00ff00
            )
            
            embed.add_field(
                name="📊 Changes Made",
                value=f"**Relationship:** {old_relationship} → {profile_after.relationship_level}\n"
                      f"**Familiarity:** {old_familiarity:.1%} → {profile_after.familiarity_level:.1%}\n"
                      f"**Trust:** {old_trust:.1%} → {profile_after.trust_score:.1%}\n"
                      f"**Memories:** {len(profile_after.memories)} messages",
                inline=False
            )
            
            embed.add_field(
                name="🎯 New Thresholds",
                value="**Acquaintance:** 5+ messages\n"
                      "**Friend:** 15+ messages\n"
                      "**Close Friend:** 30+ messages",
                inline=False
            )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            await ctx.send(f"❌ Failed to recalculate stats: {str(e)}")

    @commands.command(name='emotionstats')
    @commands.check_any(commands.is_owner(), commands.has_permissions(administrator=True))
    async def emotional_system_stats(self, ctx):
        """Show emotional memory system statistics"""
        try:
            embed = discord.Embed(
                title="💝 Emotional Memory System Stats",
                color=0xff69b4
            )
            
            # Check if we have the new sleep agent system
            if hasattr(self.bot, 'sleep_agent') and self.bot.sleep_agent:
                stats = self.bot.sleep_agent.get_system_status()
                
                embed.add_field(
                    name="Total Users",
                    value=f"👥 {stats['total_users']}",
                    inline=True
                )
                
                embed.add_field(
                    name="Total Memory Blocks",
                    value=f"🧠 {stats['total_memory_blocks']}",
                    inline=True
                )
                
                embed.add_field(
                    name="Active Users",
                    value=f"🟢 {stats['active_users']}",
                    inline=True
                )
                
                embed.add_field(
                    name="Memory File",
                    value=f"📁 {stats['memory_file']}",
                    inline=False
                )
                
                embed.add_field(
                    name="Configuration", 
                    value=f"📊 Model: {stats['config']['model']}\n" +
                          f"🔄 Trigger: {stats['config']['trigger_after_messages']} messages or {stats['config']['trigger_after_idle_minutes']} min idle\n" +
                          f"💾 FAISS: {'✅' if stats['config']['enable_faiss'] else '❌'}\n" +
                          f"🧠 Stream Thinking: {'✅' if stats['config']['stream_thinking'] else '❌'}",
                    inline=False
                )
                
            # Fallback to old emotional memory system
            elif self.bot.emotional_memory:
                stats = self.bot.emotional_memory.get_system_stats()
                
                embed.add_field(
                    name="Total Users",
                    value=f"👥 {stats['total_users']}",
                    inline=True
                )
                
                embed.add_field(
                    name="Total Memories",
                    value=f"🧠 {stats['total_memories']}",
                    inline=True
                )
                
                embed.add_field(
                    name="Storage Directory",
                    value=f"📁 {stats['storage_directory']}",
                    inline=False
                )
                
                embed.add_field(
                    name="Last Save",
                    value=f"⏰ {datetime.fromtimestamp(stats['last_save']).strftime('%Y-%m-%d %H:%M:%S')}",
                    inline=False
                )
            else:
                await ctx.send("❌ No memory system available")
                return
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            await ctx.send(f"❌ Failed to get system stats: {str(e)}")

    
# =============================================================================
# MAIN EXECUTION
# =============================================================================

async def main():
    """Main execution function"""
    # Get Discord token from environment
    token = os.getenv('DISCORD_BOT_TOKEN')

    # Debug: Show token status (first few characters only for security)
    if token:
        print(f"[OK] Discord token loaded: {token[:10]}...")
    else:
        print("[ERROR] DISCORD_BOT_TOKEN environment variable not set!")
        print("🔍 Available environment variables:")
        for key in os.environ:
            if 'DISCORD' in key.upper() or 'TOKEN' in key.upper():
                print(f"   {key}: {str(os.environ[key])[:10]}...")
        return

    # Create and run bot
    bot = OptimizedDiscordBot()

    try:
        await bot.start(token)
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"[ERROR] Bot error: {e}")
    finally:
        if bot.ollama.session:
            await bot.ollama.close_session()

if __name__ == "__main__":
    print("[INIT] Starting Optimized Discord Bot...")
    print("[INFO] Features: Ollama optimization, BPE tokenization, full tool suite, anti-repetition")
    asyncio.run(main())
  