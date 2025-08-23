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
from typing import Dict, List, Optional, Any

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
    print("[OK] POML available - Advanced prompt orchestration enabled")
except ImportError:
    POML_AVAILABLE = False
    print("[WARNING] POML not installed - Using basic prompts (pip install poml to enable)")

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
        headers = {
            "X-API-KEY": "d03c7ebd4196bf9562d419973ae064bb959dde5b",
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
        headers = {
            "X-API-KEY": "d03c7ebd4196bf9562d419973ae064bb959dde5b",
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
        headers = {
            "X-API-KEY": "d03c7ebd4196bf9562d419973ae064bb959dde5b",
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
            # Use official Ollama chat API for vision analysis (like examples/multimodal-chat.py)
            response = await ollama_client.chat(
                model=vision_model,
                messages=[{
                    'role': 'user',
                    'content': analysis_prompt,
                    'images': [image_data]
                }],
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
            # Use official Ollama chat API for vision analysis (like examples/multimodal-chat.py)
            response = await ollama_client.chat(
                model=vision_model,
                messages=[{
                    'role': 'user',
                    'content': analysis_prompt,
                    'images': [image_data]
                }],
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
            result.update({"success": True, "channel_name": channel.name})
            
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
                    "channel": message_obj.channel.name,
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
                    "channel_name": channel.name
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
                    "channel_name": channel.name,
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

class ModelSelectView(discord.ui.View):
    """Enhanced model selection with type switching and pagination"""

    def __init__(self, models, bot, model_type="main", page=1):
        super().__init__(timeout=120)  # Extended timeout for model switching
        self.bot = bot
        self.all_models = models
        self.model_type = model_type
        self.page = page
        self.models_per_page = 20  # Leave room for type selector

        # Model categorization based on common patterns
        self.model_categories = {
            "main": self.categorize_main_models(models),
            "vision": self.categorize_vision_models(models),
            "analysis": self.categorize_analysis_models(models),
            "code": self.categorize_code_models(models),
            "embedding": self.categorize_embedding_models(models)
        }

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
            current_model = getattr(self.bot, 'current_model', '') if self.model_type == "main" else getattr(self.bot, 'vision_model', '')

            for model in page_models:
                model_options.append(discord.SelectOption(
                    label=model[:100],  # Discord limit
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

        # Navigation buttons (third row)
        if self.total_pages > 1:
            prev_button = discord.ui.Button(
                label="◀ Previous",
                style=discord.ButtonStyle.secondary,
                disabled=(self.page <= 1),
                row=2
            )
            prev_button.callback = self.prev_callback
            self.add_item(prev_button)

            next_button = discord.ui.Button(
                label="Next ▶",
                style=discord.ButtonStyle.secondary,
                disabled=(self.page >= self.total_pages),
                row=2
            )
            next_button.callback = self.next_callback
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

        embed = discord.Embed(
            title="✅ Model Updated",
            description=f"{self.model_type.title()} model changed to: `{selected_model}`",
            color=0x00ff00
        )

        await interaction.response.edit_message(embed=embed, view=None)
        print(f"[CONFIG] {self.model_type.title()} model changed to: {selected_model}")

    async def prev_callback(self, interaction: discord.Interaction):
        """Handle previous page"""
        if self.page > 1:
            self.page -= 1
            self.setup_view()
            embed = self.create_embed()
            await interaction.response.edit_message(embed=embed, view=self)

    async def next_callback(self, interaction: discord.Interaction):
        """Handle next page"""
        if self.page < self.total_pages:
            self.page += 1
            self.setup_view()
            embed = self.create_embed()
            await interaction.response.edit_message(embed=embed, view=self)

    def create_embed(self):
        """Create the model selection embed"""
        current_model = getattr(self.bot, 'current_model', 'Unknown') if self.model_type == "main" else getattr(self.bot, 'vision_model', 'Unknown')

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
            "main": "💬 General chat and conversation models",
            "vision": "👁️ Image analysis and multimodal models",
            "analysis": "🧠 Reasoning and analytical models",
            "code": "💻 Code generation and programming models",
            "embedding": "🔗 Text embedding and similarity models"
        }

        if self.model_type in type_descriptions:
            embed.add_field(
                name=f"{self.model_type.title()} Models",
                value=type_descriptions[self.model_type],
                inline=False
            )

        return embed

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
        self.conversation_history = {}  # Per-channel history

        # POML template management
        self.poml_templates = {}
        self.mood_points = {}  # Per-user mood tracking
        self.load_poml_templates()
        
        print("[INIT] Optimized Discord Bot initialized")
        print(f"[CONFIG] KV Cache: {os.environ.get('OLLAMA_KV_CACHE_TYPE')}")
        print(f"[CONFIG] Flash Attention: {os.environ.get('OLLAMA_FLASH_ATTENTION')}")
    
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
        """Load POML templates if available"""
        if not POML_AVAILABLE:
            return

        template_files = {
            'personality': 'templates/personality.poml',
            'tools': 'templates/tools.poml',
            'mood_system': 'templates/mood_system.poml'
        }

        for name, filepath in template_files.items():
            try:
                if os.path.exists(filepath):
                    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                        # Clean up any problematic Unicode characters
                        content = content.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                        self.poml_templates[name] = content
                    print(f"[OK] Loaded POML template: {name}")
                else:
                    print(f"[WARNING] POML template not found: {filepath}")
            except Exception as e:
                print(f"[ERROR] Error loading POML template {name}: {e}")

    def get_user_mood(self, user_id: str) -> int:
        """Get user's current mood points (-10 to 10)"""
        return self.mood_points.get(user_id, 0)

    def adjust_user_mood(self, user_id: str, user_input: str) -> int:
        """Adjust user mood based on input and return new mood"""
        current_mood = self.get_user_mood(user_id)

        # Simple mood adjustment logic
        positive_words = ['thanks', 'thank you', 'awesome', 'great', 'love', 'amazing']
        negative_words = ['stupid', 'dumb', 'hate', 'annoying', 'bad', 'terrible']

        input_lower = user_input.lower()

        old_mood = current_mood
        
        if any(word in input_lower for word in positive_words):
            current_mood = min(10, current_mood + 1)
        elif any(word in input_lower for word in negative_words):
            current_mood = max(-10, current_mood - 1)

        self.mood_points[user_id] = current_mood
        
        # Update status if mood changed significantly
        if abs(current_mood - old_mood) >= 2:
            asyncio.create_task(self.update_dynamic_status())
        
        return current_mood

    def get_tone_from_mood(self, mood_points: int) -> str:
        """Convert mood points to tsundere tone"""
        if mood_points >= 9: return "dere-hot"
        elif mood_points >= 6: return "cheerful"
        elif mood_points >= 3: return "soft-tsun"
        elif mood_points >= 0: return "classic-tsun"
        elif mood_points >= -3: return "grumpy-tsun"
        elif mood_points >= -6: return "cold-tsun"
        else: return "explosive-tsun"

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
            
            response = await self.ollama.chat(
                model=self.current_model,
                messages=messages
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

    async def generate_poml_response(self, user_input: str, username: str, user_id: str) -> tuple[List[Dict], bool]:
        """Generate response using POML templates"""
        if not POML_AVAILABLE or 'personality' not in self.poml_templates:
            return [], False

        try:
            # Update user mood
            mood_points = self.adjust_user_mood(user_id, user_input)
            tone = self.get_tone_from_mood(mood_points)

            # Build context for POML template
            context = {
                "user_input": user_input,
                "username": username,
                "mood_points": mood_points,
                "tone": tone,
                "timestamp": datetime.now().isoformat()
            }

            # Process POML template (simplified approach)
            try:
                # Use basic POML processing with proper encoding handling
                template_content = self.poml_templates['personality']
                
                # Ensure proper encoding for Unicode characters
                if isinstance(template_content, str):
                    template_content = template_content.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                
                result = poml(template_content, context=context)

                # Handle different POML result formats
                if isinstance(result, list) and len(result) > 0:
                    # Extract content from POML list format
                    content_parts = []
                    for item in result:
                        if isinstance(item, dict) and 'content' in item:
                            content_parts.append(item['content'])
                        else:
                            content_parts.append(str(item))

                    # Create system message from POML content
                    system_content = '\n'.join(content_parts)
                    messages = [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_input}
                    ]
                    return messages, True

                elif isinstance(result, dict):
                    # Handle dict format
                    content = result.get('content', str(result))
                    messages = [
                        {"role": "system", "content": content},
                        {"role": "user", "content": user_input}
                    ]
                    return messages, True

                else:
                    print(f"[WARNING] Unexpected POML result format: {type(result)}")
                    return [], False

            except Exception as e:
                error_msg = str(e)
                if "output-schema" in error_msg:
                    print(f"[ERROR] POML component error - trying fallback: {e}")
                    # Try with a simpler approach or disable problematic features
                    try:
                        # Use the template content directly without complex processing
                        template_content = self.poml_templates['personality'].replace("output-schema", "outputformat")
                        result = poml(template_content, context=context)
                        if result:
                            system_content = str(result) if not isinstance(result, str) else result
                            messages = [
                                {"role": "system", "content": system_content},
                                {"role": "user", "content": user_input}
                            ]
                            return messages, True
                    except Exception as fallback_error:
                        print(f"[ERROR] POML fallback failed: {fallback_error}")
                
                print(f"[ERROR] POML processing error: {e}")
                return [], False

        except Exception as e:
            print(f"[ERROR] POML processing error: {e}")
            return [], False

    async def on_ready(self):
        print(f'[OK] {self.user} is online and optimized!')
        if POML_AVAILABLE:
            print(f"[INFO] POML templates loaded: {list(self.poml_templates.keys())}")
        
        # Start dynamic status updates
        self.update_status_task = self.loop.create_task(self.dynamic_status_loop())
        await self.update_dynamic_status()

    async def on_message(self, message):
        # Ignore bot messages
        if message.author.bot:
            return
            
        # Only respond to @ mentions
        if not self.user.mentioned_in(message):
            return
            
        # Process the mention
        await self.handle_mention(message)
    
    async def handle_mention(self, message):
        """Handle @ mentions with full optimization"""
        try:
            print(f"[MESSAGE] User: {message.author.display_name} ({message.author.id})")
            print(f"[MESSAGE] Channel: #{message.channel.name} ({message.channel.id})")
            print(f"[MESSAGE] Guild: {message.guild.name if message.guild else 'DM'}")
            print(f"[MESSAGE] Content: {message.content}")
            
            async with message.channel.typing():
                # Get channel history
                channel_id = str(message.channel.id)
                if channel_id not in self.conversation_history:
                    self.conversation_history[channel_id] = []
                
                history = self.conversation_history[channel_id]
                
                # Clean the message (remove @ mention)
                content = message.content
                content = re.sub(f'<@!?{self.user.id}>', '', content).strip()
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

                # Try POML first, fallback to basic prompts
                messages, used_poml = await self.generate_poml_response(
                    content,
                    message.author.display_name,
                    str(message.author.id)
                )

                if not used_poml:
                    # Fallback to basic system prompt
                    system_prompt = self.build_system_prompt()
                    messages = [{"role": "system", "content": system_prompt}]

                    # Add recent history (last 5 turns)
                    for turn in history[-5:]:
                        messages.append({"role": "user", "content": turn['user']})
                        messages.append({"role": "assistant", "content": turn['assistant']})

                    # Add current message
                    messages.append({"role": "user", "content": content})

                # Get response with tools
                response = await self.ollama.chat(
                    model=self.current_model,
                    messages=messages,
                    tools=ALL_TOOLS
                )

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
                
                # Debug: Always show tool call status
                print(f"[DEBUG] Tool calls detected: {len(tool_calls)}")
                if tool_calls:
                    print(f"[DEBUG] Tool calls: {[call.get('name') for call in tool_calls]}")
                else:
                    print(f"[DEBUG] No tool calls - AI responded without using tools")

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
                    final_response = await self.ollama.chat(
                        model=self.current_model,
                        messages=messages,
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
                        if response_text.strip().startswith('{'):
                            json_response = json.loads(response_text)
                            if 'message' in json_response:
                                actual_message = json_response['message']
                                mood = json_response.get('mood', 'neutral')
                                emoji = json_response.get('emoji', '')

                                # Create rich response with mood info
                                response_text = f"{actual_message} {emoji}"

                                # Mood embed removed - now available as !mood command
                    except json.JSONDecodeError:
                        # Not JSON, use response as-is
                        pass

                # Anti-repetition check
                response_text = self.apply_anti_repetition(response_text, history)

                # Store in history
                history.append({
                    'user': content,
                    'assistant': response_text,
                    'timestamp': time.time()
                })

                # Keep history manageable
                if len(history) > 20:
                    history = history[-15:]
                    self.conversation_history[channel_id] = history

                # Send response with paginated embeds if available
                if embeds_to_send:
                    # Send first embed with view, then additional embeds
                    embed, view = embeds_to_send[0]
                    if view:
                        await message.reply(response_text, embed=embed, view=view)
                    else:
                        await message.reply(response_text, embed=embed)
                    
                    # Send additional embeds if any (without views to avoid clutter)
                    for embed, _ in embeds_to_send[1:]:
                        await message.channel.send(embed=embed)
                elif embeds:
                    # Fallback to old embed system
                    await message.reply(response_text, embeds=embeds[:10])  # Discord limit
                else:
                    await message.reply(response_text)
                    
        except Exception as e:
            print(f"[ERROR] Error handling mention: {e}")
            await message.reply("Sorry, I encountered an error processing your message.")

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
# BOT COMMANDS
# =============================================================================

    @commands.command(name='model')
    @commands.has_permissions(administrator=True)
    async def model_select(self, ctx):
        """Select the main LLM model from available Ollama models with dropdown interface"""
        try:
            # Get list of available models from Ollama
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get('http://127.0.0.1:11434/api/tags') as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [model['name'] for model in data.get('models', [])]
                    else:
                        await ctx.send("❌ Failed to connect to Ollama. Make sure it's running.")
                        return

            if not models:
                await ctx.send("❌ No Ollama models found. Please install some models first.")
                return

            # Create paginated dropdown view
            view = ModelSelectView(models, self)
            embed = view.create_embed()

            await ctx.send(embed=embed, view=view)

        except Exception as e:
            print(f"❌ Error in model command: {e}")
            await ctx.send("❌ Error getting model list. Make sure Ollama is running.")

    @model_select.error
    async def model_select_error(self, ctx, error):
        if isinstance(error, commands.MissingPermissions):
            await ctx.send("❌ You need administrator permissions to use this command!")

    @commands.command(name='clear')
    async def clear_history(self, ctx):
        """Clear conversation history for this channel"""
        channel_id = str(ctx.channel.id)
        if channel_id in self.conversation_history:
            del self.conversation_history[channel_id]

        embed = discord.Embed(
            title="🧹 History Cleared",
            description="Conversation history cleared for this channel.",
            color=0x00ff00
        )
        await ctx.send(embed=embed)

    @commands.command(name='vision_model')
    @commands.has_permissions(administrator=True)
    async def change_vision_model(self, ctx, *, model_name: str = None):
        """Change the vision analysis model"""
        if not model_name:
            embed = discord.Embed(
                title="👁️ Current Vision Model",
                description=f"Current vision model: `{self.vision_model}`",
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

            self.vision_model = model_name

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

    @commands.command(name='status')
    async def bot_status(self, ctx):
        """Show bot status and optimization info"""
        embed = discord.Embed(
            title="Bot Status",
            color=0x0099ff
        )

        embed.add_field(
            name="🤖 Current Model",
            value=f"`{self.current_model}`",
            inline=False
        )

        embed.add_field(
            name="👁️ Vision Model",
            value=f"`{self.vision_model}`",
            inline=False
        )

        embed.add_field(
            name="Optimizations",
            value=f"KV Cache: `{os.environ.get('OLLAMA_KV_CACHE_TYPE', 'Not set')}`\n"
                  f"Flash Attention: `{os.environ.get('OLLAMA_FLASH_ATTENTION', 'Not set')}`\n"
                  f"Parallel Requests: `{os.environ.get('OLLAMA_NUM_PARALLEL', 'Not set')}`",
            inline=False
        )

        embed.add_field(
            name="💬 Active Channels",
            value=str(len(self.conversation_history)),
            inline=True
        )

        embed.add_field(
            name="Available Tools",
            value=f"{len(ALL_TOOLS)} tools loaded",
            inline=True
        )

        await ctx.send(embed=embed)

    @commands.command(name='poml')
    @commands.has_permissions(administrator=True)
    async def poml_status(self, ctx):
        """Show POML status and templates"""
        embed = discord.Embed(
            title="POML Status",
            color=0xff69b4
        )

        if POML_AVAILABLE:
            embed.add_field(
                name="POML Available",
                value="Microsoft's Prompt Orchestration Markup Language is active",
                inline=False
            )

            embed.add_field(
                name="Loaded Templates",
                value=f"{len(self.poml_templates)} templates: {', '.join(self.poml_templates.keys())}",
                inline=False
            )

            embed.add_field(
                name="🎯 Features",
                value="• Structured JSON responses\n• Dynamic mood system\n• Context-aware prompts\n• Tool schema validation",
                inline=False
            )
        else:
            embed.add_field(
                name="POML Not Available",
                value="Install with: `pip install poml`",
                inline=False
            )

        await ctx.send(embed=embed)

    @commands.command(name='mood')
    async def check_mood(self, ctx, user: discord.Member = None):
        """Check mood points for a user"""
        target_user = user or ctx.author
        user_id = str(target_user.id)
        mood_points = self.get_user_mood(user_id)
        tone = self.get_tone_from_mood(mood_points)

        embed = discord.Embed(
            title=f"{target_user.display_name}'s Mood",
            color=0xff69b4
        )

        embed.add_field(
            name="Mood Points",
            value=f"{mood_points}/10",
            inline=True
        )

        embed.add_field(
            name="Current Tone",
            value=tone.replace('-', ' ').title(),
            inline=True
        )

        # Add mood description
        mood_descriptions = {
            "dere-hot": "Overflowing sweetness!",
            "cheerful": "Happy and playful",
            "soft-tsun": "Mildly tsundere",
            "classic-tsun": "Traditional tsundere",
            "grumpy-tsun": "Annoyed but helpful",
            "cold-tsun": "Cold but not mean",
            "explosive-tsun": "Very angry tsundere!"
        }

        embed.add_field(
            name="Description",
            value=mood_descriptions.get(tone, "Neutral"),
            inline=False
        )

        await ctx.send(embed=embed)

    @commands.command(name='updatestatus')
    @commands.has_permissions(administrator=True)
    async def update_status_command(self, ctx):
        """Manually update the bot's status"""
        try:
            await ctx.send("Updating status...")
            await self.update_dynamic_status()
            await ctx.send("Status updated!")
        except Exception as e:
            await ctx.send(f"Failed to update status: {str(e)}")

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
            await bot.ollama.session.close()

if __name__ == "__main__":
    print("[INIT] Starting Optimized Discord Bot...")
    print("[INFO] Features: Ollama optimization, BPE tokenization, full tool suite, anti-repetition")
    asyncio.run(main())
