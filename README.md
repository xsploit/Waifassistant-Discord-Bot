# 🎭 Hikari-chan Discord Bot

A high-performance Discord bot with AI vision analysis, POML personality system, and comprehensive tool integration.

## ✨ Features

### 🤖 AI Capabilities
- **Dual Model System**: Separate chat and vision models for optimal performance
- **Tool Integration**: Web search, weather, calculations, image analysis, and more
- **Vision Analysis**: Advanced AI-powered image and avatar analysis
- **Smart Context**: Maintains conversation history per channel

### 🎭 POML Personality System
- **Tsundere Character**: Dynamic personality with mood tracking
- **Mood Points**: User-specific mood tracking that affects responses
- **Template-Based**: Customizable personality templates
- **JSON Responses**: Structured responses with mood and emoji data

### 🚀 Performance Optimizations
- **Optimized Ollama Client**: Custom client with connection pooling and caching
- **KV Cache**: Advanced caching for faster model responses
- **Flash Attention**: Enhanced attention mechanisms
- **Parallel Processing**: Multiple concurrent requests support

### 🛠️ Tools & Commands
- **Web Search**: Real-time web search with paginated results
- **Weather**: Current weather data with location lookup
- **Image Analysis**: Comprehensive AI vision analysis
- **User Profiling**: Discord user analysis with avatar insights
- **Discord Actions**: Server management and user interactions
- **Math Calculator**: Safe mathematical expression evaluation

## 📋 Prerequisites

### 1. Install Python 3.8+
- Download from [python.org](https://python.org)
- Ensure `pip` is installed

### 2. Install Ollama
- Download from [ollama.ai](https://ollama.ai)
- Install and start the Ollama service

### 3. Discord Bot Setup
1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Create a new application
3. Go to "Bot" section
4. Create a bot and copy the token
5. Enable "Message Content Intent"
6. Invite bot to your server with appropriate permissions

## 🔧 Installation Steps

### Step 1: Extract Files
Extract the bot files to a folder of your choice.

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (Command Prompt):
.venv\Scripts\activate.bat

# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Linux/macOS:
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
# Install all required packages with exact versions
pip install -r requirements.txt

# Verify POML installation
python -c "import poml; print('POML installed successfully!')"
```

**Key Dependencies Installed:**
- `discord.py` - Discord API wrapper
- `ollama` - Ollama API client
- `poml` - Prompt Orchestration Markup Language for advanced AI prompting
- `aiohttp` - Async HTTP client for web requests and weather API
- `python-dotenv` - Environment variable management
- `pydantic` - Data validation and parsing
- `psutil` & `GPUtil` - System monitoring

**Built-in Features (No API Keys Required):**
- ⏰ **Time & Date** - Current time and date information
- 🌤️ **Weather** - Current weather using free Open-Meteo API
- 🧮 **Calculator** - Mathematical calculations and expressions

### Step 4: Create Environment File
Create a `.env` file in the project root:
```bash
# Copy the example and edit with your tokens
cp .env.example .env
```

Edit `.env` with your actual API keys:
```env
# Discord Bot Token (Required)
DISCORD_BOT_TOKEN=your_discord_bot_token_here

# Serper API Key (Optional - for web search)
SERPER_API_KEY=your_serper_api_key_here
```

### Step 5: Set Environment Variables (Alternative Method)
If you prefer environment variables over `.env` file:

#### Windows (Command Prompt):
```cmd
set DISCORD_BOT_TOKEN=your_discord_bot_token_here
set SERPER_API_KEY=your_serper_api_key_here
```

#### Windows (PowerShell):
```powershell
$env:DISCORD_BOT_TOKEN="your_discord_bot_token_here"
$env:SERPER_API_KEY="your_serper_api_key_here"
```

#### Linux/macOS:
```bash
export DISCORD_BOT_TOKEN="your_discord_bot_token_here"
export SERPER_API_KEY="your_serper_api_key_here"
```

### Step 6: Configure Ollama Models

#### Pull Required Models:
```bash
# Chat model (default)
ollama pull hf.co/subsectmusic/qwriko3-4b-instruct-2507:Q4_K_M

# Vision model (default)
ollama pull granite3.2-vision:2b

# Alternative models (optional)
ollama pull llama3.2:3b
ollama pull llava:7b
```

#### Performance Optimizations (Optional):
```bash
# Windows
set OLLAMA_KV_CACHE_TYPE=q8_0
set OLLAMA_FLASH_ATTENTION=1
set OLLAMA_NUM_PARALLEL=4

# Linux/macOS
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_NUM_PARALLEL=4
```

### Step 7: Run the Bot
```bash
# Make sure virtual environment is activated
python optimized_discord_bot.py
```

**Expected Output:**
```
[OK] Environment variables loaded from .env file
[OK] POML available - Advanced prompt orchestration enabled
[INIT] Starting Optimized Discord Bot...
[INFO] Features: Ollama optimization, BPE tokenization, full tool suite, anti-repetition
[SUCCESS] Bot is ready and online!
```

## 🔑 API Keys (Optional but Recommended)

### Serper API (Web Search)
1. Go to [serper.dev](https://serper.dev)
2. Sign up for free account (100 free searches/month)
3. Get API key from dashboard
4. Add to your `.env` file: `SERPER_API_KEY=your_key_here`

**Benefits:** Enables web search functionality for real-time information

## 🎛️ Discord Bot Permissions

Your bot needs these permissions:
- ✅ Read Messages
- ✅ Send Messages
- ✅ Send Messages in Threads
- ✅ Embed Links
- ✅ Attach Files
- ✅ Read Message History
- ✅ Add Reactions
- ✅ Use Slash Commands (optional)

## 🧪 Testing Your Setup

### 1. Basic Chat Test
```
@Hikari-chan Hello!
```
Expected: Tsundere personality response

### 2. Tool Test
```
@Hikari-chan what time is it?
```
Expected: Current date and time

### 3. Weather Test
```
@Hikari-chan what's the weather in Tokyo?
```
Expected: Current weather data (uses free Open-Meteo API)

### 4. Image Analysis Test
Upload an image and tag the bot:
```
@Hikari-chan [upload image]
```
Expected: Detailed image analysis

### 5. Admin Commands Test
```
!status
```
Expected: Bot status with model information

## 🎮 Usage Examples

### Basic Chat
```
@Hikari-chan Hello!
@Hikari-chan How are you today?
```

### Image Analysis
```
@Hikari-chan [upload image] - Auto-analyzes uploaded images
@Hikari-chan analyze this image [upload image]
@Hikari-chan what do you see in this picture?
```

### Tools & Search
```
@Hikari-chan search for Python tutorials
@Hikari-chan what's the weather in Tokyo?
@Hikari-chan calculate 25 * 17 + 100
@Hikari-chan what time is it?
```

### User Analysis
```
@Hikari-chan analyze @username
@Hikari-chan tell me about @user
```

### Discord Actions
```
@Hikari-chan who's online?
@Hikari-chan show server emojis
@Hikari-chan get @user's avatar
```

## 🎛️ Admin Commands

### Model Management
```
!model                    # Interactive dropdown model selector
                         # - Categorized by type (main/vision/analysis/code/embedding)
                         # - Paginated interface with 20 models per page
                         # - Real-time model switching
!vision_model             # Show current vision model
!vision_model llava:7b    # Change vision model (text command)
```

### Bot Management
```
!status                   # Show bot status and metrics
!clear                    # Clear conversation history
!mood @user              # Check user's mood points
!poml                    # Show POML template status
```

## 🎭 POML Personality System

The bot uses POML (Personality-Oriented Markup Language) templates for dynamic responses:

### Mood System
- **Mood Points**: Range from -10 to +10
- **Dynamic Responses**: Personality changes based on mood
- **User-Specific**: Each user has individual mood tracking

### Personality Modes
- **dere-hot (9-10)**: Overflowing sweetness, openly affectionate
- **cheerful (6-8)**: Friendly, teasing, warm
- **soft-tsun (3-5)**: Cooperative with light sass
- **classic-tsun (0-2)**: Hot-and-cold, flustered denials
- **grumpy-tsun (-1 to -3)**: Short and spiky but helpful
- **cold-tsun (-4 to -6)**: Snappy, minimal assistance
- **explosive-tsun (-7 to -10)**: Harsh but still assists

## 🎭 POML Templates

This bot uses **POML (Prompt Orchestration Markup Language)** for advanced AI prompting. The templates are located in the `templates/` folder:

- `personality.poml` - Defines Hikari-chan's tsundere personality
- `mood_system.poml` - Manages emotional states and responses
- `tools.poml` - Configures tool usage and function calling

**Customizing POML Templates:**
```bash
# Edit personality traits
notepad templates/personality.poml

# Modify mood responses
notepad templates/mood_system.poml

# Configure tool behavior
notepad templates/tools.poml
```

## 🔧 Configuration

### Environment Variables
```bash
# Discord (Required)
DISCORD_BOT_TOKEN=your_bot_token

# API Keys (Optional)
SERPER_API_KEY=your_serper_key      # Web search (100 free/month)

# Ollama Optimizations (Optional)
OLLAMA_KV_CACHE_TYPE=q8_0           # Cache type
OLLAMA_FLASH_ATTENTION=1            # Enable flash attention
OLLAMA_NUM_PARALLEL=4               # Parallel requests
```

### Default Models
- **Chat Model**: `hf.co/subsectmusic/qwriko3-4b-instruct-2507:Q4_K_M`
- **Vision Model**: `granite3.2-vision:2b`

## 📁 File Structure After Setup
```
hikari-chan-bot/
├── optimized_discord_bot.py    # Main bot file
├── requirements.txt            # Python dependencies (frozen versions)
├── .env                        # Environment variables (create from .env.example)
├── .env.example               # Template for environment variables
├── .gitignore                 # Git ignore file
├── README.md                  # This file
├── .venv/                     # Virtual environment (recommended)
└── templates/
    ├── personality.poml       # Tsundere personality template
    ├── mood_system.poml       # Mood tracking system
    └── tools.poml             # Tool configuration
```

## 🛠️ Available Tools

1. **web_search**: Real-time web search with Serper API
2. **news_search**: News search with date filtering
3. **web_scrape**: Webpage content extraction
4. **get_weather**: Weather data with geocoding
5. **calculate**: Safe mathematical calculations
6. **get_time**: Current date and time
7. **analyze_image_tool**: AI vision analysis
8. **analyze_user_profile**: Discord user profiling
9. **dox_user**: Comprehensive user analysis
10. **discord_action**: Server management actions

## 🎯 Advanced Features

### Paginated Results
- Search results, user lists, and message history use interactive pagination
- Navigation buttons: ◀◀ ◀ ▶ ▶▶
- Customizable items per page

### Vision Analysis
- Comprehensive image analysis with 8 key areas
- Avatar personality insights
- Configurable vision models
- Support for multiple image formats

### Performance Monitoring
- Real-time response time tracking
- Tool execution monitoring
- Memory and optimization metrics
- Debug logging for troubleshooting

### Interactive Model Selection
- **Dropdown Interface**: Easy model switching with categorized dropdowns
- **Model Categories**:
  - 💬 Main Chat Models (llama, qwen, mistral, etc.)
  - 👁️ Vision/Multimodal (granite3.2-vision, llava, etc.)
  - 🧠 Analysis/Reasoning (specialized reasoning models)
  - 💻 Code Generation (deepseek-coder, starcoder, etc.)
  - 🔗 Embedding Models (nomic-embed, bge, etc.)
- **Pagination**: Navigate through 20+ models per page
- **Real-time Switching**: Instant model changes without restart

## 🔍 Troubleshooting

### Bot Won't Start
- ✅ Check Python version (3.8+)
- ✅ Verify virtual environment is activated
- ✅ Ensure all dependencies installed: `pip install -r requirements.txt`
- ✅ Check DISCORD_BOT_TOKEN is set in `.env` file
- ✅ Verify Ollama service is running
- ✅ Test POML import: `python -c "import poml"`

### No Responses
- ✅ Check bot permissions in Discord
- ✅ Verify bot is online in Discord
- ✅ Check console for error messages
- ✅ Ensure models are pulled in Ollama
- ✅ Verify POML templates are loading correctly

### POML Template Errors
- ✅ Check template syntax in `.poml` files
- ✅ Verify all template files exist in `templates/` folder
- ✅ Test template loading: Look for `[OK] POML available` in console
- ✅ Check file permissions on template directory

### Tool Errors
- ✅ Check API keys are set correctly in `.env`
- ✅ Verify internet connection
- ✅ Check Ollama model availability
- ✅ Test individual tools with `!status` command

### Performance Issues
- ✅ Set Ollama optimization variables
- ✅ Ensure sufficient RAM (4GB+ recommended)
- ✅ Check CPU usage during model loading
- ✅ Monitor virtual environment memory usage

### Debug Mode
Enable detailed logging by checking console output for:
- `[DEBUG]` messages for tool calls
- `[TOOL]` messages for tool execution
- `[POML]` messages for personality processing
- `[OK]` messages for successful initialization

## 📝 License

This project is open source. Feel free to modify and distribute.

## 🤝 Contributing

Contributions welcome! Please ensure:
- Code follows existing patterns
- POML templates are properly formatted
- All tools have proper error handling
- Documentation is updated

## 🎯 Next Steps

1. **Customize Personality**: Edit POML templates in `templates/` folder
2. **Add More Models**: Pull additional Ollama models for variety
3. **Configure Channels**: Set up dedicated channels for bot interaction
4. **Monitor Performance**: Use `!status` command to track metrics

## 📞 Getting Help

If you encounter issues:
1. Check the console output for error messages
2. Verify all environment variables are set
3. Ensure Ollama service is running
4. Test individual components (Discord connection, Ollama models, API keys)

For support:
- Check console logs for `[ERROR]` messages
- Verify environment variables are set correctly
- Ensure Ollama is running with required models
- Check Discord bot permissions in server settings

---

**Happy chatting with Hikari-chan! 🎭✨**
