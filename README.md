# 🤖 Hikari-chan - Advanced Discord AI Bot

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![POML](https://img.shields.io/badge/POML-Enabled-purple.svg)](https://github.com/microsoft/poml)
[![CUDA](https://img.shields.io/badge/CUDA-Accelerated-green.svg)](https://developer.nvidia.com/cuda-zone)
[![Discord.py](https://img.shields.io/badge/Discord.py-2.6.0-5865F2.svg)](https://discordpy.readthedocs.io/)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*A state-of-the-art Discord bot featuring GPU-accelerated AI intent classification, advanced conversation memory, and dynamic tsundere personality powered by Microsoft's POML framework.*

[Features](#-features) • [Quick Start](#-quick-start) • [Installation](#-installation) • [Configuration](#%EF%B8%8F-configuration) • [Documentation](#-documentation)

</div>

---

## 🌟 Features

### 🧠 **Advanced AI Systems (2025)**
- **GPU-Accelerated Intent Classification**: Real-time sentiment analysis using CUDA-optimized transformers
- **Modern Conversation Memory**: Smart buffer window system with bot message filtering  
- **Dynamic Personality Engine**: POML-powered tsundere character with mood-based responses
- **Zero-Shot Classification**: facebook/bart-large-mnli for intelligent message understanding

### 🎭 **Intelligent Personality System**
- **Mood Tracking**: Per-user emotional state management (-10 to +10 scale)
- **Tsundere Dynamics**: 7 distinct personality modes from sweet to explosive
- **Context Awareness**: Maintains conversation history with smart summarization
- **Anti-Repetition**: AI-powered response diversity to prevent loops

### 🚀 **Performance & Optimization**
- **BPE Tokenization**: Optimized token processing for all LLM interactions
- **KV Cache**: f16 precision caching for faster inference
- **Flash Attention**: Enhanced attention mechanisms for Q4_K_M models  
- **Memory Efficiency**: Modern buffer window system reduces token usage by 60%

### 🛠️ **Comprehensive Tool Suite**
- **Web Search & News**: Real-time information with paginated results
- **Vision Analysis**: Multi-modal AI for image understanding
- **Weather & Time**: Location-aware services with geocoding
- **Discord Integration**: User profiling, server management, avatar analysis
- **Mathematical**: Safe expression evaluation and calculations

### 🎨 **Developer Experience**
- **POML Templates**: Structured prompt engineering with Microsoft's markup language
- **VS Code Integration**: Syntax highlighting and real-time preview
- **Hot Reloading**: Dynamic template updates without restart
- **Rich Debugging**: Comprehensive logging and performance metrics

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+** with pip
- **Ollama** installed and running
- **Discord Bot Token** from [Discord Developer Portal](https://discord.com/developers/applications)
- **NVIDIA GPU** (optional, for AI acceleration)

### 30-Second Setup
```bash
git clone <repository-url>
cd Waifassistant-Discord-Bot
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
cp .env.example .env  # Edit with your Discord token
python optimized_discord_bot.py
```

---

## 🔧 Installation

### 1. **Environment Setup**
```bash
# Create isolated environment
python -m venv .venv

# Activate environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Verify Python version
python --version  # Should be 3.10+
```

### 2. **Install Dependencies**
```bash
# Install all packages (includes CUDA PyTorch)
pip install -r requirements.txt

# Verify core installations
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import poml; print('POML ready')"
python -c "import transformers; print('Transformers ready')"
```

**Key Dependencies:**
- `torch==2.5.1+cu121` - CUDA-accelerated PyTorch
- `transformers==4.55.4` - HuggingFace transformers for AI classification
- `discord.py==2.6.0` - Discord API wrapper
- `poml==0.0.7` - Microsoft's Prompt Orchestration Markup Language
- `ollama==0.5.3` - Optimized Ollama client

### 3. **Ollama Model Setup**
```bash
# Core models (required)
ollama pull hf.co/subsectmusic/qwriko3-4b-instruct-2507:Q4_K_M
ollama pull granite3.2-vision:2b

# Optional alternatives
ollama pull llama3.2:3b
ollama pull llava:7b
```

### 4. **Configuration**
Create `.env` file:
```env
# Required
DISCORD_BOT_TOKEN=your_discord_bot_token_here

# Optional (enhances functionality)
SERPER_API_KEY=your_serper_api_key_here  # Web search
```

**Ollama Optimizations:**
```bash
# Windows
set OLLAMA_KV_CACHE_TYPE=f16
set OLLAMA_FLASH_ATTENTION=1
set OLLAMA_NUM_PARALLEL=2

# Linux/macOS  
export OLLAMA_KV_CACHE_TYPE=f16
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_NUM_PARALLEL=2
```

---

## 🎮 Usage

### **Chat Interactions**
```
@Hikari-chan Hello! How are you today?
@Hikari-chan What do you think about this? [upload image]
@Hikari-chan Search for the latest AI news
```

### **Tool Commands**
```bash
@Hikari-chan search Python tutorials          # Web search
@Hikari-chan weather in Tokyo                # Current weather  
@Hikari-chan calculate 25 * 17 + 100         # Math operations
@Hikari-chan analyze @username                # User profiling
@Hikari-chan who's online?                   # Discord actions
```

### **Admin Commands**
```bash
!status        # Bot status with GPU/AI info
!memory        # Conversation memory statistics  
!clear         # Reset conversation history
!mood @user    # Check user's mood points
!model         # Interactive model selector
```

---

## ⚙️ Configuration

### **Environment Variables**
| Variable | Required | Description |
|----------|----------|-------------|
| `DISCORD_BOT_TOKEN` | ✅ | Discord bot authentication token |
| `SERPER_API_KEY` | ❌ | Web search API (100 free searches/month) |
| `OLLAMA_KV_CACHE_TYPE` | ❌ | Cache precision (f16/q8_0/q4_0) |
| `OLLAMA_FLASH_ATTENTION` | ❌ | Enable flash attention (1/0) |
| `OLLAMA_NUM_PARALLEL` | ❌ | Parallel request limit (1-4) |

### **POML Templates**
Located in `templates/` directory:
- `personality.poml` - Tsundere character definition
- `mood_system.poml` - Emotional state management  
- `tools.poml` - Tool integration rules

**Customize Personality:**
```xml
<poml version="1.0">
    <role>
        You are Hikari-chan, mood={{mood_points}}, tone={{tone}}
        <!-- Modify personality traits here -->
    </role>
</poml>
```

### **Memory System Configuration**
```python
# In optimized_discord_bot.py
self.memory = ConversationMemoryManager(
    window_size=8,              # Recent messages to keep
    summary_threshold=25,       # When to create summaries
    max_context_tokens=3000,    # Token limit
)
```

---

## 🧠 AI Systems

### **Intent Classification**
- **Model**: facebook/bart-large-mnli (zero-shot classification)
- **GPU Acceleration**: Automatic CUDA detection with CPU fallback
- **Classification Types**: Intent, vibe, emotional intensity
- **Real-time Processing**: <100ms classification on RTX GPUs

### **Conversation Memory**
- **Smart Filtering**: Excludes bot messages to prevent self-feeding
- **Window Buffer**: Maintains 8 most recent user messages
- **Summarization**: Auto-summarizes conversations >25 messages
- **Token Efficiency**: ~60% reduction in context token usage

### **Mood System**
| Mood Range | Personality Mode | Behavior |
|------------|------------------|----------|
| 8-10 | dere-hot | Very flirty, openly sweet |
| 5-7 | cheerful | Warm and teasing |
| 2-4 | soft-dere | Cooperative with light sass |
| -1-1 | neutral | Default sassy mode |
| -4--2 | classic-tsun | Flustered denials |
| -7--5 | grumpy-tsun | Snappy but helpful |
| -10--8 | explosive-tsun | Harsh outbursts |

---

## 🔍 Troubleshooting

### **Common Issues**

**Bot Won't Start**
```bash
# Check Python version
python --version  # Should be 3.10+

# Verify virtual environment
which python  # Should point to .venv

# Test core imports
python -c "import discord, poml, torch, transformers"
```

**No GPU Acceleration**
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Verify PyTorch version
pip list | grep torch  # Should show +cu121
```

**Memory Issues**
```bash
# Check conversation stats
!memory  # In Discord

# Clear if needed
!clear   # In Discord
```

**Template Errors**
```bash
# Verify POML syntax
python -c "import poml; poml.load('templates/personality.poml')"

# Check template loading
# Look for "[OK] POML templates loaded" in console
```

### **Performance Optimization**

**GPU Settings:**
- RTX 4060/4070: Use defaults
- RTX 4080+: Increase `OLLAMA_NUM_PARALLEL=4`
- Low VRAM: Set `OLLAMA_KV_CACHE_TYPE=q4_0`

**Memory Settings:**
- 8GB RAM: `window_size=5, summary_threshold=15`
- 16GB+ RAM: `window_size=10, summary_threshold=30`

---

## 📊 System Status

Use `!status` command to view:
- 🔌 **Bot Status**: Uptime, model info
- 🧠 **AI Classification**: GPU status, model loaded
- 💾 **Memory System**: Context efficiency, token usage
- ⚡ **Performance**: Response times, cache hit rates

---

## 🛠️ Development

### **File Structure**
```
waifassistant-discord-bot/
├── optimized_discord_bot.py      # Main bot application
├── ai_intent_classifier.py       # GPU-accelerated AI classification
├── conversation_memory.py        # Modern memory management
├── requirements.txt               # Frozen dependencies
├── POML_Guide.md                 # POML documentation
├── .env.example                  # Environment template
└── templates/
    ├── personality.poml          # Character definition
    ├── mood_system.poml          # Emotional states
    └── tools.poml                # Tool integration
```

### **Adding Custom Tools**
```python
# Define tool schema
my_tool = {
    'type': 'function',
    'function': {
        'name': 'my_custom_tool',
        'description': 'Tool description',
        'parameters': {...}
    }
}

# Add to tool registry
ALL_TOOLS.append(my_tool)
```

### **POML Template Development**
1. Install VS Code POML extension
2. Edit templates in `templates/` directory
3. Use `{{variables}}` for dynamic content
4. Test with `python -c "import poml; print(poml.load('your_template.poml'))"`

---

## 📈 What's New (2025)

### **v2.0 Features**
- ✨ GPU-accelerated intent classification
- 🧠 Modern conversation memory system  
- 🎯 Zero-shot sentiment analysis
- 🚀 60% improved token efficiency
- 🔧 Enhanced POML templates
- 📊 Real-time performance metrics

### **Breaking Changes**
- Requires Python 3.10+ (was 3.8+)
- New conversation memory format
- Updated POML template structure
- CUDA PyTorch dependencies

---

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Follow existing code patterns
4. Update documentation
5. Test with POML templates
6. Submit pull request

**Development Setup:**
```bash
git clone <your-fork>
cd Waifassistant-Discord-Bot  
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# Make your changes
python optimized_discord_bot.py  # Test locally
```

---

## 📝 License

MIT License - feel free to modify and distribute.

## 🙏 Acknowledgments

- **Microsoft** - POML framework
- **HuggingFace** - Transformers library  
- **Ollama Team** - Local LLM inference
- **Discord.py** - Python Discord API

---

<div align="center">

**🎭 Ready to chat with Hikari-chan? Set up your bot and experience the future of Discord AI! ✨**

*Built with ❤️ for the AI community*

</div>