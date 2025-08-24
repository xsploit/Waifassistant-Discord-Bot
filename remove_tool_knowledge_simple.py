import re

# Read the file
with open('optimized_discord_bot.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove all vector_tool_knowledge related code with simpler patterns
patterns_to_remove = [
    r'# Vector Tool Knowledge System \(2025\) - NEW!.*?self\.vector_tool_knowledge = None\s*\n',
    r'# Load vector tool knowledge state.*?print\("\[PERSISTENT STATE\] Loaded vector tool knowledge state"\)\s*\n',
    r'# Add vector tool knowledge state if available.*?state\[\'vector_tool_knowledge\'\] = self\.vector_tool_knowledge\.get_persistent_state\(\)\s*\n',
    r'self\.vector_tool_knowledge = None\s*\n',
    r'if self\.vector_tool_knowledge:.*?tool_knowledge_context = ""\s*\n',
    r'if self\.vector_tool_knowledge:.*?system_prompt \+= tool_knowledge_text\s*\n',
    r'if self\.vector_tool_knowledge:.*?print\(f"\[TOOL KNOWLEDGE\] Stored result for {tool_name}"\)\s*\n',
    r'if not self\.bot\.vector_tool_knowledge:.*?return\s*\n',
    r'results = self\.bot\.vector_tool_knowledge\.search_tool_knowledge\([^)]*\)\s*\n',
    r'stats = self\.bot\.vector_tool_knowledge\.get_stats\(\)\s*\n',
    r'from vector_tool_knowledge import VectorToolKnowledge\s*\n',
    r'self\.vector_tool_knowledge = VectorToolKnowledge\("vector_tool_knowledge"\)\s*\n',
    r'if self\.vector_tool_knowledge and \'vector_tool_knowledge\' in state:.*?self\.vector_tool_knowledge\.load_persistent_state\(state\[\'vector_tool_knowledge\'\]\)\s*\n',
    r'if self\.vector_tool_knowledge:.*?state\[\'vector_tool_knowledge\'\] = self\.vector_tool_knowledge\.get_persistent_state\(\)\s*\n',
    r'tool_knowledge_context = ""\s*\n',
    r'tool_knowledge_context = "\\n\\nRelevant Tool Knowledge:\\n"\s*\n',
    r'tool_knowledge_context \+= f"{i}\. {entry\.tool_name}: {entry\.result_summary\[:200\]}\.\.\.\\n"\s*\n',
    r'tool_knowledge_text = "\\n\\nRelevant Tool Knowledge:\\n"\s*\n',
    r'tool_knowledge_text \+= f"{i}\. {entry\.tool_name}: {entry\.result_summary\[:200\]}\.\.\.\\n"\s*\n',
    r'system_prompt \+= tool_knowledge_text\s*\n',
    r'"tool_knowledge": tool_knowledge_context\s*\n',
    r'memory_type = "TOOL_KNOWLEDGE"\s*\n',
    r'async def search_tool_knowledge\(self, ctx, \*, query: str\):.*?await ctx\.send\(f"❌ Error searching tool knowledge: {e}"\)\s*\n',
    r'async def tool_knowledge_stats\(self, ctx\):.*?await ctx\.send\(f"❌ Error getting tool knowledge stats: {e}"\)\s*\n',
    r'class ToolKnowledgeManager:.*?print\(f"\[TOOL KNOWLEDGE\] Error loading knowledge: {e}"\)\s*\n',
    r'self\.tool_knowledge = ToolKnowledgeManager\("tool_knowledge"\)\s*\n',
    r'self\.tool_knowledge = None\s*\n',
    r'if self\.tool_knowledge:.*?tool_knowledge_context = ""\s*\n',
    r'if self\.tool_knowledge:.*?system_prompt \+= tool_knowledge_text\s*\n',
    r'if self\.tool_knowledge:.*?print\(f"\[TOOL KNOWLEDGE\] Stored result for {tool_name}"\)\s*\n',
    r'"tool_knowledge": tool_knowledge_context\s*\n',
    r'memory_type = "TOOL_KNOWLEDGE"\s*\n'
]

# Remove all patterns
for pattern in patterns_to_remove:
    content = re.sub(pattern, '', content, flags=re.MULTILINE | re.DOTALL)

# Remove any remaining vector_tool_knowledge references
content = re.sub(r'self\.vector_tool_knowledge', 'None', content)
content = re.sub(r'self\.bot\.vector_tool_knowledge', 'None', content)
content = re.sub(r'vector_tool_knowledge', '', content)

# Clean up any double newlines
content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)

# Write back to file
with open('optimized_discord_bot.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Tool knowledge system removed successfully!')
