#!/usr/bin/env python3
"""
Fix both syntax error and import issue in optimized_discord_bot.py
"""

def fix_issues():
    with open('optimized_discord_bot.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: Remove the stray closing parenthesis on line 4670
    content = content.replace('                )', '')
    
    # Fix 2: Update the import to use ToolKnowledgeManager
    content = content.replace(
        '        # Vector Tool Knowledge System (2025) - NEW!\n        try:\n            from tool_knowledge import VectorToolKnowledge\n            self.tool_knowledge = VectorToolKnowledge("tool_knowledge")\n            print("[OK] Vector Tool Knowledge System initialized")\n        except ImportError as e:\n            print(f"[WARNING] Vector Tool Knowledge System not available: {e}")\n            self.tool_knowledge = None',
        '        # Tool Knowledge System (INTEGRATED FROM MERGED_BOT_FIXED.PY)\n        try:\n            self.tool_knowledge = ToolKnowledgeManager("tool_knowledge")\n            print("[OK] Tool Knowledge System initialized")\n        except Exception as e:\n            print(f"[WARNING] Tool Knowledge System not available: {e}")\n            self.tool_knowledge = None'
    )
    
    with open('optimized_discord_bot.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Both issues fixed!")

if __name__ == "__main__":
    fix_issues()
