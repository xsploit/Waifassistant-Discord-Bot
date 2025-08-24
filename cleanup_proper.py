#!/usr/bin/env python3
"""
Proper cleanup script for optimized_discord_bot.py
"""

def cleanup_proper():
    with open('optimized_discord_bot.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove the broken vector tool knowledge initialization
    content = content.replace(
        '        # Vector Tool Knowledge System (2025) - NEW!\n        try:\n            from vector_tool_knowledge import VectorToolKnowledge\n            None = VectorToolKnowledge("vector_tool_knowledge")\n            print("[OK] Vector Tool Knowledge System initialized")\n        except ImportError as e:\n            print(f"[WARNING] Vector Tool Knowledge System not available: {e}")\n            None = None',
        ''
    )
    
    # Remove broken persistent state loading
    content = content.replace(
        '                # Load vector tool knowledge state\n                if None and \'vector_tool_knowledge\' in state:\n                    None.load_persistent_state(state[\'vector_tool_knowledge\'])\n                    print("[PERSISTENT STATE] Loaded vector tool knowledge state")',
        ''
    )
    
    # Remove broken persistent state saving
    content = content.replace(
        '            # Add vector tool knowledge state if available\n            if None:\n                state[\'vector_tool_knowledge\'] = None.get_persistent_state()',
        ''
    )
    
    # Remove all broken tool knowledge search calls
    content = content.replace(
        '            if None:\n                try:\n                    # Search for relevant tool knowledge based on user input\n                    relevant_knowledge = None.search_tool_knowledge(\n                        query=user_input,\n                        user_id=user_id,\n                        limit=3  # Get top 3 most relevant pieces\n                    )\n                    \n                    if relevant_knowledge:\n                        tool_knowledge_context = "\\n\\nRelevant Tool Knowledge:\\n"\n                        for i, entry in enumerate(relevant_knowledge, 1):\n                            tool_knowledge_context += f"{i}. {entry.tool_name}: {entry.result_summary[:200]}...\\n"\n                        print(f"[TOOL KNOWLEDGE] Found {len(relevant_knowledge)} relevant knowledge entries for: {user_input[:50]}...")\n                    else:\n                        print(f"[TOOL KNOWLEDGE] No relevant knowledge found for: {user_input[:50]}...")\n                        \n                except Exception as e:\n                    print(f"[TOOL KNOWLEDGE ERROR] Failed to retrieve knowledge: {e}")',
        '            # Tool knowledge system removed'
    )
    
    # Remove all broken tool knowledge storage calls
    content = content.replace(
        '                                # Store tool knowledge for future use (NEW!)\n                                if None:\n                                    try:\n                                        # Generate a search query based on the tool call and result\n                                        search_query = f"tool:{name} args:{args} result:{str(result)[:200]}"\n                                        entry_id = None.add_tool_knowledge(\n                                            user_id=user_id,\n                                            tool_name=name,\n                                            search_query=search_query,\n                                            result_summary=str(result)[:500],\n                                            importance_score=0.7  # Tool usage is moderately important\n                                        )\n                                        print(f"[VECTOR TOOL KNOWLEDGE] Stored knowledge for {name} with ID: {entry_id}")\n                                    except Exception as e:\n                                        print(f"[VECTOR TOOL KNOWLEDGE ERROR] Failed to store tool knowledge: {e}")',
        '                                # Tool knowledge system removed'
    )
    
    # Remove broken tool knowledge commands
    content = content.replace(
        '    @commands.command(name=\'toolsearch\')\n    async def search_tool_knowledge(self, ctx, *, query: str):\n        """Search tool knowledge using vector similarity"""\n        if not self.bot.tool_knowledge:\n            await ctx.send("❌ Vector Tool Knowledge System is not available")\n            return\n            \n        try:\n            # Search for relevant tool knowledge\n            results = self.bot.tool_knowledge.search_tool_knowledge(\n                query=query,\n                limit=5,\n                threshold=0.3,\n                user_id=str(ctx.author.id)\n            )\n            \n            if not results:\n                embed = discord.Embed(\n                    title="🔍 Tool Knowledge Search",\n                    description=f"No relevant tool knowledge found for: **{query}**",\n                    color=0x00ff00\n                )\n                await ctx.send(embed=embed)\n                return\n            \n            # Create embed with results\n            embed = discord.Embed(\n                title="🔍 Tool Knowledge Search Results",\n                description=f"Found {len(results)} relevant results for: **{query}**",\n                color=0x00ff00\n            )\n            \n            for i, result in enumerate(results[:5], 1):\n                entry = result[\'entry\']\n                similarity = result[\'similarity\']\n                relevance = result[\'relevance_score\']\n                \n                # Format the result\n                value = (\n                    f"**Tool:** {entry.tool_name}\\n"\n                    f"**Query:** {entry.search_query[:100]}{\'...\' if len(entry.search_query) > 100 else \'\'}\\n"\n                    f"**Result:** {entry.result_summary[:150]}{\'...\' if len(entry.result_summary) > 150 else \'\'}\\n"\n                    f"**Similarity:** {similarity:.2f} • **Relevance:** {relevance:.2f}"\n                )\n                \n                embed.add_field(\n                    name=f"Result {i}",\n                    value=value,\n                    inline=False\n                )\n            \n            await ctx.send(embed=embed)\n            \n        except Exception as e:\n            await ctx.send(f"❌ Failed to search tool knowledge: {str(e)}")',
        '    # Tool knowledge commands removed'
    )
    
    # Remove toolstats command
    content = content.replace(
        '    @commands.command(name=\'toolstats\')\n    @commands.check_any(commands.is_owner(), commands.has_permissions(administrator=True))\n    async def tool_knowledge_stats(self, ctx):\n        """Show vector tool knowledge system statistics"""\n        if not self.bot.tool_knowledge:\n            await ctx.send("❌ Vector Tool Knowledge System is not available")\n            return\n            \n        try:\n            stats = self.bot.tool_knowledge.get_stats()\n            \n            embed = discord.Embed(\n                title="🔍 Vector Tool Knowledge Stats",\n                color=0x00ff00\n            )\n            \n            embed.add_field(\n                name="Total Entries",\n                value=f"📊 {stats[\'total_entries\']}",\n                inline=True\n            )\n            \n            embed.add_field(\n                name="Unique Tools",\n                value=f"🛠️ {stats[\'unique_tools\']}",\n                inline=True\n            )\n            \n            embed.add_field(\n                name="Unique Users",\n                value=f"👥 {stats[\'unique_users\']}",\n                inline=True\n            )\n            \n            embed.add_field(\n                name="Average Importance",\n                value=f"⭐ {stats[\'average_importance\']}",\n                inline=True\n            )\n            \n            embed.add_field(\n                name="Average Usage Count",\n                value=f"📈 {stats[\'average_usage_count\']}",\n                inline=True\n            )\n            \n            embed.add_field(\n                name="Vector Index",\n                value="✅ Active" if stats[\'vector_index_active\'] else "❌ Inactive",\n                inline=True\n            )\n            \n            embed.add_field(\n                name="Embedder",\n                value="✅ Available" if stats[\'embedder_available\'] else "❌ Unavailable",\n                inline=True\n            )\n            \n            embed.add_field(\n                name="Storage Directory",\n                value=f"📁 {stats[\'storage_directory\']}",\n                inline=False\n            )\n            \n            await ctx.send(embed=embed)\n            \n        except Exception as e:\n            await ctx.send(f"❌ Failed to get tool knowledge stats: {str(e)}")',
        '    # Tool knowledge stats command removed'
    )
    
    with open('optimized_discord_bot.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Proper cleanup completed!")

if __name__ == "__main__":
    cleanup_proper()
