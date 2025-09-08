# 🚨 POML Discord Bot Troubleshooting Guide

*Specific solutions for Discord bot POML crashes and 5+ hour downtime issues*

## 🎯 **Problem Summary**
- **Issue**: POML works initially, then breaks during bot operation
- **Symptom**: Bot crashes with cryptic Node.js errors
- **Impact**: 5+ hours of downtime, intermittent failures
- **Pattern**: Template corruption during runtime, not at startup

---

## 🔍 **Root Cause Analysis**

### **The Real Problem: Template Corruption**
Based on our deep research and testing:

1. **POML is working correctly** - your `personality.poml` is valid
2. **Template gets corrupted** during bot operation (not at startup)
3. **Corrupted template** causes POML to crash with Node.js errors
4. **Bot crashes** with no graceful fallback
5. **Restart fixes it** temporarily (corruption cleared)

### **Why This Causes 5+ Hours of Downtime**
- **No error detection** - bot doesn't know template is corrupted
- **No automatic recovery** - manual restart required
- **No corruption monitoring** - can't prevent it from happening
- **No fallback system** - bot crashes instead of degrading gracefully

---

## 🛠️ **Immediate Solutions (Without Changing POML)**

### **1. Template Integrity Monitoring**
Add this to your bot to detect corruption before it crashes:

```python
def validate_poml_template(template_content):
    """Validate POML template before processing"""
    try:
        # Basic XML validation
        if not template_content.strip().startswith('<poml'):
            return False, "Template doesn't start with <poml"
        
        # Check for basic structure
        if '<system-msg>' not in template_content:
            return False, "Missing <system-msg> tag"
            
        # Check for unclosed tags (basic)
        open_tags = template_content.count('<')
        close_tags = template_content.count('</')
        if abs(open_tags - close_tags) > 1:  # Allow for self-closing tags
            return False, f"Tag mismatch: {open_tags} open, {close_tags} close"
            
        return True, "Template appears valid"
        
    except Exception as e:
        return False, f"Validation error: {e}"
```

### **2. Automatic Template Recovery**
Add this to reload template if corruption detected:

```python
def safe_poml_processing(self, template_name, context):
    """Safely process POML with corruption detection"""
    try:
        # Get template content
        template_content = self.poml_templates.get(template_name)
        if not template_content:
            return None, "Template not found"
        
        # Validate template before processing
        is_valid, validation_msg = validate_poml_template(template_content)
        if not is_valid:
            print(f"⚠️ Template corruption detected: {validation_msg}")
            
            # Try to reload from disk
            try:
                self.load_poml_templates()  # Reload all templates
                template_content = self.poml_templates.get(template_name)
                if template_content:
                    print("✅ Template recovered from disk")
                else:
                    return None, "Template recovery failed"
            except Exception as e:
                print(f"❌ Template recovery failed: {e}")
                return None, "Template recovery failed"
        
        # Process with POML
        result = poml(template_content, context=context, chat=True)
        return result, "Success"
        
    except Exception as e:
        print(f"❌ POML processing failed: {e}")
        return None, f"Processing error: {e}"
```

### **3. Enhanced Error Handling**
Replace your current POML calls with safe versions:

```python
# OLD (crashes on corruption):
# result = poml(template_content, context=context, chat=True)

# NEW (handles corruption gracefully):
result, status = self.safe_poml_processing('personality', context)
if result is None:
    print(f"⚠️ POML failed: {status}")
    # Use fallback response system
    result = self.generate_fallback_response(context)
```

---

## 🔧 **Advanced Solutions**

### **1. Template Backup System**
Create automatic backups before each use:

```python
def backup_template_before_use(self, template_name):
    """Create backup of template before processing"""
    try:
        template_content = self.poml_templates.get(template_name)
        if template_content:
            # Create timestamped backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"templates/backup_{template_name}_{timestamp}.poml"
            
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(template_content)
            
            print(f"✅ Template backed up: {backup_path}")
            return backup_path
    except Exception as e:
        print(f"⚠️ Backup failed: {e}")
        return None
```

### **2. Corruption Detection Logging**
Track when and how corruption happens:

```python
def log_template_state(self, template_name, action, success, error=None):
    """Log template state changes for debugging"""
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "template": template_name,
        "action": action,
        "success": success,
        "error": error,
        "template_size": len(self.poml_templates.get(template_name, "")),
        "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024  # MB
    }
    
    # Log to file
    with open("poml_debug.log", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    
    print(f"📝 POML Log: {action} on {template_name} - {'✅' if success else '❌'}")
```

### **3. Health Check Command**
Add a command to check POML health:

```python
@commands.command(name='pomlhealth')
async def poml_health_check(self, ctx):
    """Check POML template health and integrity"""
    embed = discord.Embed(title="🔍 POML Health Check", color=0x00ff00)
    
    for template_name in self.poml_templates:
        template_content = self.poml_templates[template_name]
        is_valid, validation_msg = validate_poml_template(template_content)
        
        status = "✅ Healthy" if is_valid else "❌ Corrupted"
        embed.add_field(
            name=f"{template_name}",
            value=f"{status}\nSize: {len(template_content)} chars\n{validation_msg}",
            inline=False
        )
    
    await ctx.send(embed=embed)
```

---

## 🚨 **Emergency Recovery Procedures**

### **When Bot Crashes Due to POML:**

1. **Immediate Action:**
   ```bash
   # Stop the bot
   # Check template integrity
   python -c "
   with open('templates/personality.poml', 'r') as f:
       content = f.read()
       print('Template size:', len(content))
       print('Starts with <poml:', content.startswith('<poml'))
       print('Has system-msg:', '<system-msg>' in content)
   "
   ```

2. **If Template is Corrupted:**
   ```bash
   # Restore from backup
   cp templates/backup_personality_*.poml templates/personality.poml
   
   # Or restore from git
   git checkout HEAD -- templates/personality.poml
   ```

3. **Restart Bot:**
   ```bash
   # Start bot with monitoring
   python optimized_discord_bot.py
   ```

---

## 📊 **Monitoring & Prevention**

### **Daily Health Checks:**
```python
# Add to your bot's startup
async def daily_poml_health_check(self):
    """Daily template integrity check"""
    while True:
        await asyncio.sleep(86400)  # 24 hours
        
        for template_name in self.poml_templates:
            template_content = self.poml_templates[template_name]
            is_valid, validation_msg = validate_poml_template(template_content)
            
            if not is_valid:
                print(f"🚨 CRITICAL: Template {template_name} corrupted!")
                # Send alert to admin channel
                await self.send_admin_alert(f"POML template {template_name} corrupted: {validation_msg}")
                
                # Attempt recovery
                self.load_poml_templates()
```

### **Performance Monitoring:**
```python
def monitor_poml_performance(self):
    """Monitor POML processing performance"""
    start_time = time.time()
    try:
        result = poml(template, context=context, chat=True)
        processing_time = time.time() - start_time
        
        if processing_time > 5.0:  # More than 5 seconds
            print(f"⚠️ Slow POML processing: {processing_time:.2f}s")
            
        return result
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"❌ POML failed after {processing_time:.2f}s: {e}")
        raise
```

---

## 🎯 **Long-term Solutions**

### **1. Template Versioning**
- Use Git for template version control
- Implement template rollback system
- Track template changes and performance

### **2. Automated Testing**
- Test templates before deployment
- Validate syntax automatically
- Performance regression testing

### **3. Fallback Systems**
- Multiple template versions
- Graceful degradation when POML fails
- Alternative response generation methods

---

## 📚 **Reference Resources**

- **POML Research Document**: `POML_RESEARCH_REFERENCE.md`
- **Official Documentation**: [Microsoft POML Python SDK](https://microsoft.github.io/poml/latest/python/)
- **GitHub Repository**: [microsoft/poml](https://github.com/microsoft/poml)

---

## 🚀 **Next Steps**

1. **Implement template validation** immediately
2. **Add corruption detection** to prevent crashes
3. **Create backup system** for automatic recovery
4. **Monitor template health** continuously
5. **Document corruption patterns** for future prevention

*This guide provides immediate solutions to prevent your 5+ hour downtime issues while maintaining all existing POML functionality.*

