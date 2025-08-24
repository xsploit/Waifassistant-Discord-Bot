"""
AI-based Zero-Shot Intent Classification using HuggingFace Transformers
Replaces the hardcoded keyword-based approach with actual AI models
"""

from transformers import pipeline
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class IntentClassification:
    """Result of intent classification"""
    intent: str
    vibe: str
    confidence: float
    emotional_intensity: str
    context_clues: List[str]
    emotional_indicators: List[str]
    raw_scores: Dict[str, float]

class AIIntentClassifier:
    """
    Zero-shot intent classification using facebook/bart-large-mnli model
    Provides actual AI-based classification instead of keyword matching
    """
    
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        """Initialize the AI classifier"""
        try:
            logger.info(f"Loading zero-shot classification model: {model_name}")
            
            # Check for CUDA GPU availability
            if torch.cuda.is_available():
                device = 0  # Use first GPU
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"CUDA GPU detected: {device_name}")
                print(f"[GPU] Using CUDA GPU: {device_name}")
            else:
                device = -1  # Use CPU
                logger.info("No CUDA GPU detected, using CPU")
                print("[CPU] No CUDA GPU detected, falling back to CPU")
            
            self.classifier = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=device,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32  # FP16 for GPU, FP32 for CPU
            )
            
            if torch.cuda.is_available():
                logger.info("AI Intent Classifier initialized successfully with GPU acceleration")
                print("[OK] AI Intent Classifier initialized with GPU acceleration")
            else:
                logger.info("AI Intent Classifier initialized successfully on CPU")
                print("[OK] AI Intent Classifier initialized on CPU")
                
        except Exception as e:
            logger.error(f"Failed to initialize AI classifier: {e}")
            print(f"[ERROR] Failed to initialize AI classifier: {e}")
            raise
    
    def classify_message(self, message: str) -> IntentClassification:
        """
        Classify user message using AI zero-shot classification
        
        Args:
            message: User's input message
            
        Returns:
            IntentClassification with AI-determined intent and vibe
        """
        try:
            # Define intent categories
            intent_labels = [
                "question", "request", "compliment", "complaint", 
                "casual conversation", "emotional expression", "command"
            ]
            
            # Define vibe categories  
            vibe_labels = [
                "positive", "negative", "neutral", "playful", 
                "serious", "flirty", "sarcastic", "angry"
            ]
            
            # Classify intent
            intent_result = self.classifier(
                message,
                candidate_labels=intent_labels,
                hypothesis_template="This message is a {}."
            )
            
            # Classify vibe
            vibe_result = self.classifier(
                message,
                candidate_labels=vibe_labels,
                hypothesis_template="This message has a {} tone."
            )
            
            # Classify emotional intensity
            intensity_labels = ["high emotional intensity", "medium emotional intensity", "low emotional intensity"]
            intensity_result = self.classifier(
                message,
                candidate_labels=intensity_labels,
                hypothesis_template="This message has {}."
            )
            
            # Extract top predictions
            top_intent = intent_result['labels'][0]
            intent_confidence = intent_result['scores'][0]
            
            top_vibe = vibe_result['labels'][0]
            vibe_confidence = vibe_result['scores'][0]
            
            emotional_intensity = intensity_result['labels'][0].replace(" emotional intensity", "")
            
            # Determine context clues and emotional indicators using AI
            context_result = self.classifier(
                message,
                candidate_labels=[
                    "contains personal information", "asks for help", "expresses frustration",
                    "shows excitement", "mentions time urgency", "uses informal language",
                    "contains technical terms", "expresses gratitude", "shows confusion"
                ],
                hypothesis_template="This message {}."
            )
            
            # Get top 3 context clues with confidence > 0.3
            context_clues = [
                label for label, score in zip(context_result['labels'], context_result['scores'])
                if score > 0.3
            ][:3]
            
            # Detect emotional indicators
            emotion_result = self.classifier(
                message,
                candidate_labels=[
                    "contains excitement", "shows frustration", "expresses happiness",
                    "indicates sadness", "shows anger", "displays confusion",
                    "expresses love", "shows gratitude", "indicates fear"
                ],
                hypothesis_template="This message {}."
            )
            
            emotional_indicators = [
                label.replace("contains ", "").replace("shows ", "").replace("expresses ", "").replace("indicates ", "").replace("displays ", "") 
                for label, score in zip(emotion_result['labels'], emotion_result['scores'])
                if score > 0.25
            ][:3]
            
            # Combine confidence scores
            overall_confidence = (intent_confidence + vibe_confidence) / 2
            
            # Create raw scores dictionary
            raw_scores = {
                "intent_scores": dict(zip(intent_result['labels'], intent_result['scores'])),
                "vibe_scores": dict(zip(vibe_result['labels'], vibe_result['scores'])),
                "intensity_scores": dict(zip(intensity_result['labels'], intensity_result['scores']))
            }
            
            return IntentClassification(
                intent=top_intent,
                vibe=top_vibe,
                confidence=overall_confidence,
                emotional_intensity=emotional_intensity,
                context_clues=context_clues,
                emotional_indicators=emotional_indicators,
                raw_scores=raw_scores
            )
            
        except Exception as e:
            logger.error(f"Error in AI classification: {e}")
            # Fallback to basic classification
            return self._fallback_classification(message)
    
    def _fallback_classification(self, message: str) -> IntentClassification:
        """Fallback classification if AI fails"""
        return IntentClassification(
            intent="casual conversation",
            vibe="neutral", 
            confidence=0.5,
            emotional_intensity="medium",
            context_clues=["fallback_mode"],
            emotional_indicators=["neutral"],
            raw_scores={}
        )
    
    def get_classification_explanation(self, message: str) -> str:
        """
        Get detailed explanation of how the message was classified
        Useful for debugging and understanding AI decisions
        """
        try:
            classification = self.classify_message(message)
            
            explanation = f"""
AI Classification Results for: "{message}"

Intent: {classification.intent} (confidence: {classification.confidence:.2f})
Vibe: {classification.vibe}
Emotional Intensity: {classification.emotional_intensity}
Context Clues: {', '.join(classification.context_clues)}
Emotional Indicators: {', '.join(classification.emotional_indicators)}

Detailed Scores:
Intent Scores: {classification.raw_scores.get('intent_scores', {})}
Vibe Scores: {classification.raw_scores.get('vibe_scores', {})}
            """
            return explanation
            
        except Exception as e:
            return f"Error generating explanation: {e}"

def test_ai_classifier():
    """Test function to demonstrate the AI classifier"""
    try:
        classifier = AIIntentClassifier()
        
        test_messages = [
            "Hey, can you help me with something?",
            "You're so amazing, I love talking to you!",
            "This is so frustrating, nothing is working!",
            "What's the weather like today?",
            "lol that's hilarious 😂",
            "I'm feeling really sad right now...",
            "DM @user and tell them I said hi"
        ]
        
        print("AI Intent Classification Test Results:")
        print("=" * 50)
        
        for message in test_messages:
            result = classifier.classify_message(message)
            print(f"\nMessage: '{message}'")
            print(f"Intent: {result.intent}")
            print(f"Vibe: {result.vibe}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Emotional Intensity: {result.emotional_intensity}")
            print(f"Context: {', '.join(result.context_clues)}")
            print("-" * 30)
            
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_ai_classifier()