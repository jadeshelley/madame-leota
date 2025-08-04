"""
ChatGPT Client for Madame Leota
Handles AI API communication with fallback options
"""

import asyncio
import logging
import random
from typing import Optional
from config import GROQ_API_KEY, GROQ_MODEL, LEOTA_PERSONALITY

# Try to import Groq, but don't fail if it's not available
try:
    from groq import AsyncGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️ Groq not available, will use offline fallback")

class ChatGPTClient:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.conversation_history = [
            {"role": "system", "content": LEOTA_PERSONALITY}
        ]
        
        # Initialize Groq client only if available
        if GROQ_AVAILABLE and GROQ_API_KEY:
            try:
                self.client = AsyncGroq(api_key=GROQ_API_KEY)
                self.groq_available = True
            except Exception as e:
                self.logger.warning(f"Groq initialization failed: {e}")
                self.groq_available = False
        else:
            self.groq_available = False
            
        # Offline fallback responses
        self.fallback_responses = [
            "Ah, mortal... the spirits are restless today. What secrets do you seek?",
            "The crystal ball grows cloudy... but I sense your presence, seeker of truth.",
            "Welcome to my mystical realm, where past and future intertwine like threads of fate.",
            "The spirits whisper of your arrival... what questions burn in your soul?",
            "Through the mists of time, I see you standing before me. Speak your heart's desire.",
            "The veil between worlds grows thin... what knowledge do you seek from beyond?",
            "Ah, a new soul has entered my domain. The crystal ball reveals much about you...",
            "The ancient spirits stir at your presence. What mysteries shall we unravel together?",
            "Welcome, seeker of wisdom. The crystal ball holds answers to your deepest questions.",
            "The mystical forces align as you approach. What fortune shall I reveal to you?"
        ]
        
    async def test_connection(self):
        """Test the AI API connection with fallback"""
        if not self.groq_available:
            self.logger.info("Using offline fallback mode - no API connection needed")
            return True
            
        try:
            response = await self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": "Test connection"}],
                max_tokens=10
            )
            self.logger.info("Groq API connection successful")
            return True
        except Exception as e:
            self.logger.warning(f"Groq API connection failed: {e}")
            self.logger.info("Falling back to offline mode")
            self.groq_available = False
            return True  # Return True so the app can continue with fallback
    
    async def get_response(self, user_input: str) -> Optional[str]:
        """Get a response from AI in character as Madame Leota with fallback"""
        try:
            # Add user message to conversation history
            self.conversation_history.append({"role": "user", "content": user_input})
            
            # Try Groq API first if available
            if self.groq_available:
                try:
                    response = await self.client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=self.conversation_history,
                        max_tokens=150,  # Keep responses relatively short for better lip sync
                        temperature=0.8,  # Add some personality variation
                        presence_penalty=0.2,  # Encourage varied responses
                        frequency_penalty=0.1
                    )
                    
                    ai_response = response.choices[0].message.content.strip()
                    self.logger.debug(f"Groq AI Response: {ai_response}")
                    
                except Exception as e:
                    self.logger.warning(f"Groq API failed, using fallback: {e}")
                    self.groq_available = False
                    ai_response = self._get_fallback_response(user_input)
            else:
                # Use offline fallback
                ai_response = self._get_fallback_response(user_input)
            
            # Add AI response to conversation history
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            
            # Keep conversation history manageable (last 10 exchanges)
            if len(self.conversation_history) > 21:  # 1 system + 20 messages
                # Keep system message and last 10 exchanges
                self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-20:]
            
            return ai_response
            
        except Exception as e:
            self.logger.error(f"Error getting AI response: {e}")
            return self._get_fallback_response(user_input)
    
    def _get_fallback_response(self, user_input: str) -> str:
        """Get a fallback response when API is unavailable"""
        # Simple keyword-based responses for common inputs
        user_input_lower = user_input.lower()
        
        if any(word in user_input_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return random.choice([
                "Ah, greetings mortal! The spirits have foretold your arrival.",
                "Welcome to my mystical realm, seeker of truth.",
                "The crystal ball grows bright at your presence. What brings you here?"
            ])
        elif any(word in user_input_lower for word in ['fortune', 'future', 'predict', 'tell me']):
            return random.choice([
                "The mists of time reveal... your path is filled with unexpected twists.",
                "The spirits whisper of great changes ahead in your life.",
                "I see shadows and light dancing in your future, mortal."
            ])
        elif any(word in user_input_lower for word in ['love', 'relationship', 'heart']):
            return random.choice([
                "The crystal ball shows matters of the heart... love's path is never straight.",
                "Ah, the heart's desires... the spirits see passion and patience in your future.",
                "Love's mysteries unfold before me... trust in the journey ahead."
            ])
        elif any(word in user_input_lower for word in ['career', 'work', 'job', 'money']):
            return random.choice([
                "The spirits speak of prosperity and challenges in your professional path.",
                "Your career path is illuminated by the crystal ball... success awaits the patient.",
                "The mystical forces align for your ambitions, but timing is everything."
            ])
        else:
            # Generic mystical response
            return random.choice(self.fallback_responses)
    
    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = [
            {"role": "system", "content": LEOTA_PERSONALITY}
        ]
        self.logger.info("Conversation history reset")
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the current conversation"""
        if len(self.conversation_history) <= 1:
            return "No conversation yet."
        
        user_messages = [msg["content"] for msg in self.conversation_history if msg["role"] == "user"]
        return f"Conversation with {len(user_messages)} user messages." 