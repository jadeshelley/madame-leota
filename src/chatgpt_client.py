"""
ChatGPT Client for Madame Leota
Handles Groq AI API communication (free alternative to OpenAI)
"""

import asyncio
import logging
from typing import Optional
from groq import AsyncGroq
from config import GROQ_API_KEY, GROQ_MODEL, LEOTA_PERSONALITY

class ChatGPTClient:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client = AsyncGroq(api_key=GROQ_API_KEY)
        self.conversation_history = [
            {"role": "system", "content": LEOTA_PERSONALITY}
        ]
        
    async def test_connection(self):
        """Test the Groq API connection"""
        try:
            response = await self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": "Test connection"}],
                max_tokens=10
            )
            self.logger.info("Groq API connection successful")
            return True
        except Exception as e:
            self.logger.error(f"Groq API connection failed: {e}")
            raise
    
    async def get_response(self, user_input: str) -> Optional[str]:
        """Get a response from Groq AI in character as Madame Leota"""
        try:
            # Add user message to conversation history
            self.conversation_history.append({"role": "user", "content": user_input})
            
            # Get response from Groq
            response = await self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=self.conversation_history,
                max_tokens=150,  # Keep responses relatively short for better lip sync
                temperature=0.8,  # Add some personality variation
                presence_penalty=0.2,  # Encourage varied responses
                frequency_penalty=0.1
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Add AI response to conversation history
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            
            # Keep conversation history manageable (last 10 exchanges)
            if len(self.conversation_history) > 21:  # 1 system + 20 messages
                # Keep system message and last 10 exchanges
                self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-20:]
            
            self.logger.debug(f"AI Response: {ai_response}")
            return ai_response
            
        except Exception as e:
            self.logger.error(f"Error getting Groq response: {e}")
            return None
    
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