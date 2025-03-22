"""
AI-powered chatbot models using OpenAI and Google Gemini APIs.
This module contains classes for interacting with different AI models.
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class OpenAIModel:
    """
    Class to handle interactions with OpenAI API.
    """
    
    def __init__(self):
        """Initialize the OpenAI client."""
        try:
            from openai import OpenAI
            
            # Get API key from environment variable
            self.api_key = os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                logger.warning("OpenAI API key not found in environment variables")
                self.client = None
            else:
                self.client = OpenAI(api_key=self.api_key)
                self.model = "gpt-4o"  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
                logger.info("OpenAI client initialized successfully")
        except ImportError:
            logger.error("OpenAI library not installed")
            self.client = None
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {str(e)}")
            self.client = None
    
    def get_response(self, user_input: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Get a response from the OpenAI model.
        
        Args:
            user_input: The user's message
            chat_history: List of previous exchanges in the conversation
            
        Returns:
            The model's response as a string
        """
        if not self.client:
            return "OpenAI API is not configured properly. Please check your API key."
        
        try:
            # Format chat history for OpenAI API
            messages = []
            
            # Add system message
            messages.append({
                "role": "system", 
                "content": "You are a helpful assistant. Respond conversationally to the user's messages."
            })
            
            # Add chat history
            if chat_history:
                for exchange in chat_history[-10:]:  # Limit to most recent 10 exchanges
                    messages.append({"role": "user", "content": exchange.get("user", "")})
                    messages.append({"role": "assistant", "content": exchange.get("bot", "")})
            
            # Add current user message
            messages.append({"role": "user", "content": user_input})
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500
            )
            
            # Extract and return the response text
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error getting response from OpenAI: {str(e)}")
            return f"I'm sorry, I encountered an error: {str(e)}"


class GeminiModel:
    """
    Class to handle interactions with Google's Gemini API.
    """
    
    def __init__(self):
        """Initialize the Gemini client."""
        try:
            import google.generativeai as genai
            
            # Get API key from environment variable
            self.api_key = os.environ.get("GEMINI_API_KEY")
            if not self.api_key:
                logger.warning("Gemini API key not found in environment variables")
                self.genai = None
            else:
                self.genai = genai
                self.genai.configure(api_key=self.api_key)
                # Initialize the model
                self.model = self.genai.GenerativeModel('gemini-pro')
                logger.info("Gemini client initialized successfully")
        except ImportError:
            logger.error("Google GenerativeAI library not installed")
            self.genai = None
        except Exception as e:
            logger.error(f"Error initializing Gemini client: {str(e)}")
            self.genai = None
    
    def get_response(self, user_input: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Get a response from the Gemini model.
        
        Args:
            user_input: The user's message
            chat_history: List of previous exchanges in the conversation
            
        Returns:
            The model's response as a string
        """
        if not self.genai:
            return "Gemini API is not configured properly. Please check your API key."
        
        try:
            # Start a chat session
            chat = self.model.start_chat(history=[])
            
            # Add chat history
            if chat_history:
                for exchange in chat_history[-10:]:  # Limit to most recent 10 exchanges
                    chat.send_message(exchange.get("user", ""))
            
            # Send current message and get response
            response = chat.send_message(user_input)
            
            # Return the text response
            return response.text
            
        except Exception as e:
            logger.error(f"Error getting response from Gemini: {str(e)}")
            return f"I'm sorry, I encountered an error: {str(e)}"