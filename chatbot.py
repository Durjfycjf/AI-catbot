import os
import logging
import random
import re
from typing import List, Dict, Any, Optional

# Try importing nltk for the rule-based fallback system
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    nltk_available = True
    
    # Download necessary NLTK data
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('omw-1.4', quiet=True)  # Open Multilingual WordNet data
        logging.info("NLTK resources downloaded successfully")
    except Exception as e:
        logging.warning(f"Error downloading NLTK resources: {str(e)}")
        
except ImportError:
    nltk_available = False

# Try importing AI model classes
try:
    from ai_models import OpenAIModel, GeminiModel
    ai_models_available = True
except ImportError:
    ai_models_available = False

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Chatbot:
    """
    A class that handles interactions to create a conversational AI chatbot.
    This implementation supports multiple backend systems:
    1. OpenAI API (if API key is available)
    2. Google Gemini API (if API key is available)
    3. NLTK-based rule approach (as fallback, no API key required)
    """
    
    def __init__(self, model_preference="auto"):
        """
        Initialize the chatbot with the preferred model.
        
        Args:
            model_preference (str): Which model to prioritize
                - "openai": Use OpenAI if available, fall back to others
                - "gemini": Use Gemini if available, fall back to others
                - "nltk": Use rule-based NLTK approach only
                - "auto" (default): Try OpenAI, then Gemini, then NLTK
        """
        self.model_preference = model_preference
        self.available_models = []
        
        # Try to initialize each model in order of preference
        if ai_models_available and model_preference != "nltk":
            # Try OpenAI first (or if specified)
            if model_preference in ["auto", "openai"]:
                self.openai_model = OpenAIModel()
                if self.openai_model.client:
                    self.available_models.append("openai")
                    logger.info("OpenAI model initialized successfully")
                else:
                    logger.warning("OpenAI model initialization failed, will try alternatives")
            
            # Try Gemini second (or if specified)
            if model_preference in ["auto", "gemini"] or (model_preference == "openai" and "openai" not in self.available_models):
                self.gemini_model = GeminiModel()
                if self.gemini_model.genai:
                    self.available_models.append("gemini")
                    logger.info("Gemini model initialized successfully")
                else:
                    logger.warning("Gemini model initialization failed, will try alternatives")
        
        # Always initialize NLTK as fallback
        if nltk_available:
            self.initialize_nltk_fallback()
            if "nltk" not in self.available_models:
                self.available_models.append("nltk")
            logger.info("Rule-based fallback model initialized successfully")
        
        # Log which models are available
        if not self.available_models:
            logger.warning("No models available, chatbot functionality will be limited")
        else:
            logger.info(f"Chatbot initialized with models: {', '.join(self.available_models)}")
    
    def initialize_nltk_fallback(self):
        """Initialize the NLTK rule-based fallback system."""
        if not nltk_available:
            logger.warning("NLTK not available for fallback")
            return
            
        # Initialize the lemmatizer for word normalization
        self.lemmatizer = WordNetLemmatizer()
        
        # Define patterns and responses
        self.patterns_responses = {
            r'hello|hi|hey|howdy': [
                "Hello! How can I help you today?",
                "Hi there! What can I do for you?",
                "Hey! Nice to meet you. What's on your mind?"
            ],
            r'how are you|how\'s it going': [
                "I'm just a program, but I'm functioning well! How can I assist you?",
                "I'm doing great, thanks for asking! How can I help?"
            ],
            r'bye|goodbye|see you|farewell': [
                "Goodbye! Feel free to come back if you have more questions.",
                "Farewell! Have a great day!",
                "See you later! Take care!"
            ],
            r'thank|thanks': [
                "You're welcome!",
                "Happy to help!",
                "Anytime! That's what I'm here for."
            ],
            r'name|your name|who are you': [
                "I'm a simple AI chatbot created to help answer your questions.",
                "I'm an AI assistant here to chat with you."
            ],
            r'(what|how) (can|do) you do': [
                "I can answer questions, have a conversation, or just chat about various topics.",
                "I'm designed to engage in conversation and provide information on a variety of subjects."
            ],
            r'weather|temperature|forecast': [
                "I don't have access to real-time weather data, but I'd be happy to chat about other topics!",
                "I can't check the current weather, but I can help with other questions you might have."
            ],
            r'joke|tell.*joke|funny': [
                "Why don't scientists trust atoms? Because they make up everything!",
                "What did one wall say to the other wall? I'll meet you at the corner!",
                "Why did the scarecrow win an award? Because he was outstanding in his field!"
            ],
            r'time|current time|what time': [
                "I don't have access to the current time, but your device should have that information!",
                "I can't tell you the exact time right now, but I'm always ready to chat."
            ],
            r'help|assist|support': [
                "I'm here to help! Feel free to ask me any questions or just chat.",
                "I'm at your service! What do you need assistance with?",
                "How can I assist you today? Just let me know what you're looking for."
            ],
            r'model|which model|what model|how do you work': [
                f"I'm currently using the following models: {', '.join(self.available_models)}. I'll try to give you the best responses possible!",
                f"My brain is powered by: {', '.join(self.available_models)}. I'm here to assist you with various tasks and questions."
            ]
        }
        
        # Default responses for when no pattern matches
        self.default_responses = [
            "I'm not sure I understand. Could you rephrase that?",
            "That's an interesting point. Can you tell me more?",
            "I don't have specific information about that, but I'm happy to chat about something else.",
            "I'm still learning and may not have the answer to that. What else would you like to talk about?",
            "That's beyond my current capabilities, but I'm here if you have other questions."
        ]
    
    def get_response(self, user_input: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Get a response based on user input and conversation history.
        
        Args:
            user_input: The user's message
            chat_history: List of previous exchanges in the conversation
            
        Returns:
            The chatbot's response
        """
        if not user_input or not user_input.strip():
            return "Please type a message to start the conversation."
            
        try:
            # Use OpenAI if available and preferred
            if "openai" in self.available_models and self.model_preference in ["auto", "openai"]:
                try:
                    return self.openai_model.get_response(user_input, chat_history)
                except Exception as e:
                    logger.error(f"Error with OpenAI model: {str(e)}, falling back to alternative")
            
            # Use Gemini if available and preferred, or if OpenAI failed
            if "gemini" in self.available_models and (self.model_preference in ["auto", "gemini"] or 
                    (self.model_preference == "openai" and "openai" in self.available_models)):
                try:
                    return self.gemini_model.get_response(user_input, chat_history)
                except Exception as e:
                    logger.error(f"Error with Gemini model: {str(e)}, falling back to rule-based approach")
            
            # Fall back to NLTK rule-based approach
            if "nltk" in self.available_models:
                return self.get_nltk_response(user_input, chat_history)
            
            # If no models are available
            return "I'm sorry, no AI models are available at the moment. Please try again later."
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I'm sorry, I encountered an error. Let's try a different topic."
    
    def get_nltk_response(self, user_input: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Get a response using the NLTK rule-based approach."""
        try:
            # Preprocess the user input
            processed_input = self._preprocess_text(user_input)
            
            # Check for context in chat history
            context_response = self._check_context(user_input, chat_history)
            if context_response:
                return context_response
            
            # Check for pattern matches
            for pattern, responses in self.patterns_responses.items():
                if re.search(pattern, processed_input, re.IGNORECASE):
                    return random.choice(responses)
            
            # If no patterns match, return a default response
            return random.choice(self.default_responses)
            
        except Exception as e:
            logger.error(f"Error in NLTK response: {str(e)}")
            return "I'm sorry, I encountered an error with the rule-based system."
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text by tokenizing, removing stopwords, and lemmatizing.
        
        Args:
            text: Text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not nltk_available:
            return text.lower()  # Simple fallback if NLTK is not available
            
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords and lemmatize
            stop_words = set(stopwords.words('english'))
            filtered_tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
            
            # Join back into a string
            return ' '.join(filtered_tokens)
        except Exception as e:
            logger.warning(f"Error preprocessing text: {str(e)}")
            return text  # Return original text if preprocessing fails
    
    def _check_context(self, user_input: str, chat_history: Optional[List[Dict[str, str]]]) -> Optional[str]:
        """
        Check conversation context to provide more relevant responses.
        
        Args:
            user_input: The user's message
            chat_history: List of previous exchanges in the conversation
            
        Returns:
            A context-based response or None if no context found
        """
        if not chat_history or len(chat_history) == 0:
            return None
            
        # Check last few exchanges for context
        recent_history = chat_history[-3:] if len(chat_history) >= 3 else chat_history
        
        # Simple context handling for follow-up questions
        if any(word in user_input.lower() for word in ['why', 'how', 'what about', 'tell me more']):
            last_exchange = recent_history[-1]
            if 'weather' in last_exchange.get('user', '').lower():
                return "I can't provide real-time weather information, but you might want to check a weather app for accurate forecasts."
            elif 'joke' in last_exchange.get('user', '').lower():
                return "Here's another one: Why did the bicycle fall over? Because it was two-tired!"
                
        return None
    
    def get_command_line_response(self, user_input: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Special version of get_response for command-line interface that includes
        error handling and formatted output.
        
        Args:
            user_input: The user's message
            chat_history: List of previous exchanges in the conversation
            
        Returns:
            The AI's response formatted for command-line display
        """
        try:
            response = self.get_response(user_input, chat_history)
            return response
        except Exception as e:
            logger.error(f"Error in command line response: {str(e)}")
            return f"Error: {str(e)}"

# Command-line interface for the chatbot
def main():
    """Run the chatbot in command-line mode."""
    print("Welcome to the AI Chatbot! Type 'exit' or 'quit' to end the conversation.")
    print("Type 'clear' to clear the conversation history.")
    print("-" * 50)
    
    chatbot = Chatbot()
    chat_history = []
    
    while True:
        user_input = input("\nYou: ")
        
        # Check for exit commands
        if user_input.lower() in ['exit', 'quit']:
            print("\nGoodbye! Thank you for chatting.")
            break
            
        # Check for clear command
        if user_input.lower() == 'clear':
            chat_history = []
            print("\nConversation history cleared.")
            continue
        
        # Get response from chatbot
        response = chatbot.get_command_line_response(user_input, chat_history)
        
        # Update chat history
        chat_history.append({
            'user': user_input,
            'bot': response
        })
        
        print(f"\nAI: {response}")
        print("-" * 50)

if __name__ == "__main__":
    main()
