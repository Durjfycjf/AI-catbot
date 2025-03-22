import os
import logging
from flask import Flask, render_template, request, jsonify, session
from chatbot import Chatbot

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key-for-development")

# Check for API keys and determine the best model to use
openai_api_key = os.environ.get("OPENAI_API_KEY")
gemini_api_key = os.environ.get("GEMINI_API_KEY")

# Decide on model preference based on available API keys
if openai_api_key:
    model_preference = "openai"
    logger.info("Using OpenAI as primary model")
elif gemini_api_key:
    model_preference = "gemini"
    logger.info("Using Gemini as primary model")
else:
    model_preference = "nltk"
    logger.info("No API keys found, using NLTK rule-based fallback only")

# Initialize the chatbot with the appropriate model preference
chatbot = Chatbot(model_preference=model_preference)

@app.route('/')
def home():
    """Render the main chat interface."""
    # Initialize or reset the chat history when accessing the home page
    if 'chat_history' not in session:
        session['chat_history'] = []
    return render_template('index.html', chat_history=session['chat_history'])

@app.route('/about')
def about():
    """Render the about page with information about the chatbot."""
    return render_template('about.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Process a chat message and return the AI response."""
    try:
        user_message = request.json.get('message', '')
        
        if not user_message.strip():
            return jsonify({'error': 'Please enter a message'}), 400
        
        # Initialize chat history if it doesn't exist
        if 'chat_history' not in session:
            session['chat_history'] = []
        
        # Get chatbot response
        response = chatbot.get_response(user_message, session['chat_history'])
        
        # Update chat history
        new_exchange = {
            'user': user_message,
            'bot': response
        }
        
        session['chat_history'].append(new_exchange)
        session.modified = True
        
        return jsonify({
            'response': response,
            'history': session['chat_history']
        })
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/reset', methods=['POST'])
def reset_chat():
    """Reset the chat history."""
    try:
        session['chat_history'] = []
        return jsonify({'status': 'Chat history reset successfully'})
    except Exception as e:
        logger.error(f"Error resetting chat: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500
        
@app.route('/model', methods=['GET'])
def get_model():
    """Get the current AI model being used."""
    try:
        available_models = chatbot.available_models
        current_model = chatbot.model_preference
        
        # Check if API keys are available
        openai_available = "openai" in available_models
        gemini_available = "gemini" in available_models
        nltk_available = "nltk" in available_models
        
        return jsonify({
            'current_model': current_model,
            'available_models': available_models,
            'openai_available': openai_available,
            'gemini_available': gemini_available,
            'nltk_available': nltk_available
        })
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500
        
@app.route('/model', methods=['POST'])
def set_model():
    """Change the AI model being used."""
    try:
        model_name = request.json.get('model', 'auto')
        
        # Validate model name
        if model_name not in ['auto', 'openai', 'gemini', 'nltk']:
            return jsonify({'error': 'Invalid model name'}), 400
            
        # Recreate the chatbot with the new model preference
        global chatbot
        chatbot = Chatbot(model_preference=model_name)
        
        return jsonify({
            'status': 'Model updated successfully',
            'current_model': chatbot.model_preference,
            'available_models': chatbot.available_models
        })
        
    except Exception as e:
        logger.error(f"Error setting model: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('index.html', error="Page not found"), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    return render_template('index.html', error="Server error occurred"), 500
