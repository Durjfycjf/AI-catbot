document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const chatContainer = document.getElementById('chatContainer');
    const resetButton = document.getElementById('resetChat');
    const typingIndicator = document.getElementById('typingIndicator');
    const currentModelDisplay = document.getElementById('currentModel');
    const modelItems = document.querySelectorAll('.dropdown-item[data-model]');
    const checkModelsButton = document.getElementById('checkModels');
    
    // Track current model
    let currentModel = 'auto';
    
    // Function to add messages to chat
    function addMessage(message, isUser = false, isSystem = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = isUser ? 'message user-message' : 'message bot-message';
        
        if (isSystem) {
            messageDiv.classList.add('system-message');
        }
        
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="message-text">
                    <p>${message}</p>
                </div>
            </div>
        `;
        
        chatContainer.appendChild(messageDiv);
        
        // Scroll to bottom of chat
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    
    // Function to show typing indicator
    function showTypingIndicator() {
        typingIndicator.classList.remove('d-none');
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    
    // Function to hide typing indicator
    function hideTypingIndicator() {
        typingIndicator.classList.add('d-none');
    }
    
    // Function to check which models are available
    function checkAvailableModels() {
        showTypingIndicator();
        
        fetch('/model', {
            method: 'GET'
        })
        .then(response => response.json())
        .then(data => {
            hideTypingIndicator();
            
            // Update the current model display
            currentModelDisplay.textContent = data.current_model.charAt(0).toUpperCase() + data.current_model.slice(1);
            currentModel = data.current_model;
            
            // Show available models as a system message
            let modelMessage = 'Available AI models:';
            
            if (data.openai_available) {
                modelMessage += '\n✓ OpenAI (GPT-4o)';
            } else {
                modelMessage += '\n✗ OpenAI (API key not configured)';
            }
            
            if (data.gemini_available) {
                modelMessage += '\n✓ Google Gemini';
            } else {
                modelMessage += '\n✗ Google Gemini (API key not configured)';
            }
            
            if (data.nltk_available) {
                modelMessage += '\n✓ NLTK (Rule-Based Fallback)';
            }
            
            modelMessage += `\n\nCurrent model: ${data.current_model.toUpperCase()}`;
            
            // Display as system message
            addMessage(modelMessage, false, true);
        })
        .catch(error => {
            hideTypingIndicator();
            console.error('Error checking models:', error);
            addMessage('Error checking available models. Please try again.', false, true);
        });
    }
    
    // Function to handle form submission
    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const message = userInput.value.trim();
        if (!message) return;
        
        // Add user message to chat
        addMessage(message, true);
        
        // Clear input field
        userInput.value = '';
        
        // Show typing indicator
        showTypingIndicator();
        
        // Send message to server
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Hide typing indicator
            hideTypingIndicator();
            
            // Add bot response to chat
            if (data.error) {
                addMessage(`Error: ${data.error}`, false);
            } else {
                addMessage(data.response, false);
            }
        })
        .catch(error => {
            // Hide typing indicator
            hideTypingIndicator();
            
            // Add error message to chat
            addMessage(`Sorry, there was an error processing your request: ${error.message}`, false);
            console.error('Error:', error);
        });
    });
    
    // Function to handle chat reset
    resetButton.addEventListener('click', function() {
        // Send reset request to server
        fetch('/reset', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            // Clear chat container
            chatContainer.innerHTML = '';
            
            // Add welcome message
            addMessage('Hello! I\'m your AI assistant. How can I help you today?', false);
        })
        .catch(error => {
            console.error('Error resetting chat:', error);
            addMessage('Error resetting chat. Please try again.', false);
        });
    });
    
    // Function to change the AI model
    function changeModel(modelName) {
        showTypingIndicator();
        
        fetch('/model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model: modelName })
        })
        .then(response => response.json())
        .then(data => {
            hideTypingIndicator();
            
            // Update the current model display
            currentModelDisplay.textContent = data.current_model.charAt(0).toUpperCase() + data.current_model.slice(1);
            currentModel = data.current_model;
            
            // Display confirmation message
            addMessage(`Model switched to: ${data.current_model.toUpperCase()}`, false, true);
        })
        .catch(error => {
            hideTypingIndicator();
            console.error('Error changing model:', error);
            addMessage('Error changing model. Please try again.', false, true);
        });
    }
    
    // Add event listeners to model selection dropdown items
    modelItems.forEach(item => {
        item.addEventListener('click', function(e) {
            e.preventDefault();
            const modelName = this.getAttribute('data-model');
            changeModel(modelName);
        });
    });
    
    // Add event listener to check models button
    checkModelsButton.addEventListener('click', function(e) {
        e.preventDefault();
        checkAvailableModels();
    });
    
    // Check available models on page load
    checkAvailableModels();
    
    // Focus on input field when page loads
    userInput.focus();
});
