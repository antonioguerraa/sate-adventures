document.addEventListener('DOMContentLoaded', function() {
  // Elements
  const chatBox = document.getElementById('chat-box');
  const messageInput = document.getElementById('message-input');
  const sendButton = document.getElementById('send-button');
  const settingsButton = document.getElementById('settings-button');
  const closeSettingsButton = document.getElementById('close-settings');
  const settingsPanel = document.getElementById('settings-panel');
  const overlay = document.getElementById('overlay');
  const temperatureSlider = document.getElementById('temperature');
  const temperatureValue = document.getElementById('temperature-value');
  const maxTokensInput = document.getElementById('max-tokens');
  const splashScreen = document.getElementById('splash-screen');
  
  // State
  let isGenerating = false;
  
  // Hide splash screen after 2 seconds
  setTimeout(() => {
    splashScreen.style.opacity = '0';
    setTimeout(() => {
      splashScreen.style.display = 'none';
    }, 500);
  }, 2000);
  
  // Add initial bot message
  addBotMessage("Hi there! I'm your Shakespeare-inspired Mini LLM. Send me a prompt and I'll continue it in my own Shakespearean style.");
  
  // Settings panel control
  settingsButton.addEventListener('click', function() {
    settingsPanel.classList.add('active');
    overlay.classList.add('active');
  });
  
  closeSettingsButton.addEventListener('click', function() {
    settingsPanel.classList.remove('active');
    overlay.classList.remove('active');
  });
  
  overlay.addEventListener('click', function() {
    settingsPanel.classList.remove('active');
    overlay.classList.remove('active');
  });
  
  // Update temperature value display
  temperatureSlider.addEventListener('input', function() {
    temperatureValue.textContent = parseFloat(this.value).toFixed(1);
  });
  
  // Send message on enter
  messageInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  
  // Send message on button click
  sendButton.addEventListener('click', sendMessage);
  
  // Function to send user message and get response
  function sendMessage() {
    const message = messageInput.value.trim();
    if (isGenerating || message === '') return;
    
    // Clear input
    messageInput.value = '';
    
    // Add user message to chat
    addUserMessage(message);
    
    // Show typing indicator
    const typingIndicator = addTypingIndicator();
    
    // Disable input while generating
    isGenerating = true;
    
    // Get settings values
    const temperature = parseFloat(temperatureSlider.value);
    const maxTokens = parseInt(maxTokensInput.value);
    
    // Make API request
    fetch('/generate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        prompt: message,
        temperature: temperature,
        max_tokens: maxTokens
      })
    })
    .then(response => response.json())
    .then(data => {
      // Remove typing indicator
      chatBox.removeChild(typingIndicator);
      
      // Add bot response
      if (data.error) {
        addBotMessage(`Error: ${data.error}`);
      } else {
        addBotMessage(data.generated_text);
      }
      
      // Re-enable input
      isGenerating = false;
      messageInput.focus();
    })
    .catch(error => {
      // Remove typing indicator
      chatBox.removeChild(typingIndicator);
      
      // Add error message
      addBotMessage("Sorry, something went wrong. Please try again.");
      
      // Re-enable input
      isGenerating = false;
      console.error('Error:', error);
    });
  }
  
  // Function to add a user message to the chat
  function addUserMessage(text) {
    const messageElement = document.createElement('div');
    messageElement.className = 'message user-message message-appear';
    messageElement.innerHTML = `<div class="message-content">${escapeHTML(text)}</div>`;
    chatBox.appendChild(messageElement);
    scrollToBottom();
  }
  
  // Function to add a bot message to the chat
  function addBotMessage(text) {
    const messageElement = document.createElement('div');
    messageElement.className = 'message bot-message message-appear';
    messageElement.innerHTML = `<div class="message-content">${formatBotMessage(text)}</div>`;
    chatBox.appendChild(messageElement);
    scrollToBottom();
  }
  
  // Function to add typing indicator
  function addTypingIndicator() {
    const typingElement = document.createElement('div');
    typingElement.className = 'message bot-message typing-indicator';
    typingElement.innerHTML = `
      <div class="message-content">
        <div class="typing-bubble"></div>
        <div class="typing-bubble"></div>
        <div class="typing-bubble"></div>
      </div>
    `;
    chatBox.appendChild(typingElement);
    scrollToBottom();
    return typingElement;
  }
  
  // Function to scroll chat to bottom
  function scrollToBottom() {
    chatBox.scrollTop = chatBox.scrollHeight;
  }
  
  // Function to format bot messages
  function formatBotMessage(text) {
    // Replace newlines with <br>
    return escapeHTML(text).replace(/\n/g, '<br>');
  }
  
  // Function to escape HTML
  function escapeHTML(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
});

// Handle service worker for PWA if needed
if ('serviceWorker' in navigator) {
  window.addEventListener('load', function() {
    navigator.serviceWorker.register('/service-worker.js').then(function(registration) {
      console.log('ServiceWorker registration successful');
    }, function(err) {
      console.log('ServiceWorker registration failed: ', err);
    });
  });
}