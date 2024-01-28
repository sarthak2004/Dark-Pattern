// popup.js

// Function to get the current tab URL using Chrome Extension API
function getCurrentTabUrl(callback) {
    chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
        var url = tabs[0].url;
        callback(url);
    });
}

// Function to send the current tab URL to Flask server
function sendUrlToServer(url) {
    // Send an HTTP POST request to Flask server
    fetch('http://yourflaskserver.com/receive_url', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url: url }),
    })
    .then(response => {
        console.log('URL sent to server successfully.');
    })
    .catch(error => {
        console.error('Error sending URL to server:', error);
    });
}

// Get the current tab URL and send it to Flask server when the popup is opened
document.addEventListener('DOMContentLoaded', function() {
    getCurrentTabUrl(function(url) {
        sendUrlToServer(url);
    });
});
