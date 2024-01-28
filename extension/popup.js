document.addEventListener('DOMContentLoaded', function() {
    getCurrentTabUrl(function(url) {
        document.getElementById('current_url').textContent = url;
    });
});

function getCurrentTabUrl(callback) {
    chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
        var url = tabs[0].url;
        callback(url);
    });
}
