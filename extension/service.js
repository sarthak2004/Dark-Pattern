var tab;

// Query the active tab to get the URL
chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
  tab = tabs[0];
  console.log("URL:", tab.url);
});

// Set up the event listener for the button
document.getElementById('getDataButton').addEventListener('click', function () {
  if (tab && tab.url) {
    const link = tab.url;

    // Show loading screen
    document.getElementById('loadingScreen').style.display = 'flex';

    fetch('http://127.0.0.1:5000/analysis', {  // Replace with your backend URL
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ link: link })
    })
      .then(response => response.json())
      .then(data => {
        const id = data.id;
        // Hide loading screen
        document.getElementById('loadingScreen').style.display = 'none';

        // Parse and plot the data
        if (id == 1) {
          var fig = JSON.parse(data.data);
          Plotly.newPlot('graph', fig.data, fig.layout);
        }
        else if (id == 2) {
          document.getElementById('graph').innerHTML = "Website is not scrappable";
        }
        else {
          document.getElementById('graph').innerHTML = "No Dark Patterns Found";
        }
      })
      .catch(error => {
        console.error('Error:', error);
        // Hide loading screen
        document.getElementById('loadingScreen').style.display = 'none';
      });
  } else {
    console.error('No active tab found or tab URL is not available.');
  }
});
