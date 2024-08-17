document.getElementById('getDataButton').addEventListener('click', function (e) {
  e.preventDefault();
  // Show loading screen
  document.getElementById('loadingScreen').style.display = 'flex';
  link = document.getElementById("link").value;
  console.log(link);
  fetch('/analysis', {  // Replace with your backend URL
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ link: link })
  })
    .then(response => response.json())
    .then(data => {
      const id = data.id;
      document.getElementById('loadingScreen').style.display = 'none';
      // Parse and plot the data
      if (id == 1) {
        var fig = JSON.parse(data.data);
        document.getElementById('graph-container').innerHTML = "";
        Plotly.newPlot('graph-container', fig.data, fig.layout);
      }
      else if (id == 2) {
        document.getElementById('graph-container').innerHTML = "Website is not scrappable";
      }
      else {
        document.getElementById('graph-container').innerHTML = "No Dark Patterns Found";
      }
    })
    .catch(error => {
      console.error('Error:', error);
      // Hide loading screen
      document.getElementById('loadingScreen').style.display = 'none';
    });
});