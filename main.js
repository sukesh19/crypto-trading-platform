// main.js
document.addEventListener('DOMContentLoaded', () => {
    // ... (previous JavaScript logic) ...

    const mediaDisplay = document.getElementById('media-content');
    const displayButtons = document.querySelectorAll('.display-media');

    displayButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetId = button.dataset.target;
            const targetElement = document.getElementById(targetId);

            if (targetElement) {
                mediaDisplay.innerHTML = ''; // Clear previous content

                if (targetId === 'stock-table') {
                    mediaDisplay.innerHTML = targetElement.outerHTML;
                } else if(targetId === 'controls'){
                    mediaDisplay.innerHTML = document.getElementById('controls').outerHTML;
                }
                else if (targetId === 'news-container') {
                    mediaDisplay.innerHTML = document.getElementById('news').outerHTML;
                }
                else if (targetId === 'stockChart' || targetId === 'returnsChart') {
                    const canvasCopy = targetElement.cloneNode(true);
                    mediaDisplay.appendChild(canvasCopy);
                    // Re-render the chart if needed (depends on how you manage charts)
                    if (targetId === 'stockChart') {
                        // Example re-render (adapt to your chart management)
                        const newCtx = canvasCopy.getContext('2d');
                        new Chart(newCtx, {
                            type: 'line',
                            data: {
                                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
                                datasets: [{
                                    label: 'Stock Price',
                                    data: [12, 19, 3, 5, 2],
                                    borderColor: 'rgb(75, 192, 192)',
                                    tension: 0.1
                                }]
                            },
                            options: {
                                scales: {
                                    y: {
                                        beginAtZero: false
                                    }
                                }
                            }
                        });
                    }
                    else if (targetId === 'returnsChart'){
                        const newCtx = canvasCopy.getContext('2d');
                        new Chart(newCtx, {
                            type: 'line',
                            data: {
                                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
                                datasets: [{
                                    label: 'Returns',
                                    data: [5, 10, 8, 12, 15],
                                    borderColor: 'rgb(255, 99, 132)',
                                    tension: 0.1
                                }]
                            },
                            options: {
                                scales: {
                                    y: {
                                        beginAtZero: false
                                    }
                                }
                            }
                        });
                    }
                }
            }
        });
    });

    // ... (chart rendering logic) ...
});