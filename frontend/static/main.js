// Get elements for easier access
const form = document.getElementById('detection-form');
const videoStreamImg = document.getElementById('video-stream');
const startButton = document.getElementById('startButton');
const statusBox = document.getElementById('status-box');
const statusMessage = document.getElementById('status-message');
const placeholderText = document.getElementById('placeholder-text');

const apiUrl = "/stream/start"; 

// Initial setup
statusBox.className = 'status-box status-ready';

// Function to update status with new style
function updateStatus(message, state) {
    statusMessage.textContent = message;
    statusBox.className = 'status-box status-' + state;
    
    // Update icon based on state
    const icon = statusBox.querySelector('.status-icon');
    if (state === 'live') icon.textContent = '🟢';
    else if (state === 'ready') icon.textContent = '💡';
    else if (state === 'pending') icon.textContent = '⏳';
    else if (state === 'error') icon.textContent = '❌';
}

// --- Stream Error Handling ---
videoStreamImg.onerror = function() {
    updateStatus('ERROR: Stream failed or ended. Check video URL and FastAPI logs.', 'error');
    startButton.disabled = false; 
    videoStreamImg.style.display = 'none';
    placeholderText.style.display = 'block'; // Show placeholder again
    videoStreamImg.src = ""; 
};


// --- Form Submission Logic ---
form.addEventListener('submit', function(e) {
    e.preventDefault();
    
    const videoUrl = document.getElementById('videoUrl').value;
    const classesText = document.getElementById('classes').value;
    const classes = classesText.split(',').map(c => c.trim()).filter(c => c.length > 0);
    
    if (classes.length === 0) {
        updateStatus('ERROR: Please enter at least one detection class.', 'error');
        return;
    }

    const classesTextJoined = classes.join(','); 
    const streamURL = `${apiUrl}?video_url=${encodeURIComponent(videoUrl)}&classes=${encodeURIComponent(classesTextJoined)}`;

    // 1. Set PENDING status
    updateStatus('Starting stream and compiling vocabulary...', 'pending');
    startButton.disabled = true;
    placeholderText.style.display = 'none'; // Hide placeholder
    
    // 2. Set the stream source
    videoStreamImg.src = streamURL;
    videoStreamImg.style.display = 'block';
    
    // 3. Set LIVE status after a short delay (allowing the backend time to start processing)
    // Note: This is an *optimistic* update. The onerror handler catches true failure.
    setTimeout(() => {
        updateStatus('Stream LIVE! Detection running.', 'live');
    }, 1000); 
});