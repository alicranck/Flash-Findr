// --- UI Elements ---
console.log("Main.js loaded"); // Debugging
const videoUrlInput = document.getElementById('videoUrl');
const startButton = document.getElementById('startButton');
const videoStreamImg = document.getElementById('video-stream');
const placeholderOverlay = document.getElementById('placeholder-overlay');
const logContent = document.getElementById('log-content');
const statusIndicator = document.getElementById('status-indicator');
const videoFileInput = document.getElementById('videoFile');
const videoFileForm = document.getElementById('videoFileForm');

// Tool Cards
const toggleDetection = document.getElementById('toggle-detection');
const cardDetection = document.getElementById('card-detection');
const bodyDetection = document.getElementById('body-detection');

const toggleCaptioning = document.getElementById('toggle-captioning');
const cardCaptioning = document.getElementById('card-captioning');
const bodyCaptioning = document.getElementById('body-captioning');

// Inputs
const classesInput = document.getElementById('classes');
const confidenceInput = document.getElementById('confidence');
const confValDisplay = document.getElementById('conf-val');

// --- Event Listeners ---

// Toggle Logic
toggleDetection.addEventListener('change', (e) => {
    toggleTool(cardDetection, bodyDetection, e.target.checked);
});

toggleCaptioning.addEventListener('change', (e) => {
    toggleTool(cardCaptioning, bodyCaptioning, e.target.checked);
});

confidenceInput.addEventListener('input', (e) => {
    confValDisplay.textContent = e.target.value;
});

function toggleTool(card, body, isActive) {
    if (isActive) {
        card.classList.add('active-card');
        body.classList.remove('disabled-body');
        log(`[UI] Tool enabled: ${card.id}`);
    } else {
        card.classList.remove('active-card');
        body.classList.add('disabled-body');
        log(`[UI] Tool disabled: ${card.id}`);
    }
}

// --- Logging System ---
function log(message) {
    console.log(message); // Debugging
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    const time = new Date().toLocaleTimeString();
    entry.textContent = `[${time}] ${message}`;
    logContent.appendChild(entry);
    logContent.scrollTop = logContent.scrollHeight;
}

function setStatus(status) {
    statusIndicator.className = `status-indicator status-${status.toLowerCase()}`;
    statusIndicator.textContent = status;
}


// --- Video File Upload ---
videoFileInput.addEventListener('change', (event) => {
    event_target = event.target;
    if (event_target.files && event_target.files.length > 0) {
        videoUrlInput.value = `${event_target.files[0].name}`;
    }
});

videoFileForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const submitButton = event.submitter || videoFileForm.querySelector('button[type="submit"]');
    if (submitButton) {
        submitButton.disabled = true;
        submitButton.classList.add('loading');
    }

    try {
        const event_target = event.target;
        const formData = new FormData(event_target);
        const file = formData.get('video');
        if (file) {
            const response = await fetch('/upload_file', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            log(`[SYSTEM] File uploaded successfully: ${data.file_path}`);
            videoUrlInput.value = data.file_path;
            event_target.reset();
        }
    } catch (error) {
        log(`[ERROR] File upload failed: ${error.message}`);
    } finally {
        if (submitButton) {
            submitButton.disabled = false;
            submitButton.classList.remove('loading');
        }
    }
});

// --- Stream Logic ---
let currentSessionId = null;
const initButton = document.getElementById('initButton');
const initStatus = document.getElementById('init-status');

initButton.addEventListener('click', async () => {
    const videoUrl = videoUrlInput.value;
    if (!videoUrl) {
        log("[ERROR] No video URL provided.");
        return;
    }

    // Build Configuration
    const toolSettings = {};

    // Detection Config
    if (toggleDetection.checked) {
        const vocab = classesInput.value.split(',').map(s => s.trim()).filter(s => s);
        if (vocab.length === 0) {
            log("[ERROR] Detection enabled but vocabulary is empty.");
            return;
        }
        toolSettings.ov_detection = {
            vocabulary: vocab,
            imgsz: 416,
            conf_threshold: parseFloat(confidenceInput.value),
            trigger: { "type": "stride", "value": 3 }
        };
    }

    // Captioning Config
    if (toggleCaptioning.checked) {
        toolSettings.captioning = {
            imgsz: 480,
            trigger: { "type": "scene_change", "threshold": 0.2 }
        };
    }

    if (Object.keys(toolSettings).length === 0) {
        log("[WARN] No tools enabled. Stream will pass through raw video.");
    }

    const payload = {
        video_url: videoUrl,
        pipeline_configuration: {
            tool_settings: toolSettings
        }
    };

    log("[SYSTEM] Creating session...");
    initButton.disabled = true;
    initButton.textContent = "⏳ Creating...";
    initStatus.textContent = "Creating session...";

    try {
        // 1. Create Session
        const response = await fetch('/session/init', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Session init failed");
        }

        const data = await response.json();
        currentSessionId = data.session_id;
        log(`[SYSTEM] Session created: ${currentSessionId}`);

        // 2. Initialize Pipeline (Load Models)
        log("[SYSTEM] Initializing pipeline (loading models)...");
        initButton.textContent = "⏳ Loading Models...";
        initStatus.textContent = "Loading models... (this may take a moment)";

        const initResponse = await fetch(`/session/${currentSessionId}/initialize_pipeline`, {
            method: 'POST'
        });

        if (!initResponse.ok) {
            const err = await initResponse.json();
            throw new Error(err.detail || "Pipeline init failed");
        }

        log("[SYSTEM] Pipeline initialized successfully.");
        initStatus.textContent = "System Ready. Click Start to stream.";
        initStatus.style.color = "var(--success)";

        initButton.textContent = "✅ Initialized";
        startButton.disabled = false;
        startButton.classList.add('pulse-animation'); // Optional visual cue

    } catch (error) {
        log(`[ERROR] ${error.message}`);
        initButton.disabled = false;
        initButton.textContent = "⚙️ Initialize";
        initStatus.textContent = "Initialization failed.";
        initStatus.style.color = "var(--danger)";
        currentSessionId = null;
    }
});

startButton.addEventListener('click', () => {
    if (!currentSessionId) {
        log("[ERROR] No active session. Initialize first.");
        return;
    }

    log(`[SYSTEM] Starting stream for session: ${currentSessionId}`);
    startButton.disabled = true;
    startButton.textContent = "🔴 Streaming";
    initStatus.textContent = "Streaming active";

    startStream(currentSessionId);
});

// --- Visualization Logic (Konva) ---
let stage, layer;
const canvasOverlay = document.getElementById('canvas-overlay');
const videoWrapper = document.getElementById('video-wrapper');

function initKonva() {
    stage = new Konva.Stage({
        container: 'canvas-overlay',
        width: videoWrapper.clientWidth,
        height: videoWrapper.clientHeight,
    });
    layer = new Konva.Layer();
    stage.add(layer);
}

// Resize Observer to keep canvas synced with video container
const resizeObserver = new ResizeObserver(() => {
    if (stage) {
        stage.width(videoWrapper.clientWidth);
        stage.height(videoWrapper.clientHeight);
    }
});
resizeObserver.observe(videoWrapper);


function startStream(sessionId) {
    const streamUrl = `/stream/${sessionId}`;
    const wsUrl = `ws://${window.location.host}/ws/stream/${sessionId}`;

    // 1. Start Video Stream
    videoStreamImg.onload = () => {
        log("[SYSTEM] Stream connection established.");
        setStatus("LIVE");
        placeholderOverlay.style.display = 'none';
        startButton.textContent = "🔴 Stream Active";

        // Init Konva once video is ready
        if (!stage) initKonva();
    };

    videoStreamImg.onerror = () => {
        log("[ERROR] Stream connection lost or failed.");
        setStatus("ERROR");
        startButton.disabled = false;
        startButton.textContent = "🚀 Retry Stream";
    };

    videoStreamImg.src = streamUrl;

    // 2. Start WebSocket for Metadata
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        log("[WS] Connected to metadata stream.");
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        drawMetadata(data);
    };

    ws.onerror = (err) => {
        log("[WS] Error: " + err);
    };
}

const captionOverlay = document.getElementById('caption-overlay');

function drawMetadata(data) {
    if (!layer) return;
    layer.destroyChildren(); // Clear previous frame

    // --- Caption Handling ---
    if (data.caption) {
        captionOverlay.textContent = data.caption;
        captionOverlay.classList.remove('hidden');
    } else {
    }

    // Calculate scaling factors
    const naturalW = videoStreamImg.naturalWidth || 640;
    const naturalH = videoStreamImg.naturalHeight || 640;
    const displayedW = videoWrapper.clientWidth;
    const displayedH = videoWrapper.clientHeight;

    const scaleX = displayedW / naturalW;
    const scaleY = displayedH / naturalH;

    // Get current pointer position for hover logic
    const pointer = stage.getPointerPosition();

    // Draw Boxes with names
    if (data.boxes) {
        data.boxes.forEach(box => {
            const [x1, y1, x2, y2] = box.xyxy;
            const w = (x2 - x1) * scaleX;
            const h = (y2 - y1) * scaleY;
            const x = x1 * scaleX;
            const y = y1 * scaleY;

            const color = getColor(box.cls);

            // Check for hover
            let isHover = false;
            if (pointer) {
                if (pointer.x >= x && pointer.x <= x + w &&
                    pointer.y >= y && pointer.y <= y + h) {
                    isHover = true;
                }
            }

            // Rect
            const rect = new Konva.Rect({
                x: x,
                y: y,
                width: w,
                height: h,
                stroke: color,
                strokeWidth: isHover ? 4 : 2, // Thicker stroke on hover
                shadowColor: color,
                shadowBlur: isHover ? 15 : 0, // Glow on hover
                name: 'bbox'
            });

            if (isHover) {
                const tooltip = new Konva.Label({
                    x: x,
                    y: y - 10,
                    opacity: 1
                });

                tooltip.add(new Konva.Tag({
                    fill: color,
                    pointerDirection: 'down',
                    pointerWidth: 10,
                    pointerHeight: 10,
                    lineJoin: 'round',
                    shadowColor: 'black',
                    shadowBlur: 10,
                    shadowOffset: { x: 10, y: 10 },
                    shadowOpacity: 0.5
                }));

                const className = (data.class_names && data.class_names[box.cls]) ? data.class_names[box.cls] : box.cls;
                tooltip.add(new Konva.Text({
                    text: `${className} ${box.id}: ${box.conf.toFixed(2)}`,
                    fontFamily: 'monospace',
                    fontSize: 14,
                    padding: 5,
                    fill: 'black'
                }));

                layer.add(rect);
                layer.add(tooltip);
            } else {
                layer.add(rect);
            }
        });
    }
    layer.batchDraw();
}

// Helper for colors
const colors = ['#00f2ff', '#7000ff', '#00ff9d', '#ffbd2e', '#ff0055'];
function getColor(idx) {
    return colors[idx % colors.length];
}