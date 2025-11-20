// --- UI Elements ---
const videoUrlInput = document.getElementById('videoUrl');
const startButton = document.getElementById('startButton');
const videoStreamImg = document.getElementById('video-stream');
const placeholderOverlay = document.getElementById('placeholder-overlay');
const logContent = document.getElementById('log-content');
const statusIndicator = document.getElementById('status-indicator');

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

// --- Stream Logic ---
startButton.addEventListener('click', async () => {
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
        toolSettings.detection = {
            vocabulary: vocab,
            conf_threshold: parseFloat(confidenceInput.value),
            trigger: { "type": "stride", "value": 3 }
        };
    }

    // Captioning Config
    if (toggleCaptioning.checked) {
        toolSettings.captioning = {
            trigger: { "type": "scene_change", "threshold": 0.33 }
        };
    }

    if (Object.keys(toolSettings).length === 0) {
        log("[WARN] No tools enabled. Stream will pass through raw video.");
    }

    const payload = {
        video_url: videoUrl,
        pipeline: {
            tool_settings: toolSettings
        }
    };

    log("[SYSTEM] Initializing session...");
    startButton.disabled = true;
    startButton.textContent = "⏳ Initializing...";

    try {
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
        const sessionId = data.session_id;

        log(`[SYSTEM] Session created: ${sessionId}`);
        startStream(sessionId);

    } catch (error) {
        log(`[ERROR] ${error.message}`);
        startButton.disabled = false;
        startButton.textContent = "🚀 Initialize Stream";
    }
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
        log(data);
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
        // Optional: Auto-hide if no caption is sent for a while?
        // For now, we keep it until a new one comes or explicitly cleared.
        // If you want to clear it when data.caption is missing/null:
        // captionOverlay.classList.add('hidden');
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

            // Label (Visible only on hover)
            const label = new Konva.Text({
                x: x,
                y: y - 10,
                text: `${box.conf.toFixed(2)}`, // You might want class name here too if available in map
                fontSize: 16,
                fill: '#fff',
                fontFamily: 'monospace',
                padding: 4,
                background: color, // Konva Text doesn't support background directly like this, need a Label group
                visible: isHover
            });

            // Better Label: Group with Tag and Text
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

                tooltip.add(new Konva.Text({
                    text: `${data.class_names[box.cls]}: ${box.conf.toFixed(2)}`,
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

    // Draw Masks
    // if (data.masks) {
    //     data.masks.forEach(mask => {
    //         // mask is a list of [x, y] points
    //         const points = mask.flatMap(pt => [pt[0] * scaleX, pt[1] * scaleY]);

    //         const poly = new Konva.Line({
    //             points: points,
    //             fill: 'rgba(0, 242, 255, 0.2)', // Semi-transparent cyan
    //             stroke: 'cyan',
    //             strokeWidth: 1,
    //             closed: true
    //         });
    //         layer.add(poly);
    //     });
    // }

    layer.batchDraw();
}

// Helper for colors
const colors = ['#00f2ff', '#7000ff', '#00ff9d', '#ffbd2e', '#ff0055'];
function getColor(idx) {
    return colors[idx % colors.length];
}