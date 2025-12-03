// Main Orchestrator - Coordinates all modules

import { UI } from './modules/ui.js';
import { API } from './modules/api.js';
import { Renderer } from './modules/visualization/renderer.js';
import { Stream } from './modules/stream.js';

console.log("Main.js loaded");

// Initialize modules
const ui = new UI();
const api = new API(CONFIG.API_BASE_URL);
const videoStreamImg = document.getElementById('video-stream');
const videoWrapper = document.getElementById('video-wrapper');
const renderer = new Renderer(videoStreamImg, videoWrapper, 'canvas-overlay');
const stream = new Stream(api, renderer, ui);

let currentSessionId = null;

// --- File Upload Handler ---
ui.videoFileInput.addEventListener('change', (event) => {
    const event_target = event.target;
    if (event_target.files && event_target.files.length > 0) {
        // Auto submit
        ui.videoFileForm.requestSubmit();
    }
});

ui.videoFileForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const submitButton = event.submitter || ui.videoFileForm.querySelector('button[type="submit"]');
    if (submitButton) {
        submitButton.disabled = true;
        submitButton.classList.add('loading');
    }

    try {
        const event_target = event.target;
        const formData = new FormData(event_target);
        const file = formData.get('video');
        if (file) {
            const data = await api.uploadFile(formData);
            ui.log(`[SYSTEM] File uploaded successfully: ${data.file_path}`);
            ui.videoUrlInput.value = data.file_path;
            event_target.reset();
        }
    } catch (error) {
        ui.log(`[ERROR] File upload failed: ${error.message}`);
    } finally {
        if (submitButton) {
            submitButton.disabled = false;
            submitButton.classList.remove('loading');
        }
    }
});

// --- Initialize Button Handler ---
ui.initButton.addEventListener('click', async () => {
    const videoUrl = ui.videoUrlInput.value;
    if (!videoUrl) {
        ui.log("[ERROR] No video URL provided.");
        return;
    }

    // Build Configuration
    let toolSettings;
    try {
        toolSettings = ui.getActiveToolConfig();
    } catch (error) {
        ui.log(`[ERROR] ${error.message}`);
        return;
    }

    if (Object.keys(toolSettings).length === 0) {
        ui.log("[WARN] No tools enabled. Stream will pass through raw video.");
    }

    ui.log("[SYSTEM] Creating session...");
    ui.initButton.disabled = true;
    ui.initButton.textContent = "⏳ Creating...";
    ui.initStatus.textContent = "Creating session...";

    try {
        // 1. Create Session
        const data = await api.createSession(videoUrl, toolSettings);
        currentSessionId = data.session_id;
        ui.log(`[SYSTEM] Session created: ${currentSessionId}`);

        // 2. Initialize Pipeline (Load Models)
        ui.log("[SYSTEM] Initializing pipeline (loading models)...");
        ui.initButton.textContent = "⏳ Loading Models...";
        ui.initStatus.textContent = "Loading models... (this may take a moment)";

        await api.initializePipeline(currentSessionId);

        ui.log("[SYSTEM] Pipeline initialized successfully.");
        ui.initStatus.textContent = "System Ready. Click Start to stream.";
        ui.initStatus.style.color = "var(--success)";

        ui.initButton.textContent = "Initialized";
        ui.startButton.disabled = false;
        ui.startButton.classList.add('pulse-animation');

    } catch (error) {
        ui.log(`[ERROR] ${error.message}`);
        ui.initButton.disabled = false;
        ui.initButton.textContent = "Initialize";
        ui.initStatus.textContent = "Initialization failed.";
        ui.initStatus.style.color = "var(--danger)";
        currentSessionId = null;
    }
});

// --- Start Button Handler ---
ui.startButton.addEventListener('click', () => {
    if (!currentSessionId) {
        ui.log("[ERROR] No active session. Initialize first.");
        return;
    }

    ui.log(`[SYSTEM] Starting stream for session: ${currentSessionId}`);
    ui.startButton.disabled = true;
    ui.startButton.textContent = "🔴 Streaming";
    ui.initStatus.textContent = "Streaming active";

    stream.start(currentSessionId);
});

// --- Reset Button Handler ---
ui.resetButton.addEventListener('click', async () => {
    ui.log("[SYSTEM] Resetting session...");
    ui.resetButton.disabled = true;
    ui.resetButton.textContent = "⏳ Resetting...";

    try {
        const data = await api.resetSession();
        ui.log(`[SYSTEM] ${data.message}`);

        // Reset UI state
        currentSessionId = null;
        ui.initButton.disabled = false;
        ui.initButton.textContent = "Initialize";
        ui.startButton.disabled = true;
        ui.startButton.textContent = "Start Stream";
        ui.initStatus.textContent = "System not initialized";
        ui.initStatus.style.color = "";
        ui.setStatus("READY");

        // Stop stream if active
        stream.stop();

        ui.log("[SYSTEM] Ready for new session.");
    } catch (error) {
        ui.log(`[ERROR] Reset failed: ${error.message}`);
    } finally {
        ui.resetButton.disabled = false;
        ui.resetButton.textContent = "Reset";
    }
});