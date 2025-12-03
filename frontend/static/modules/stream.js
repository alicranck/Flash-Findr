// Stream Module - Handles video streaming and WebSocket connections

export class Stream {
    constructor(api, renderer, ui) {
        this.api = api;
        this.renderer = renderer;
        this.ui = ui;
        this.videoStreamImg = ui.videoUrlInput.nextElementSibling || document.getElementById('video-stream');
        this.placeholderOverlay = document.getElementById('placeholder-overlay');
        this.ws = null;
    }

    start(sessionId) {
        const streamUrl = this.api.getStreamUrl(sessionId);
        const wsUrl = this.api.getWebSocketUrl(sessionId);

        // 1. Start Video Stream
        this.videoStreamImg.onload = () => {
            this.ui.log("[SYSTEM] Stream connection established.");
            this.ui.setStatus("LIVE");
            this.placeholderOverlay.style.display = 'none';
            this.ui.startButton.textContent = "🔴 Stream Active";

            // Init Konva once video is ready
            if (!this.renderer.stage) {
                this.renderer.initKonva();
            }
        };

        this.videoStreamImg.onerror = () => {
            this.ui.log("[ERROR] Stream connection lost or failed.");
            this.ui.setStatus("ERROR");
            this.ui.startButton.disabled = false;
            this.ui.startButton.textContent = "🚀 Retry Stream";
        };

        this.videoStreamImg.src = streamUrl;

        // 2. Start WebSocket for Metadata
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            this.ui.log("[WS] Connected to metadata stream.");
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.renderer.drawMetadata(data);
        };

        this.ws.onerror = (err) => {
            this.ui.log("[WS] Error: " + err);
        };
    }

    stop() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
}
