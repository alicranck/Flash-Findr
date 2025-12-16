// API Module - Handles all API interactions

export class API {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }

    async uploadFile(formData) {
        const response = await fetch(`${this.baseUrl}/upload_file`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('File upload failed');
        }

        return await response.json();
    }

    async createSession(videoUrl, toolSettings) {
        const payload = {
            video_url: videoUrl,
            pipeline_configuration: {
                tool_settings: toolSettings
            }
        };

        const response = await fetch(`${this.baseUrl}/session/init`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Session init failed");
        }

        return await response.json();
    }

    async initializePipeline(sessionId) {
        const response = await fetch(`${this.baseUrl}/session/${sessionId}/initialize_pipeline`, {
            method: 'POST'
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Pipeline init failed");
        }

        return await response.json();
    }

    async resetSession() {
        const response = await fetch(`${this.baseUrl}/session/reset`, {
            method: 'POST'
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Session reset failed");
        }

        return await response.json();
    }

    getStreamUrl(sessionId) {
        return `${this.baseUrl}/stream/${sessionId}`;
    }

    getWebSocketUrl(sessionId) {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        return `${protocol}//${window.location.host}/api/ws/stream/${sessionId}`;
    }
}
