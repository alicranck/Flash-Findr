// UI Module - Handles UI state, tool selection, and logging

export class UI {
    constructor() {
        // UI Elements
        this.videoUrlInput = document.getElementById('videoUrl');
        this.startButton = document.getElementById('startButton');
        this.initButton = document.getElementById('initButton');
        this.resetButton = document.getElementById('resetButton');
        this.initStatus = document.getElementById('init-status');
        this.logContent = document.getElementById('log-content');
        this.statusIndicator = document.getElementById('status-indicator');
        this.videoFileInput = document.getElementById('videoFile');
        this.videoFileForm = document.getElementById('videoFileForm');

        // Tool Cards
        this.toggleDetection = document.getElementById('toggle-detection');
        this.cardDetection = document.getElementById('card-detection');
        this.bodyDetection = document.getElementById('body-detection');

        this.toggleCaptioning = document.getElementById('toggle-captioning');
        this.cardCaptioning = document.getElementById('card-captioning');
        this.bodyCaptioning = document.getElementById('body-captioning');

        this.togglePose = document.getElementById('toggle-pose');
        this.cardPose = document.getElementById('card-pose');
        this.bodyPose = document.getElementById('body-pose');

        // Inputs
        this.classesInput = document.getElementById('classes');
        this.confidenceInput = document.getElementById('confidence');
        this.confValDisplay = document.getElementById('conf-val');

        // Tools configuration
        this.tools = [
            { toggle: this.toggleDetection, card: this.cardDetection, body: this.bodyDetection, name: 'ov_detection' },
            { toggle: this.toggleCaptioning, card: this.cardCaptioning, body: this.bodyCaptioning, name: 'captioning' },
            { toggle: this.togglePose, card: this.cardPose, body: this.bodyPose, name: 'pose_estimation' }
        ];

        this.setupToolToggles();
        this.setupConfidenceSlider();
    }

    setupToolToggles() {
        this.tools.forEach(tool => {
            tool.toggle.addEventListener('change', (e) => {
                if (e.target.checked) {
                    // Disable others
                    this.tools.forEach(t => {
                        if (t !== tool) {
                            t.toggle.checked = false;
                            this.toggleTool(t.card, t.body, false);
                        }
                    });
                    this.toggleTool(tool.card, tool.body, true);
                } else {
                    this.toggleTool(tool.card, tool.body, false);
                }
            });
        });
    }

    setupConfidenceSlider() {
        this.confidenceInput.addEventListener('input', (e) => {
            this.confValDisplay.textContent = e.target.value;
        });
    }

    toggleTool(card, body, isActive) {
        if (isActive) {
            card.classList.add('active-card');
            body.classList.remove('disabled-body');
            this.log(`[UI] Tool enabled: ${card.id}`);
        } else {
            card.classList.remove('active-card');
            body.classList.add('disabled-body');
            this.log(`[UI] Tool disabled: ${card.id}`);
        }
    }

    log(message) {
        console.log(message);
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        const time = new Date().toLocaleTimeString();
        entry.textContent = `[${time}] ${message}`;
        this.logContent.appendChild(entry);
        this.logContent.scrollTop = this.logContent.scrollHeight;
    }

    setStatus(status) {
        this.statusIndicator.className = `status-indicator status-${status.toLowerCase()}`;
        this.statusIndicator.textContent = status;
    }

    getActiveToolConfig() {
        const toolSettings = {};

        // Detection Config
        if (this.toggleDetection.checked) {
            const vocab = this.classesInput.value.split(',').map(s => s.trim()).filter(s => s);
            if (vocab.length === 0) {
                throw new Error("Detection enabled but vocabulary is empty.");
            }
            toolSettings.ov_detection = {
                vocabulary: vocab,
                imgsz: 416,
                conf_threshold: parseFloat(this.confidenceInput.value),
                trigger: { "type": "stride", "value": 3 }
            };
        }

        // Captioning Config
        if (this.toggleCaptioning.checked) {
            toolSettings.captioning = {
                imgsz: 480,
                trigger: { "type": "scene_change", "threshold": 0.2 }
            };
        }

        // Pose Config
        if (this.togglePose.checked) {
            toolSettings.pose_estimation = {
                imgsz: 640,
                conf_threshold: 0.5
            };
        }

        return toolSettings;
    }
}
