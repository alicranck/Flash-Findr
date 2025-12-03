// Renderer Module - Handles Konva setup and coordinates visualization

import { drawBoxes } from './boxes.js';
import { drawPoses } from './poses.js';

export class Renderer {
    constructor(videoStreamImg, videoWrapper, canvasOverlayId) {
        this.videoStreamImg = videoStreamImg;
        this.videoWrapper = videoWrapper;
        this.canvasOverlayId = canvasOverlayId;
        this.stage = null;
        this.layer = null;
        this.captionOverlay = document.getElementById('caption-overlay');

        this.setupResizeObserver();
    }

    initKonva() {
        this.stage = new Konva.Stage({
            container: this.canvasOverlayId,
            width: this.videoWrapper.clientWidth,
            height: this.videoWrapper.clientHeight,
        });
        this.layer = new Konva.Layer();
        this.stage.add(this.layer);
    }

    setupResizeObserver() {
        const resizeObserver = new ResizeObserver(() => {
            if (this.stage) {
                this.stage.width(this.videoWrapper.clientWidth);
                this.stage.height(this.videoWrapper.clientHeight);
            }
        });
        resizeObserver.observe(this.videoWrapper);
    }

    calculateScaling() {
        // Calculate scaling factors based on actual displayed image dimensions
        const naturalW = this.videoStreamImg.naturalWidth || 640;
        const naturalH = this.videoStreamImg.naturalHeight || 640;

        // Use the actual displayed dimensions of the image element
        const displayedW = this.videoStreamImg.offsetWidth || this.videoStreamImg.clientWidth;
        const displayedH = this.videoStreamImg.offsetHeight || this.videoStreamImg.clientHeight;

        const scaleX = displayedW / naturalW;
        const scaleY = displayedH / naturalH;

        // Calculate offset if image is centered in container
        const containerW = this.videoWrapper.clientWidth;
        const containerH = this.videoWrapper.clientHeight;
        const offsetX = (containerW - displayedW) / 2;
        const offsetY = (containerH - displayedH) / 2;

        return { scaleX, scaleY, offsetX, offsetY };
    }

    drawMetadata(data) {
        if (!this.layer) return;
        this.layer.destroyChildren(); // Clear previous frame

        // --- Caption Handling ---
        if (data.caption) {
            this.captionOverlay.textContent = data.caption;
            this.captionOverlay.classList.remove('hidden');
        }

        const { scaleX, scaleY, offsetX, offsetY } = this.calculateScaling();

        // Get current pointer position for hover logic
        const pointer = this.stage.getPointerPosition();

        // Draw Boxes
        drawBoxes(this.layer, data, scaleX, scaleY, offsetX, offsetY, pointer);

        // Draw Poses
        drawPoses(this.layer, data, scaleX, scaleY, offsetX, offsetY);

        this.layer.batchDraw();
    }
}
