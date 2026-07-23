/**
 * Sanctum frontend application entry point.
 *
 * Requests local webcam video stream for AR background, initializes Three.js WebGL canvas,
 * and handles WebSocket streams from the Python backend.
 */

import { SceneManager } from './scene/SceneManager.js';
import { WebSocketClient } from './net/WebSocketClient.js';

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws';

// 1. Initialize transparent Three.js WebGL scene
const sceneManager = new SceneManager();
sceneManager.init();

// 2. Request webcam video stream for local AR background display
const videoEl = document.getElementById('webcam');
const cameraEl = document.getElementById('camera');

async function initWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'user',
            },
            audio: false,
        });

        if (videoEl) {
            videoEl.srcObject = stream;
            await videoEl.play();
        }

        if (cameraEl) cameraEl.textContent = 'Camera: active';
    } catch (err) {
        console.warn('[Camera] Local webcam access denied or unavailable:', err);
        if (cameraEl) cameraEl.textContent = 'Camera: offline / synthetic';
    }
}

initWebcam();

// 3. Connect to Python FastAPI WebSocket backend
const ws = new WebSocketClient(WS_URL);
ws.onMessage((msg) => {
    switch (msg.type) {
        case 'landmarks':
            sceneManager.updateLandmarks(msg.hands);
            break;
        case 'effect_state':
            sceneManager.updateEffects(msg);
            break;
        case 'gesture':
            console.log(`[Gesture Detected] ${msg.gesture_class} (${(msg.confidence * 100).toFixed(1)}%) -> Action: ${msg.action}`);
            break;
        default:
            break;
    }
});

ws.connect();

// Update status HUD
const statusEl = document.getElementById('status');
ws.onOpen(() => { if (statusEl) statusEl.textContent = 'Status: connected'; });
ws.onClose(() => { if (statusEl) statusEl.textContent = 'Status: disconnected'; });
