/**
 * Sanctum frontend application entry point.
 *
 * Initializes Three.js WebGL canvas and handles WebSocket streams from the Python backend.
 */

import { SceneManager } from './scene/SceneManager.js';
import { WebSocketClient } from './net/WebSocketClient.js';

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws';

// 1. Initialize transparent Three.js WebGL scene
const sceneManager = new SceneManager();
sceneManager.init();

// 2. Camera background element
const videoImgEl = document.getElementById('webcam');

// 3. Connect to Python FastAPI WebSocket backend
const ws = new WebSocketClient(WS_URL);
ws.onMessage((msg) => {
    switch (msg.type) {
        case 'landmarks':
            // Update 3D skeleton joints
            sceneManager.updateLandmarks(msg.hands);
            // Update live camera video background frame
            if (msg.image_b64 && videoImgEl) {
                videoImgEl.src = msg.image_b64;
            }
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
