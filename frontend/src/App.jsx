/**
 * Sanctum frontend application entry point.
 *
 * Initializes the Three.js scene, connects the WebSocket client, and
 * starts the render loop.
 */

import { SceneManager } from './scene/SceneManager.js';
import { WebSocketClient } from './net/WebSocketClient.js';

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws';

// Initialize scene
const sceneManager = new SceneManager();
sceneManager.init();

// Connect to server
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
            console.log(`Gesture: ${msg.gesture_class} (${msg.confidence.toFixed(2)})`);
            break;
        default:
            break;
    }
});

ws.connect();

// Update status overlay
const statusEl = document.getElementById('status');
ws.onOpen(() => { statusEl.textContent = 'Status: connected'; });
ws.onClose(() => { statusEl.textContent = 'Status: disconnected'; });
