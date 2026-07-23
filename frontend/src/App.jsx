/**
 * Sanctum frontend application entry point.
 *
 * Initializes Three.js WebGL canvas and handles WebSocket streams from the Python backend.
 * Camera video is streamed from the Python server (not browser getUserMedia).
 */

import { SceneManager } from './scene/SceneManager.js';
import { WebSocketClient } from './net/WebSocketClient.js';

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws';

// 1. Initialize transparent Three.js WebGL scene
const sceneManager = new SceneManager();
sceneManager.init();

// 2. DOM Elements
const videoImgEl = document.getElementById('webcam');
const handsEl = document.getElementById('hands');
const gestureBanner = document.getElementById('gesture-banner');
let gestureBannerTimeout = null;

function showGestureBanner(text) {
    if (gestureBanner) {
        gestureBanner.textContent = text;
        gestureBanner.style.display = 'block';
        if (gestureBannerTimeout) clearTimeout(gestureBannerTimeout);
        gestureBannerTimeout = setTimeout(() => {
            gestureBanner.style.display = 'none';
        }, 2000);
    }
}

// 3. Connect to Python FastAPI WebSocket backend
const ws = new WebSocketClient(WS_URL);
ws.onMessage((msg) => {
    switch (msg.type) {
        case 'landmarks':
            // Update live camera video background
            if (msg.image_b64 && videoImgEl) {
                videoImgEl.src = msg.image_b64;
            }
            // Update hand count HUD
            if (handsEl) {
                const count = msg.hands ? msg.hands.length : 0;
                handsEl.textContent = `Hands: ${count}`;
                handsEl.style.color = count > 0 ? '#00ff88' : '#ff4444';
            }
            // Update 3D skeleton overlay
            sceneManager.updateLandmarks(msg.hands);
            break;
        case 'effect_state':
            sceneManager.updateEffects(msg);
            break;
        case 'gesture':
            console.log(`[Gesture] ${msg.gesture_class} (${(msg.confidence * 100).toFixed(1)}%) -> ${msg.action}`);
            showGestureBanner(`✨ ${msg.gesture_class.toUpperCase()} → ${msg.action}`);
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
