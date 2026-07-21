/**
 * Three.js scene manager — scene, camera, renderer, and render loop.
 */

import * as THREE from 'three';

export class SceneManager {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this._animationId = null;
        this._lastTime = 0;
        this._frameCount = 0;
    }

    init() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0a0f);

        // Camera
        this.camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        this.camera.position.z = 5;

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        document.body.appendChild(this.renderer.domElement);

        // Handle resize
        window.addEventListener('resize', () => this._onResize());

        // Start render loop
        this._animate();
    }

    updateLandmarks(hands) {
        // TODO: Pass to HandOverlay for debug skeleton rendering
    }

    updateEffects(effectState) {
        // TODO: Update PortalMaterial uniforms, trigger effects
    }

    _animate() {
        this._animationId = requestAnimationFrame(() => this._animate());

        // FPS counter
        const now = performance.now();
        this._frameCount++;
        if (now - this._lastTime >= 1000) {
            const fps = Math.round(this._frameCount * 1000 / (now - this._lastTime));
            const fpsEl = document.getElementById('fps');
            if (fpsEl) fpsEl.textContent = `FPS: ${fps}`;
            this._frameCount = 0;
            this._lastTime = now;
        }

        this.renderer.render(this.scene, this.camera);
    }

    _onResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    dispose() {
        if (this._animationId) cancelAnimationFrame(this._animationId);
        this.renderer.dispose();
    }
}
