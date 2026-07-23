/**
 * Three.js scene manager — scene, camera, renderer, and render loop.
 *
 * Manages rendering of hand skeleton overlays and portal VFX materials
 * synced to incoming WebSocket data.
 */

import * as THREE from 'three';
import { HandOverlay } from './HandOverlay.js';
import { PortalMaterial } from './PortalMaterial.js';

export class SceneManager {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.handOverlay = null;
        this.portalMaterial = null;
        this.portalMesh = null;
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

        // 1. Hand skeleton overlay
        this.handOverlay = new HandOverlay(this.scene);

        // 2. Portal VFX material & circle mesh
        this.portalMaterial = new PortalMaterial();
        const portalGeo = new THREE.CircleGeometry(1, 64);
        this.portalMesh = new THREE.Mesh(portalGeo, this.portalMaterial);
        this.portalMesh.visible = false;
        this.scene.add(this.portalMesh);

        // Handle resize
        window.addEventListener('resize', () => this._onResize());

        // Start render loop
        this._animate();
    }

    updateLandmarks(hands) {
        if (this.handOverlay) {
            this.handOverlay.update(hands);
        }
    }

    updateEffects(effectState) {
        if (!effectState) return;

        if (effectState.portal && effectState.portal.active) {
            this.portalMesh.visible = true;
            const nowSec = performance.now() / 1000.0;
            this.portalMaterial.update(nowSec, effectState.portal);

            if (effectState.portal.center) {
                const cx = effectState.portal.center[0];
                const cy = effectState.portal.center[1];
                // Map normalized coords (0..1) to Three.js camera space
                this.portalMesh.position.set(
                    (cx - 0.5) * 4.0,
                    -(cy - 0.5) * 3.0,
                    0.0
                );
            }
        } else {
            this.portalMesh.visible = false;
        }
    }

    _animate() {
        this._animationId = requestAnimationFrame(() => this._animate());

        // Update portal shader time uniform per frame if visible
        if (this.portalMesh && this.portalMesh.visible) {
            this.portalMaterial.uniforms.uTime.value = performance.now() / 1000.0;
        }

        // FPS counter
        const now = performance.now();
        this._frameCount++;
        if (now - this._lastTime >= 1000) {
            const fps = Math.round((this._frameCount * 1000) / (now - this._lastTime));
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
        if (this.handOverlay) this.handOverlay.dispose();
        if (this.portalMaterial) this.portalMaterial.dispose();
        this.renderer.dispose();
    }
}
