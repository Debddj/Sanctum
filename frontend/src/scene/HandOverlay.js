/**
 * Debug hand landmark skeleton overlay for Three.js scene.
 *
 * Renders MediaPipe hand landmarks as spheres and connected line segments
 * matching the 21-point hand topology. Expects RAW MediaPipe coordinates
 * in 0-1 normalized range (NOT wrist-relative).
 */

import * as THREE from 'three';

const HAND_CONNECTIONS = [
    // Thumb
    [0, 1], [1, 2], [2, 3], [3, 4],
    // Index finger
    [0, 5], [5, 6], [6, 7], [7, 8],
    // Middle finger
    [5, 9], [9, 10], [10, 11], [11, 12],
    // Ring finger
    [9, 13], [13, 14], [14, 15], [15, 16],
    // Pinky finger
    [0, 17], [13, 17], [17, 18], [18, 19], [19, 20],
];

export class HandOverlay {
    constructor(scene) {
        this.scene = scene;
        this.handsGroup = new THREE.Group();
        this.scene.add(this.handsGroup);

        this._jointMaterial = new THREE.MeshBasicMaterial({ color: 0x00ffff });
        this._boneMaterial = new THREE.LineBasicMaterial({ color: 0x00ff88, linewidth: 2 });
        this._jointGeometry = new THREE.SphereGeometry(0.04, 8, 8);
    }

    update(hands) {
        // Clear previous frame mesh objects
        while (this.handsGroup.children.length > 0) {
            const child = this.handsGroup.children.pop();
            if (child.geometry) child.geometry.dispose();
        }

        if (!hands || hands.length === 0) return;

        hands.forEach((hand) => {
            const landmarks = hand.landmarks;
            if (!landmarks || landmarks.length < 21) return;

            // 1. Draw joint spheres
            const positions = landmarks.map((lm) => {
                // lm is [x, y, z] in 0-1 MediaPipe normalized coords
                const x = lm[0];
                const y = lm[1];
                const z = lm[2] || 0;

                // Map to Three.js camera-space:
                // Mirror X to match selfie-mirrored video
                // Scale to fill visible frustum at z=5
                const vec = new THREE.Vector3(
                    -(x - 0.5) * 8.0,   // mirror + scale
                    -(y - 0.5) * 6.0,   // flip Y (screen Y is top-down)
                    -z * 2.0
                );
                const jointMesh = new THREE.Mesh(this._jointGeometry, this._jointMaterial);
                jointMesh.position.copy(vec);
                this.handsGroup.add(jointMesh);
                return vec;
            });

            // 2. Draw connecting bone lines
            const linePositions = [];
            HAND_CONNECTIONS.forEach(([startIdx, endIdx]) => {
                const p1 = positions[startIdx];
                const p2 = positions[endIdx];
                if (p1 && p2) {
                    linePositions.push(p1.x, p1.y, p1.z);
                    linePositions.push(p2.x, p2.y, p2.z);
                }
            });

            if (linePositions.length > 0) {
                const lineGeometry = new THREE.BufferGeometry();
                lineGeometry.setAttribute(
                    'position',
                    new THREE.Float32BufferAttribute(linePositions, 3)
                );
                const lineMesh = new THREE.LineSegments(lineGeometry, this._boneMaterial);
                this.handsGroup.add(lineMesh);
            }
        });
    }

    dispose() {
        this.scene.remove(this.handsGroup);
        this._jointMaterial.dispose();
        this._boneMaterial.dispose();
        this._jointGeometry.dispose();
    }
}
