/**
 * Hand landmark skeleton overlay for Three.js scene.
 *
 * Renders MediaPipe hand landmarks as cyan spheres and green connected bone
 * line segments matching the 21-point hand topology.
 * Dynamically computes exact camera frustum bounds for 1:1 screen alignment.
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
        this._tipMaterial = new THREE.MeshBasicMaterial({ color: 0xff0066 }); // Bright neon pink fingertip highlight
        this._boneMaterial = new THREE.LineBasicMaterial({ color: 0x00ff88, linewidth: 3 });
        this._jointGeometry = new THREE.SphereGeometry(0.12, 12, 12);
        this._tipGeometry = new THREE.SphereGeometry(0.18, 12, 12);

        // Fingertip landmark indices
        this._tipIndices = new Set([4, 8, 12, 16, 20]);
    }

    update(hands, camera) {
        // Clear previous frame mesh objects
        while (this.handsGroup.children.length > 0) {
            const child = this.handsGroup.children.pop();
            if (child.geometry && child.geometry !== this._jointGeometry && child.geometry !== this._tipGeometry) {
                child.geometry.dispose();
            }
        }

        if (!hands || hands.length === 0 || !camera) return;

        // Calculate exact visible camera frustum dimensions at z=0
        const vFovRad = (camera.fov * Math.PI) / 180.0;
        const visibleHeight = 2.0 * Math.tan(vFovRad / 2.0) * camera.position.z;
        const visibleWidth = visibleHeight * camera.aspect;

        hands.forEach((hand) => {
            const landmarks = hand.landmarks;
            if (!landmarks || landmarks.length < 21) return;

            // 1. Draw joint spheres
            const positions = landmarks.map((lm, idx) => {
                const x = lm[0]; // 0-1 MediaPipe normalized
                const y = lm[1];
                const z = lm[2] || 0;

                // Exact 1:1 mapping to camera frustum bounds
                const vec = new THREE.Vector3(
                    (x - 0.5) * visibleWidth,
                    -(y - 0.5) * visibleHeight,
                    -z * 3.0
                );

                const isTip = this._tipIndices.has(idx);
                const geo = isTip ? this._tipGeometry : this._jointGeometry;
                const mat = isTip ? this._tipMaterial : this._jointMaterial;
                const jointMesh = new THREE.Mesh(geo, mat);
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
        this._tipMaterial.dispose();
        this._boneMaterial.dispose();
        this._jointGeometry.dispose();
        this._tipGeometry.dispose();
    }
}
