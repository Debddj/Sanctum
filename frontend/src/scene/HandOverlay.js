/**
 * Debug hand landmark skeleton overlay.
 *
 * Renders MediaPipe hand landmarks as connected line segments in the
 * Three.js scene for visual debugging during development.
 */

// TODO: Implement in Phase 1
// - Create BufferGeometry for 21 landmark points per hand
// - Draw connections matching MediaPipe hand topology
// - Update positions from server WebSocket landmark data
// - Toggle visibility via debug overlay

export class HandOverlay {
    constructor(scene) {
        this.scene = scene;
    }

    update(hands) {
        // TODO: Update landmark geometry from server data
    }

    dispose() {
        // TODO: Remove meshes from scene
    }
}
