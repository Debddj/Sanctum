/**
 * Portal shader material — receives centroid/normal uniforms from the server.
 *
 * Uses the custom radial noise portal fragment shader to render the portal
 * VFX, parameterized by the detected hand position and orientation.
 */

import * as THREE from 'three';
import portalVertSrc from './shaders/portal.vert.glsl?raw';
import portalFragSrc from './shaders/portal.frag.glsl?raw';

export class PortalMaterial extends THREE.ShaderMaterial {
    constructor() {
        super({
            vertexShader: portalVertSrc,
            fragmentShader: portalFragSrc,
            uniforms: {
                uTime: { value: 0.0 },
                uPortalCenter: { value: new THREE.Vector2(0.5, 0.5) },
                uPortalRadius: { value: 0.3 },
                uPortalNormal: { value: new THREE.Vector3(0, 0, 1) },
                uIntensity: { value: 1.0 },
                uColor1: { value: new THREE.Color(0xff6600) },
                uColor2: { value: new THREE.Color(0x00ccff) },
            },
            transparent: true,
            side: THREE.DoubleSide,
        });
    }

    update(time, portalState) {
        this.uniforms.uTime.value = time;
        if (portalState) {
            if (portalState.center) {
                this.uniforms.uPortalCenter.value.set(
                    portalState.center[0],
                    portalState.center[1]
                );
            }
            if (portalState.radius !== undefined) {
                this.uniforms.uPortalRadius.value = portalState.radius;
            }
            if (portalState.intensity !== undefined) {
                this.uniforms.uIntensity.value = portalState.intensity;
            }
        }
    }
}
