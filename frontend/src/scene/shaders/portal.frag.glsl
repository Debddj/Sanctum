// Portal fragment shader — radial noise portal texture
//
// Creates a swirling, glowing portal effect parameterized by center position,
// radius, and time. Uses layered simplex noise for organic turbulence.

precision highp float;

uniform float uTime;
uniform vec2 uPortalCenter;
uniform float uPortalRadius;
uniform vec3 uPortalNormal;
uniform float uIntensity;
uniform vec3 uColor1;
uniform vec3 uColor2;

varying vec2 vUv;

// Simple 2D hash for noise
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

// 2D noise
float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));

    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Fractal Brownian Motion
float fbm(vec2 p) {
    float value = 0.0;
    float amplitude = 0.5;
    for (int i = 0; i < 5; i++) {
        value += amplitude * noise(p);
        p *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

void main() {
    vec2 uv = vUv - uPortalCenter;
    float dist = length(uv);
    float angle = atan(uv.y, uv.x);

    // Swirling noise
    float swirl = angle + uTime * 0.5 + fbm(uv * 4.0 + uTime * 0.3) * 3.0;
    float noiseVal = fbm(vec2(swirl, dist * 5.0 - uTime * 0.2));

    // Radial falloff
    float ring = smoothstep(uPortalRadius + 0.05, uPortalRadius - 0.05, dist);
    float innerGlow = smoothstep(uPortalRadius * 0.3, 0.0, dist);

    // Color mixing
    vec3 color = mix(uColor1, uColor2, noiseVal);
    color += vec3(innerGlow * 0.5);

    // Final alpha
    float alpha = ring * noiseVal * uIntensity;
    alpha = clamp(alpha, 0.0, 1.0);

    gl_FragColor = vec4(color, alpha);
}
