// Time-reversal fragment shader — optical flow distortion / reality unraveling pixel warp
precision highp float;

uniform sampler2D uSceneTexture;
uniform sampler2D uDistortionMap;
uniform float uProgress;

varying vec2 vUv;

void main() {
    // Sample optical flow vectors from distortion map
    vec2 flow = texture2D(uDistortionMap, vUv).rg - 0.5;
    
    // Apply time-reversal pixel displacement
    vec2 offset = flow * uProgress * 0.05;
    vec4 sceneColor = texture2D(uSceneTexture, vUv + offset);
    
    // Chromatic aberration / temporal distortion tint
    vec4 redShift = texture2D(uSceneTexture, vUv + offset * 1.2);
    vec4 blueShift = texture2D(uSceneTexture, vUv + offset * 0.8);
    
    gl_FragColor = vec4(redShift.r, sceneColor.g, blueShift.b, sceneColor.a);
}
