# Gesture Specification

Canonical definitions for all recognized gestures in Sanctum.

## Gesture Classes

### 1. Sling Ring (`sling_ring`)
- **Type**: Temporal (sequence-dependent)
- **Motion**: Circular hand motion tracing a portal opening, similar to the Doctor Strange sling ring gesture
- **Detection**: Requires ~30 frames of continuous circular motion in the landmark trajectory
- **Dispatch Action**: `spawn_portal` — creates a portal VFX at the traced circle's centroid with orientation from hand normal vector
- **Cooldown**: 1000ms

### 2. Mudra Hold (`mudra_hold`)
- **Type**: Static (single-frame)
- **Pose**: Specific finger configuration (e.g., thumb + index touching, remaining fingers extended)
- **Detection**: Static pose held for sufficient frames above confidence threshold
- **Dispatch Action**: `activate_shield`
- **Cooldown**: 500ms

### 3. Pinch Pull (`pinch_pull`)
- **Type**: Temporal
- **Motion**: Thumb + index finger pinch followed by a pulling motion away from the camera
- **Detection**: Pinch detection followed by increasing Z-depth in landmarks over ~20 frames
- **Dispatch Action**: `trigger_time_reversal` — activates ring buffer reverse playback
- **Cooldown**: 800ms

### 4. Open Palm (`open_palm`)
- **Type**: Static
- **Pose**: Fully open hand with fingers spread
- **Detection**: All fingers extended, sufficient inter-finger distance
- **Dispatch Action**: `dismiss_effect` — clears all active VFX
- **Cooldown**: 300ms

## Landmark Reference

MediaPipe hand landmarks (21 points):
```
 0: WRIST
 1-4: THUMB (CMC, MCP, IP, TIP)
 5-8: INDEX (MCP, PIP, DIP, TIP)
 9-12: MIDDLE (MCP, PIP, DIP, TIP)
13-16: RING (MCP, PIP, DIP, TIP)
17-20: PINKY (MCP, PIP, DIP, TIP)
```

## Classifier Notes

- The sling ring gesture is the key temporal gesture — it's where LSTM sequence memory should visibly beat a static-pose baseline
- All gestures should be evaluated with per-class confusion matrices before optimization
- Latency per inference must be measured for the before/after TensorRT comparison
