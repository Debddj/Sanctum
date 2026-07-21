"""Data collection CLI — record labeled gesture samples from webcam or synthetic generator.

Usage:
    python scripts/record_gesture_samples.py --gesture sling_ring --count 50 --output data/landmark_sequences/
    python scripts/record_gesture_samples.py --synthetic --count 40 --output data/landmark_sequences/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.landmarks.normalization import normalize_landmarks

GESTURE_CLASSES = {
    "sling_ring": 0,
    "mudra_hold": 1,
    "pinch_pull": 2,
    "open_palm": 3,
}


def generate_synthetic_sequence(gesture_name: str, window_size: int = 30) -> np.ndarray:
    """Generate a synthetic 30-frame landmark sequence for testing without camera."""
    t = np.linspace(0, 2 * np.pi, window_size)
    sequence = np.zeros((window_size, 21, 3), dtype=np.float32)

    # Base hand layout
    base_lms = np.random.normal(0, 0.1, (21, 3)).astype(np.float32)
    base_lms[0] = [0, 0, 0]  # Wrist at origin

    for i in range(window_size):
        frame_lms = base_lms.copy()

        if gesture_name == "sling_ring":
            radius = 0.5
            frame_lms[8] = [radius * np.cos(t[i]), radius * np.sin(t[i]), 0.1 * np.sin(t[i])]
        elif gesture_name == "pinch_pull":
            z_offset = -0.05 * i
            frame_lms[4] = [0.02, 0.02, z_offset]
            frame_lms[8] = [0.02, 0.02, z_offset]
        elif gesture_name == "mudra_hold":
            frame_lms += np.random.normal(0, 0.01, (21, 3)).astype(np.float32)
        elif gesture_name == "open_palm":
            frame_lms += np.random.normal(0, 0.005, (21, 3)).astype(np.float32)

        sequence[i] = normalize_landmarks(frame_lms, method="wrist_relative")

    return sequence.reshape(window_size, 63)


def record_samples(
    gesture_name: str,
    target_count: int,
    output_dir: Path,
    use_synthetic: bool = False,
) -> list[dict]:
    """Record or generate landmark sequences and return manifest entries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    label = GESTURE_CLASSES[gesture_name]
    manifest_entries = []

    print(f"[Record] Generating {target_count} samples for gesture: {gesture_name} (label={label})")

    if use_synthetic:
        for idx in range(target_count):
            seq = generate_synthetic_sequence(gesture_name)
            filename = f"{gesture_name}_{idx:04d}.npy"
            filepath = output_dir / filename
            np.save(filepath, seq)
            manifest_entries.append({"file": filename, "label": label, "gesture": gesture_name})
        return manifest_entries

    # Live camera recording mode
    try:
        from src.capture.webcam_stream import WebcamStream
        from src.landmarks.mediapipe_hands import MediaPipeHands

        stream = WebcamStream().start()
        hands_extractor = MediaPipeHands()
    except Exception as e:
        print(f"[Record] Camera / MediaPipe access failed: {e}. Falling back to synthetic generation.")
        return record_samples(gesture_name, target_count, output_dir, use_synthetic=True)

    recorded = 0
    buffer: list[np.ndarray] = []

    print("[Record] Capturing 30-frame sequence...")

    while recorded < target_count:
        frame = stream.read()
        if frame is None:
            continue

        frame_rgb = frame[:, :, ::-1]
        hands = hands_extractor.extract(frame_rgb)

        if hands:
            norm_lms = normalize_landmarks(hands[0].landmarks)
            buffer.append(norm_lms.flatten())

            if len(buffer) == 30:
                seq = np.array(buffer, dtype=np.float32)
                filename = f"{gesture_name}_{recorded:04d}.npy"
                np.save(output_dir / filename, seq)
                manifest_entries.append({"file": filename, "label": label, "gesture": gesture_name})

                recorded += 1
                buffer.clear()
                print(f"Recorded sample {recorded}/{target_count}")

    stream.stop()
    hands_extractor.close()
    return manifest_entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Record labeled gesture samples.")
    parser.add_argument(
        "--gesture",
        choices=list(GESTURE_CLASSES.keys()),
        default="sling_ring",
        help="Gesture class name",
    )
    parser.add_argument("--count", type=int, default=50, help="Number of sequence samples to record")
    parser.add_argument("--output", type=str, default="data/landmark_sequences", help="Output directory")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic samples without camera")
    parser.add_argument("--all-classes", action="store_true", help="Record samples for all gesture classes")

    args = parser.parse_args()
    output_path = PROJECT_ROOT / args.output
    splits_path = PROJECT_ROOT / "data" / "splits"
    splits_path.mkdir(parents=True, exist_ok=True)

    all_entries = []
    classes_to_record = list(GESTURE_CLASSES.keys()) if args.all_classes else [args.gesture]

    for g_name in classes_to_record:
        entries = record_samples(
            gesture_name=g_name,
            target_count=args.count,
            output_dir=output_path,
            use_synthetic=args.synthetic,
        )
        all_entries.extend(entries)

    # Train / val / test split (70 / 15 / 15)
    np.random.shuffle(all_entries)
    n_total = len(all_entries)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)

    train_entries = all_entries[:n_train]
    val_entries = all_entries[n_train : n_train + n_val]
    test_entries = all_entries[n_train + n_val :]

    with open(splits_path / "train_manifest.json", "w") as f:
        json.dump(train_entries, f, indent=2)

    with open(splits_path / "val_manifest.json", "w") as f:
        json.dump(val_entries, f, indent=2)

    with open(splits_path / "test_manifest.json", "w") as f:
        json.dump(test_entries, f, indent=2)

    print(
        f"[Record] Done! Created dataset with {n_total} samples: "
        f"train={len(train_entries)}, val={len(val_entries)}, test={len(test_entries)}"
    )


if __name__ == "__main__":
    main()
