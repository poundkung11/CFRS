# Classroom Facial Recognition System

A real-time facial recognition service designed for classroom attendance tracking. Built with dynamic thresholding, pose-aware matching, and a centroid-based tracker that prevents duplicate check-ins.

---

## Features

- **Dynamic Threshold** — recognition confidence threshold scales smoothly with face size (near vs. far from camera)
- **Pose Penalty** — reduces threshold when a face is turned sideways, reducing false positives
- **Weighted Distance Matching** — uses a weighted average of the two closest database encodings instead of raw minimum, guarding against outlier images
- **Margin Check** — requires a clear gap between the best and second-best match before confirming identity
- **Dynamic Blur Detection** — Laplacian variance test with a threshold that adapts to face width
- **Centroid Tracker + Confirmation Timer** — a face must be consistently present for 5 seconds before being marked _Verified_
- **`confirmed_names_db` set** — each person is logged to the database only once per session, even if they re-enter the frame
- **Thread-safe tracker** — `threading.Lock` guards shared state, ready for multi-threaded extension
- **Batch Landmark Extraction** — all face landmarks are computed once per frame, not per face

---

## Key_Option

Prioritize all tasks thoroughly and review available processes optimizing logistics

---

## Project Structure

```
classroom-facial-recognition/
│
├── main.py                  # Entry point — capture loop + tracker logic
│
├── known_faces/             # Reference images for enrolled students
│   ├── alice_01.jpg         # Filename format: NAME_anything.jpg
│   ├── alice_02.jpg         # Multiple photos per person are supported
│   └── bob_01.png
│
├── logs/                    # (Optional) Attendance or event logs
├── data/                    # (Optional) Cached encodings or exports
│
├── requirements.txt
├── .gitattributes
└── README.md
```

> **Naming convention for `known_faces/`:** the part before the first `_` becomes the identity label (uppercased). So `alice_01.jpg` → `ALICE`, `john_smith_02.png` → `JOHN`.

---

## Requirements

- Python 3.8+
- A webcam (OpenCV-compatible)
- CMake + a C++ compiler (required by `dlib`, which backs `face_recognition`)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Minimal `requirements.txt`:

```
face_recognition
opencv-python
numpy
```

### macOS (Apple Silicon)

```bash
brew install cmake
pip install face_recognition opencv-python numpy
```

### Ubuntu / Debian

```bash
sudo apt-get install build-essential cmake libopenblas-dev liblapack-dev
pip install face_recognition opencv-python numpy
```

---

## Quick Start

1. **Add reference photos** to `known_faces/` using the naming convention above. More photos per person = better accuracy.

2. **Run the system:**

```bash
python main.py
```

3. **On-screen colours:**

| Colour | Meaning                                             |
| ------ | --------------------------------------------------- |
| Yellow | Face detected, verifying identity (countdown shown) |
| Green  | Identity confirmed (_Verified_)                     |
| Red    | Unknown face                                        |
| Orange | Moving or blurry — skipped                          |

4. **Press `q`** to quit.

---

## Configuration

All tunable parameters live at the top of `ClassroomFacialRecognitionService.__init__` and in the `__main__` block:

| Parameter           | Default  | Description                                          |
| ------------------- | -------- | ---------------------------------------------------- |
| `CONFIRMATION_TIME` | `5.0` s  | Seconds a face must be stable before confirmation    |
| `TIMEOUT`           | `10.0` s | Seconds before a lost track is discarded             |
| `MAX_DISTANCE`      | `50` px  | Max centroid movement to re-link a track             |
| `MIN_CONFIDENCE`    | `60.0` % | Minimum sigmoid confidence to accept a match         |
| `WEIGHT_BEST`       | `0.7`    | Weight for the closest encoding in weighted distance |
| `WEIGHT_SECOND`     | `0.3`    | Weight for the second-closest encoding               |
| `CONFIDENCE_K`      | `12`     | Steepness of the sigmoid confidence curve            |

---

## Architecture

```
Camera Frame
     │
     ▼
process_frame()
  ├─ Resize + HOG face detection
  ├─ Batch: face encodings (num_jitters=2)
  ├─ Batch: face landmarks → pose penalty
  ├─ Dynamic blur check (per face)
  ├─ Dynamic threshold + margin (per face)
  └─ Weighted distance → identity decision
           │
           ▼
     Centroid Tracker  (main loop)
  ├─ Match detection → existing track (Euclidean distance)
  ├─ Create new track if unmatched
  ├─ Confirmation timer (5 s)
  └─ confirmed_names_db → one-time DB log per identity
```

---

## Known Limitations

- **HOG model** is used for speed; swap to `model="cnn"` in `face_locations()` for better accuracy on GPU.
- **Re-entry across sessions** — `confirmed_names_db` is in-memory only; it resets when the script restarts.
- **Centroid tracker** is position-only; fast lateral movement can break track continuity.
- **`resize_scale=1`** (default) processes full-resolution frames. Pass `0.5` for a significant FPS boost on high-resolution cameras.

---
