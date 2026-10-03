# License Plate Recognition (OpenCV + SSD-MobileNet)

A two-stage deep-learning pipeline that finds vehicle license plates in images, video files or a live camera, and reads the characters on them.

```
camera/video frame
      |
      v
[1] PLATE DETECTOR  (SSD-MobileNet, 300x300)  ->  where is the plate?
      |  crop
      v
[2] CHARACTER OCR   (SSD-MobileNet, 300x300)  ->  which letters/digits, and where?
      |
      v
[3] LAYOUT + FIXES  sort into lines, read left-to-right, fix 0/D, G/6 mix-ups
      |
      v
[4] TRACKER         vote over 15 frames so the text doesn't flicker  ->  "30G-49729"
```

Both networks come from the open-source project *Real-time Auto License Plate Recognition with Jetson Nano* (trained on Vietnamese plates). The original project runs them one at a time with NVIDIA's `detectnet-camera.py`. This code **chains them together** into a complete reader, runs them with plain OpenCV (so it works on any computer, not only a Jetson), and adds a tracker, logging and tests.

## Setup

1. Install Python 3.9 or newer (python.org). Tick "Add Python to PATH".
2. Install the libraries:
   ```
   pip install -r requirements.txt
   ```
3. Put these files **in your project folder, next to the `networks` folder** (the one containing `az_plate/` and `az_ocr/`):
   ```
   your-project/
   |-- networks/                 (the trained models - already in your project)
   |-- plate_recognition.py
   |-- test_pipeline.py
   |-- samples/
   `-- requirements.txt
   ```
   Or keep them anywhere and add `--models path/to/networks` to every command.

## Run

```bash
# One image (a window opens; press any key to close)
python plate_recognition.py --source samples/moto_38P1_05694.jpg

# Save the annotated result instead of/as well as showing it
python plate_recognition.py --source samples/new_30G_49729.jpg --output result.jpg

# Webcam (press q to quit)
python plate_recognition.py --source 0

# A video file, saving the annotated video and a CSV log of plates
python plate_recognition.py --source traffic.mp4 --output result.mp4 --log plates.csv

# Jetson Nano CSI camera
python plate_recognition.py --source csi
```

| Option | Meaning |
|---|---|
| `--source` | image path, video path, webcam number (`0`), or `csi` |
| `--models` | path to the `networks` folder |
| `--output` | save the annotated image/video |
| `--log` | save recognised plates to a CSV file (video/camera) |
| `--skip N` | process only every Nth frame - **raise this if it is slow** |
| `--plate-threshold` | plate detection confidence, default 0.20 |
| `--char-threshold` | character confidence, default 0.30 |
| `--no-invert` | faster, but weaker on blue/red/green plates |
| `--no-fixes` | turn off the plate-format corrections |
| `--no-display` | no window (for servers) |
| `--cuda` | use the GPU (only if OpenCV was built with CUDA) |

Run the automated tests with `python test_pipeline.py` (15 tests; the 5 that run the neural networks are skipped automatically if `networks/` is not found).

## How it works

- **Detection** - the SSD network outputs 3000 candidate boxes; boxes below the confidence threshold are dropped, and Non-Maximum Suppression removes duplicates.
- **Reading** - the same kind of network, trained on characters (0-9, A-Z), is run on the cropped plate. Characters are grouped into one or two lines by their vertical position, then sorted left to right.
- **Light-on-dark plates** - the character model was trained mostly on dark letters on light plates. Blue, red and green plates are the opposite, so when the first reading is not confident, an inverted grayscale copy is also read and the better reading (higher total confidence) is kept.
- **Vietnamese format rules** - the bottom line of a 2-line plate is normally all digits (so `G` there is really `6`), and the first two characters are the province code (so `3D` means `30`). Plates that legitimately contain letters (military, diplomatic) are not changed.
- **Tracker** - plates are followed between frames and the most-voted text wins. Each confirmed plate is written once to the CSV log.
- **False alarms** - a detection is only accepted if at least 5 characters can be read on it.

## Measured results

Tested on the 6 sample photos (real photos from the project's dataset - the models may have seen similar images during training, so treat these as optimistic):

| Test | Result |
|---|---|
| Plates located in 6 photos | 6 / 6 |
| Plates read exactly (whole photos) | 3 / 6, character accuracy 83% |
| Close-up plate crops | 5 / 6 exact, character accuracy 96% |
| Street photos with the plate painted over | 0 false plates (6 photos) |

Exact reads: `30G-49729`, `29E-01566`, `38P1-05694`. Partly right: the blue plate `80B-6789` is read as `B6789`, the military plate `BH-52-50` as `8H5250`. The diplomatic plate (a red line is printed across its digits) is read poorly.

**Speed** (single-core cloud CPU, 600x340 video; a normal laptop with several cores is faster - check the FPS shown on screen):

| Setting | FPS |
|---|---|
| default | about 5 |
| `--no-invert` | about 7 |
| `--skip 2` | about 9 |
| `--skip 3` | about 13 (but fast-moving plates can be missed) |

## Limitations

- Trained on Vietnamese plates; other countries' layouts/fonts will read worse.
- Small or far-away plates (under about 100 pixels wide), motion blur and strong glare reduce accuracy.
- Similar-looking characters (`0/D`, `8/B`, `G/6`, `N/M`) are the main source of errors.
- Only reads plates it can detect first; very small plates may be missed by the detector.
- The tests use 6 photos, so the accuracy figures are indicative, not a rigorous benchmark.

## Jetson Nano

The code is written to be Jetson-compatible (`--source csi` builds the standard GStreamer pipeline for a CSI camera such as the Raspberry Pi Camera v2), but **it has not been tested on Jetson hardware**. Notes:

- JetPack already includes OpenCV with GStreamer support; do not `pip install opencv-python` there.
- The Nano's CPU is slow. Use `--skip 3 --no-invert` and a 640x480 camera.
- `--cuda` only helps if your OpenCV build has CUDA DNN support.
- For full GPU speed, use NVIDIA's `jetson-inference` with the commands in the original project's `doc/plate-detect.md` and `doc/plate-ocr.md`. That path runs one model at a time.

## Credits

- Pretrained models, dataset and sample images: *Real-time Auto License Plate Recognition with Jetson Nano* project (see its `doc/` folder).
- Libraries: OpenCV, NumPy.
