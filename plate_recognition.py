#!/usr/bin/env python3
# ============================================================
# Real-time License Plate Recognition
#
# A two-stage deep-learning pipeline built with OpenCV:
#
#   Stage 1  PLATE DETECTOR  (SSD-MobileNet)  finds the plate in the picture
#   Stage 2  CHARACTER OCR   (SSD-MobileNet)  finds every letter/digit on it
#   Then     LAYOUT          sorts the characters into lines and reads
#                            them left-to-right  ->  "30G-49729"
#
# Works on: a single image, a video file, a webcam, or a Jetson Nano
# CSI camera. See README.md for details.
#
# Examples:
#   python plate_recognition.py --source samples/moto_38P1_05694.jpg
#   python plate_recognition.py --source 0                  (webcam)
#   python plate_recognition.py --source video.mp4 --output result.mp4
# ============================================================

import argparse
import csv
import os
import sys
import time
from collections import Counter, deque, namedtuple
from datetime import datetime

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
FONT = cv2.FONT_HERSHEY_SIMPLEX

# One thing the SSD network found: a box (x, y, w, h in pixels), how sure it
# is (0 to 1), and its label ("plate", "A", "7", ...)
Detection = namedtuple("Detection", "x y w h confidence label")

# One fully-read plate
PlateResult = namedtuple("PlateResult", "box confidence text text_confidence")


# ------------------------------------------------------------
# STEP 1: LOAD THE MODELS
# ------------------------------------------------------------

def find_models_dir(user_path=None):
    """Find the folder that contains az_plate/ and az_ocr/ (the trained models)."""
    if user_path:
        candidates = [user_path]
    else:
        candidates = [
            os.path.join(SCRIPT_DIR, "networks"),
            os.path.join(SCRIPT_DIR, "..", "networks"),
            os.path.join(os.getcwd(), "networks"),
        ]
    for folder in candidates:
        plate_ok = os.path.isfile(os.path.join(folder, "az_plate", "az_plate_ssdmobilenetv1.onnx"))
        ocr_ok = os.path.isfile(os.path.join(folder, "az_ocr", "az_ocr_ssdmobilenetv1_2.onnx"))
        if plate_ok and ocr_ok:
            return os.path.abspath(folder)

    sys.exit(
        "ERROR: could not find the trained models.\n"
        "  Put this script in your project folder, next to the 'networks' folder\n"
        "  (the one containing az_plate/ and az_ocr/), or point to it with:\n"
        "      python plate_recognition.py --models path/to/networks"
    )


class SSDDetector:
    """Wraps one SSD-MobileNet model (used for BOTH the plate and the characters).

    The network takes a 300x300 picture and returns 3000 candidate boxes with
    a score for each class. We keep the confident ones and remove duplicates
    with Non-Maximum Suppression (NMS).
    """

    def __init__(self, onnx_path, labels_path, use_cuda=False):
        self.net = cv2.dnn.readNetFromONNX(onnx_path)
        if use_cuda:
            # Only works if OpenCV was built with CUDA (e.g. on a Jetson)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        with open(labels_path) as file:
            self.labels = [line.strip() for line in file if line.strip()]
        self.output_names = list(self.net.getUnconnectedOutLayersNames())

    def detect(self, image, threshold=0.5, nms_threshold=0.45):
        """Return a list of Detection found in the image."""
        height, width = image.shape[:2]

        # Pre-processing: resize to 300x300, BGR -> RGB, scale pixels to about -1..1
        blob = cv2.dnn.blobFromImage(image, 1.0 / 128.0, (300, 300),
                                     (127, 127, 127), swapRB=True)
        self.net.setInput(blob)
        outputs = dict(zip(self.output_names, self.net.forward(self.output_names)))
        scores = outputs["scores"][0]      # shape (3000, number_of_classes)
        boxes = outputs["boxes"][0]        # shape (3000, 4)  x1, y1, x2, y2 in 0..1

        # For each candidate box: which class is it, and how sure are we?
        class_ids = scores[:, 1:].argmax(axis=1) + 1        # skip class 0 = background
        confidences = scores[np.arange(len(scores)), class_ids]
        candidates = np.where(confidences >= threshold)[0]
        if len(candidates) == 0:
            return []

        rects, confs, labels = [], [], []
        for i in candidates:
            x1, y1, x2, y2 = np.clip(boxes[i], 0.0, 1.0)
            w, h = int((x2 - x1) * width), int((y2 - y1) * height)
            if w < 2 or h < 2:
                continue
            rects.append([int(x1 * width), int(y1 * height), w, h])
            confs.append(float(confidences[i]))
            labels.append(self.labels[class_ids[i]])
        if not rects:
            return []

        # Remove overlapping duplicates, keeping the most confident box
        kept = cv2.dnn.NMSBoxes(rects, confs, 0.0, nms_threshold)
        return [Detection(*rects[i], confs[i], labels[i])
                for i in np.array(kept).flatten()]


# ------------------------------------------------------------
# STEP 2: TURN CHARACTER BOXES INTO TEXT
# ------------------------------------------------------------

def group_into_lines(chars, max_lines=2):
    """Sort character detections into lines (top to bottom), each line
    ordered left to right. Vietnamese plates have 1 or 2 lines."""
    if not chars:
        return []

    # Ignore tiny specks that are much smaller than the typical character
    median_h = sorted(c.h for c in chars)[len(chars) // 2]
    chars = [c for c in chars if c.h >= 0.5 * median_h]

    by_height = sorted(chars, key=lambda c: c.y + c.h / 2)
    centers = [c.y + c.h / 2 for c in by_height]

    # A new line starts wherever the vertical gap is bigger than 60% of a character
    gaps = [(centers[i + 1] - centers[i], i) for i in range(len(centers) - 1)]
    splits = sorted(i for gap, i in gaps if gap > 0.6 * median_h)
    if len(splits) > max_lines - 1:      # too many lines: keep only the biggest gaps
        biggest = sorted(gaps, reverse=True)[:max_lines - 1]
        splits = sorted(i for _, i in biggest)

    lines, start = [], 0
    for split in splits:
        lines.append(by_height[start:split + 1])
        start = split + 1
    lines.append(by_height[start:])

    return [sorted(line, key=lambda c: c.x + c.w / 2) for line in lines]


# Letters the network often confuses with digits
LETTER_TO_DIGIT = {"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1",
                   "Z": "2", "S": "5", "G": "6", "B": "8"}


def fix_plate_lines(lines):
    """Correct common mix-ups using the format of Vietnamese plates.

    Rule 1: on a 2-line plate the bottom line is normally all digits
            (e.g. 497.29), so a 'G' there is really a '6'.
    Rule 2: the first two characters are the province code (digits),
            so '3D' really means '30'.
    """
    lines = list(lines)
    if len(lines) == 2:
        bottom = lines[1]
        digit_count = sum(ch.isdigit() for ch in bottom)
        if len(bottom) >= 4 and digit_count > len(bottom) / 2:
            lines[1] = "".join(LETTER_TO_DIGIT.get(ch, ch) for ch in bottom)

    if lines and len(lines[0]) >= 3 and lines[0][0].isdigit() and lines[0][1] in LETTER_TO_DIGIT:
        top = lines[0]
        lines[0] = top[0] + LETTER_TO_DIGIT[top[1]] + top[2:]
    return lines


# ------------------------------------------------------------
# STEP 3: THE FULL PIPELINE
# ------------------------------------------------------------

class PlateRecognizer:
    def __init__(self, models_dir, use_cuda=False, plate_threshold=0.20,
                 char_threshold=0.3, min_chars=5, apply_fixes=True, max_plates=5,
                 try_inverted=True):
        self.plate_detector = SSDDetector(
            os.path.join(models_dir, "az_plate", "az_plate_ssdmobilenetv1.onnx"),
            os.path.join(models_dir, "az_plate", "labels.txt"), use_cuda)
        self.char_detector = SSDDetector(
            os.path.join(models_dir, "az_ocr", "az_ocr_ssdmobilenetv1_2.onnx"),
            os.path.join(models_dir, "az_ocr", "labels.txt"), use_cuda)
        self.plate_threshold = plate_threshold
        self.char_threshold = char_threshold
        self.min_chars = min_chars          # fewer characters than this = not a real plate
        self.apply_fixes = apply_fixes
        self.max_plates = max_plates
        self.try_inverted = try_inverted    # also read light-on-dark plates

    @staticmethod
    def _quality(chars):
        """How good is a reading? Sum of the confidences, so reading MORE
        characters with HIGHER confidence scores better."""
        return sum(c.confidence for c in chars)

    def read_plate(self, plate_image):
        """OCR a cropped plate. Returns (text, confidence) or None.

        The character model was trained mostly on dark letters on a light
        plate. Blue, red and green plates have LIGHT letters on a DARK
        background, so when the first reading is not confident we also try
        an inverted grayscale copy and keep whichever reading is better.
        """
        chars = self.char_detector.detect(plate_image, self.char_threshold, 0.4)

        confident = (len(chars) >= 6 and
                     self._quality(chars) / len(chars) >= 0.85)
        if self.try_inverted and not confident:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            inverted = cv2.cvtColor(255 - gray, cv2.COLOR_GRAY2BGR)
            alternative = self.char_detector.detect(inverted, self.char_threshold, 0.4)
            if self._quality(alternative) > self._quality(chars):
                chars = alternative

        if len(chars) < self.min_chars:
            return None

        lines = group_into_lines(chars)
        lines = ["".join(c.label for c in line) for line in lines]
        if self.apply_fixes:
            lines = fix_plate_lines(lines)
        text = "-".join(lines)              # 2-line plate: "30G-49729"
        confidence = sum(c.confidence for c in chars) / len(chars)
        return text, confidence

    def process(self, frame):
        """Find and read every plate in one frame. Returns a list of PlateResult."""
        frame_h, frame_w = frame.shape[:2]
        plates = self.plate_detector.detect(frame, self.plate_threshold, 0.4)
        plates = sorted(plates, key=lambda p: p.confidence, reverse=True)[:self.max_plates]

        results = []
        for plate in plates:
            x1, y1 = max(0, plate.x), max(0, plate.y)
            x2, y2 = min(frame_w, plate.x + plate.w), min(frame_h, plate.y + plate.h)
            if x2 - x1 < 20 or y2 - y1 < 10:
                continue                                    # too small to read
            read = self.read_plate(frame[y1:y2, x1:x2])     # crop the plate out
            if read is None:
                continue
            text, text_conf = read
            results.append(PlateResult((x1, y1, x2 - x1, y2 - y1),
                                       plate.confidence, text, text_conf))
        return results


# ------------------------------------------------------------
# STEP 4: VIDEO EXTRAS (steady text + logging)
# ------------------------------------------------------------

def iou(a, b):
    """How much do two boxes overlap? 0 = not at all, 1 = identical."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter_w = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    inter_h = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class PlateTracker:
    """Follows each plate from frame to frame and picks its text by voting.

    A single frame can misread a character, but over 15 frames the correct
    reading wins the vote, so the text on screen stops flickering.
    """

    def __init__(self, history=15, max_missing=10, iou_threshold=0.3):
        self.history = history
        self.max_missing = max_missing
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1

    def update(self, results):
        """Match this frame's plates to known tracks.
        Returns a list of (PlateResult, track) pairs."""
        matched = []
        free_tracks = list(self.tracks)
        for result in results:
            best, best_iou = None, self.iou_threshold
            for track in free_tracks:
                overlap = iou(result.box, track["box"])
                if overlap > best_iou:
                    best, best_iou = track, overlap
            if best is None:
                best = {"id": self.next_id, "reads": deque(maxlen=self.history),
                        "missing": 0, "logged": False, "box": result.box}
                self.next_id += 1
                self.tracks.append(best)
            else:
                free_tracks.remove(best)
            best["box"] = result.box
            best["missing"] = 0
            best["reads"].append((result.text, result.text_confidence))
            matched.append((result, best))

        for track in free_tracks:                    # plates not seen this frame
            track["missing"] += 1
        self.tracks = [t for t in self.tracks if t["missing"] <= self.max_missing]
        return matched

    @staticmethod
    def stable_text(track):
        """The most-voted text for this plate (votes weighted by confidence)."""
        votes = Counter()
        for text, conf in track["reads"]:
            votes[text] += conf
        text = votes.most_common(1)[0][0]
        confs = [c for t, c in track["reads"] if t == text]
        return text, sum(confs) / len(confs)


class PlateLogger:
    """Saves each confirmed plate to a CSV file (once per plate)."""

    def __init__(self, path, cooldown_seconds=10):
        self.path = path
        self.cooldown = cooldown_seconds
        self.last_logged = {}
        is_new = not os.path.exists(path) or os.path.getsize(path) == 0
        self.file = open(path, "a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        if is_new:
            self.writer.writerow(["timestamp", "plate", "confidence"])

    def log(self, text, confidence):
        now = time.time()
        if now - self.last_logged.get(text, 0) < self.cooldown:
            return False
        self.last_logged[text] = now
        self.writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                              text, f"{confidence:.2f}"])
        self.file.flush()
        return True

    def close(self):
        self.file.close()


# ------------------------------------------------------------
# STEP 5: DRAWING AND INPUT/OUTPUT
# ------------------------------------------------------------

def draw_plate(frame, box, text, confidence):
    """Draw the plate box and its text on the frame."""
    x, y, w, h = box
    color = (0, 200, 0) if confidence >= 0.7 else (0, 165, 255)     # green / orange
    scale = max(0.5, min(1.5, frame.shape[1] / 900))
    thickness = max(1, int(scale * 2))

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness + 1)
    label = f"{text}  {confidence * 100:.0f}%"
    (tw, th), baseline = cv2.getTextSize(label, FONT, scale, thickness)
    top = max(0, y - th - baseline - 8)
    cv2.rectangle(frame, (x, top), (x + tw + 10, top + th + baseline + 8), color, -1)
    cv2.putText(frame, label, (x + 5, top + th + 3), FONT, scale,
                (0, 0, 0), thickness, cv2.LINE_AA)


def draw_status(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}   (q = quit)", (10, 28), FONT, 0.7,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}   (q = quit)", (10, 28), FONT, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)


def csi_pipeline(width, height, fps=30):
    """GStreamer pipeline for a Jetson Nano CSI camera (e.g. Raspberry Pi cam v2)."""
    return (f"nvarguscamerasrc ! video/x-raw(memory:NVMM), width={width}, "
            f"height={height}, format=NV12, framerate={fps}/1 ! nvvidconv ! "
            f"video/x-raw, width={width}, height={height}, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! appsink drop=1")


def open_video(source, width, height):
    """Open a webcam number, 'csi', a video file, or a network stream."""
    if source == "csi":
        capture = cv2.VideoCapture(csi_pipeline(width, height), cv2.CAP_GSTREAMER)
    elif source.isdigit():
        capture = cv2.VideoCapture(int(source))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    else:
        capture = cv2.VideoCapture(source)
    return capture


def show(window, frame, state):
    """Show a frame. If this computer has no display, switch to console-only."""
    try:
        cv2.imshow(window, frame)
        return True
    except cv2.error:
        if not state.get("warned"):
            print("[!] Cannot open a display window (is opencv-python-headless "
                  "installed?). Continuing without a window.")
            state["warned"] = True
        return False


# ------------------------------------------------------------
# STEP 6: RUN ON AN IMAGE OR A VIDEO
# ------------------------------------------------------------

def run_on_image(recognizer, args):
    image = cv2.imread(args.source)
    if image is None:
        sys.exit(f"ERROR: could not read the image '{args.source}'.")

    start = time.time()
    results = recognizer.process(image)
    elapsed = (time.time() - start) * 1000

    if not results:
        print("No license plate found in this image.")
    for i, r in enumerate(results, 1):
        print(f"Plate {i}: {r.text}   (plate confidence {r.confidence:.2f}, "
              f"text confidence {r.text_confidence:.2f})")
        draw_plate(image, r.box, r.text, r.text_confidence)
    print(f"Processed in {elapsed:.0f} ms")

    if args.output:
        cv2.imwrite(args.output, image)
        print(f"Saved result image: {args.output}")
    if not args.no_display:
        if show("License Plate Recognition (press any key)", image, {}):
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def run_on_video(recognizer, args):
    capture = open_video(args.source, args.width, args.height)
    if not capture.isOpened():
        sys.exit(f"ERROR: could not open the video source '{args.source}'.\n"
                 "  Use a webcam number (0), a video file path, or 'csi'.")

    tracker = PlateTracker()
    logger = PlateLogger(args.log) if args.log else None
    writer, display_state = None, {}
    show_window = not args.no_display
    seen_plates = set()
    frame_number, fps, last_matched = 0, 0.0, []

    print("Running... press 'q' in the video window (or Ctrl+C) to stop.")
    try:
        while True:
            start = time.time()
            ok, frame = capture.read()
            if not ok:
                break                                   # end of the video

            # Detect + read plates. With --skip N this only happens every Nth
            # frame; in between we just redraw the last known plates.
            if frame_number % args.skip == 0:
                results = recognizer.process(frame)
                last_matched = tracker.update(results)
                for result, track in last_matched:
                    text, conf = tracker.stable_text(track)
                    if len(track["reads"]) >= 3 and not track["logged"]:
                        track["logged"] = True          # confirmed after 3 sightings
                        if text not in seen_plates:
                            print(f"[frame {frame_number}] Plate: {text} ({conf:.2f})")
                        seen_plates.add(text)
                        if logger:
                            logger.log(text, conf)

            for result, track in last_matched:
                text, conf = tracker.stable_text(track)
                draw_plate(frame, result.box, text, conf)

            current_fps = 1.0 / max(time.time() - start, 1e-6)
            fps = current_fps if fps == 0 else 0.9 * fps + 0.1 * current_fps
            draw_status(frame, fps)

            if args.output:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                                             capture.get(cv2.CAP_PROP_FPS) or 20, (w, h))
                writer.write(frame)

            if show_window:
                show_window = show("License Plate Recognition", frame, display_state)
                if show_window and cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

            frame_number += 1
            if args.max_frames and frame_number >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        capture.release()
        if writer:
            writer.release()
            print(f"Saved result video: {args.output}")
        if logger:
            logger.close()
            print(f"Plate log saved to: {args.log}")
        cv2.destroyAllWindows()

    print(f"Done. Processed {frame_number} frames. "
          f"Distinct plates read: {sorted(seen_plates) if seen_plates else 'none'}")


# ------------------------------------------------------------
# MAIN PROGRAM
# ------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Real-time license plate recognition (OpenCV + SSD-MobileNet)")
    p.add_argument("--source", default="0",
                   help="image file, video file, webcam number (0), or 'csi' for a Jetson camera")
    p.add_argument("--models", default=None, help="path to the 'networks' folder")
    p.add_argument("--output", default=None, help="save the annotated image/video to this file")
    p.add_argument("--log", default=None, help="save recognised plates to this CSV file (video only)")
    p.add_argument("--no-display", action="store_true", help="don't open a window")
    p.add_argument("--plate-threshold", type=float, default=0.20, help="plate detection confidence (default 0.20)")
    p.add_argument("--char-threshold", type=float, default=0.3, help="character confidence (default 0.3)")
    p.add_argument("--no-fixes", action="store_true", help="turn off the plate-format corrections")
    p.add_argument("--no-invert", action="store_true", help="don't try reading light-on-dark plates")
    p.add_argument("--skip", type=int, default=1, help="process every Nth frame (raise this if it is slow)")
    p.add_argument("--width", type=int, default=640, help="camera width")
    p.add_argument("--height", type=int, default=480, help="camera height")
    p.add_argument("--max-frames", type=int, default=0, help="stop after this many frames (0 = no limit)")
    p.add_argument("--cuda", action="store_true", help="use the GPU (needs OpenCV built with CUDA)")
    args = p.parse_args()
    args.skip = max(1, args.skip)
    return args


def main():
    args = parse_args()
    models_dir = find_models_dir(args.models)
    print(f"Loading models from: {models_dir}")
    recognizer = PlateRecognizer(models_dir, use_cuda=args.cuda,
                                 plate_threshold=args.plate_threshold,
                                 char_threshold=args.char_threshold,
                                 apply_fixes=not args.no_fixes,
                                 try_inverted=not args.no_invert)

    if args.source.lower().endswith(IMAGE_EXTENSIONS):
        run_on_image(recognizer, args)
    else:
        run_on_video(recognizer, args)


if __name__ == "__main__":
    main()
