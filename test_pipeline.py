# ============================================================
# Automated tests for the License Plate Recognition project.
# Run with:   python test_pipeline.py
#
#  - "Unit tests" check the small helper functions (no models needed).
#  - "Integration tests" run the real neural networks on the images in
#    samples/. They are skipped if the 'networks' folder is not found.
# ============================================================

import csv
import os
import tempfile
import unittest

import cv2

import plate_recognition as pr

HERE = os.path.dirname(os.path.abspath(__file__))

# What is really written on each sample photo
EXPECTED = {
    "new_30G_49729.jpg": "30G49729",
    "taxi_29E_01566.jpg": "29E01566",
    "moto_38P1_05694.jpg": "38P105694",
    "state_80B_6789.jpg": "80B6789",
    "military_BH5250.jpg": "BH5250",
    "diplomatic_NG01.jpg": "41291NG01",
}


def chars(spec):
    """Helper: build fake character detections from (label, x, y) tuples."""
    return [pr.Detection(x, y, 20, 40, 0.9, label) for label, x, y in spec]


class UnitTests(unittest.TestCase):
    def test_two_line_plate_is_grouped_and_ordered(self):
        # Characters given in scrambled order: "30G" on top, "49" below
        found = chars([("9", 130, 100), ("G", 60, 10), ("3", 0, 10),
                       ("4", 100, 100), ("0", 30, 10)])
        lines = pr.group_into_lines(found)
        self.assertEqual(["".join(c.label for c in line) for line in lines], ["30G", "49"])

    def test_single_line_plate_stays_one_line(self):
        found = chars([("8", 0, 10), ("0", 30, 12), ("B", 60, 9), ("6", 100, 11)])
        self.assertEqual(len(pr.group_into_lines(found)), 1)

    def test_no_characters(self):
        self.assertEqual(pr.group_into_lines([]), [])

    def test_fix_bottom_line_letters_become_digits(self):
        self.assertEqual(pr.fix_plate_lines(["30G", "4G7S9"]), ["30G", "46759"])

    def test_fix_province_code(self):
        self.assertEqual(pr.fix_plate_lines(["3DG", "49729"]), ["30G", "49729"])

    def test_letters_plates_are_not_damaged(self):
        # Military / diplomatic plates legitimately start with or contain letters
        self.assertEqual(pr.fix_plate_lines(["BH5250"]), ["BH5250"])
        self.assertEqual(pr.fix_plate_lines(["41291", "NG01"]), ["41291", "NG01"])

    def test_iou(self):
        self.assertEqual(pr.iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(pr.iou((0, 0, 10, 10), (50, 50, 10, 10)), 0.0)
        self.assertAlmostEqual(pr.iou((0, 0, 10, 10), (5, 0, 10, 10)), 1 / 3, places=3)

    def test_tracker_votes_out_a_misread(self):
        tracker = pr.PlateTracker()
        box = (100, 100, 80, 40)
        reads = ["30G-49729", "30G-49729", "3DG-49729", "30G-49729", "30G-49729"]
        for text in reads:
            matched = tracker.update([pr.PlateResult(box, 0.9, text, 0.8)])
        text, _ = tracker.stable_text(matched[0][1])
        self.assertEqual(text, "30G-49729")
        self.assertEqual(len(tracker.tracks), 1)       # same plate -> one track

    def test_tracker_separates_two_plates(self):
        tracker = pr.PlateTracker()
        tracker.update([pr.PlateResult((0, 0, 50, 30), 0.9, "AAA", 0.8),
                        pr.PlateResult((300, 200, 50, 30), 0.9, "BBB", 0.8)])
        self.assertEqual(len(tracker.tracks), 2)

    def test_logger_writes_once_per_plate(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "log.csv")
            logger = pr.PlateLogger(path, cooldown_seconds=60)
            self.assertTrue(logger.log("30G-49729", 0.9))
            self.assertFalse(logger.log("30G-49729", 0.9))   # repeat within cooldown
            self.assertTrue(logger.log("29E-01566", 0.8))
            logger.close()
            with open(path) as file:
                rows = list(csv.reader(file))
            self.assertEqual(len(rows), 3)                   # header + 2 plates


def load_recognizer():
    try:
        return pr.PlateRecognizer(pr.find_models_dir())
    except SystemExit:
        return None


@unittest.skipIf(load_recognizer() is None, "trained models ('networks' folder) not found")
class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recognizer = load_recognizer()

    def read(self, filename):
        image = cv2.imread(os.path.join(HERE, "samples", filename))
        self.assertIsNotNone(image, f"missing sample {filename}")
        return self.recognizer.process(image)

    def test_every_sample_plate_is_located(self):
        for filename in EXPECTED:
            with self.subTest(filename):
                self.assertEqual(len(self.read(filename)), 1)

    def test_clear_plates_are_read_exactly(self):
        for filename in ("new_30G_49729.jpg", "taxi_29E_01566.jpg", "moto_38P1_05694.jpg"):
            with self.subTest(filename):
                results = self.read(filename)
                self.assertEqual(results[0].text.replace("-", ""), EXPECTED[filename])

    def test_hard_plates_are_mostly_right(self):
        # Light-on-dark / tiny plates are harder: require most characters correct
        for filename in ("state_80B_6789.jpg", "military_BH5250.jpg"):
            with self.subTest(filename):
                text = self.read(filename)[0].text.replace("-", "")
                want = EXPECTED[filename]
                matches = sum(a == b for a, b in zip(reversed(text), reversed(want)))
                self.assertGreaterEqual(matches / len(want), 0.6)

    def test_two_line_plate_has_a_dash(self):
        self.assertEqual(self.read("new_30G_49729.jpg")[0].text, "30G-49729")

    def test_background_is_not_a_plate(self):
        # Paint over the plates in a photo: nothing should be reported
        image = cv2.imread(os.path.join(HERE, "samples", "taxi_29E_01566.jpg"))
        for p in self.recognizer.plate_detector.detect(image, 0.1, 0.4):
            cv2.rectangle(image, (p.x, p.y), (p.x + p.w, p.y + p.h), (90, 90, 90), -1)
        self.assertEqual(self.recognizer.process(image), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
