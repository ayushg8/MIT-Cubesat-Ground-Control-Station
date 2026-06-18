"""Focused regression tests for receiver validation boundaries."""

import hashlib
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from receiver.packet_handler import validate_filename, validate_transfer


class PacketHandlerTest(unittest.TestCase):
    def test_valid_transfer(self):
        payload = b"lunar image bytes"
        result = validate_transfer(
            payload,
            len(payload),
            hashlib.md5(payload).hexdigest(),
            {"pass_number": 1},
        )
        self.assertTrue(result.valid)

    def test_rejects_path_traversal_and_absolute_paths(self):
        for filename in ("../escape.jpg", "images/frame.jpg", "/tmp/frame.jpg"):
            with self.subTest(filename=filename):
                self.assertFalse(validate_filename(filename).valid)

    def test_accepts_plain_filename(self):
        self.assertTrue(validate_filename("pass1_img00.jpg").valid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
