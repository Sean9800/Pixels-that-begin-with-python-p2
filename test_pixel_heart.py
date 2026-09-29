import importlib.util
import pathlib
import unittest

import numpy as np

MODULE_PATH = pathlib.Path(__file__).with_name("pixel heart.py")
spec = importlib.util.spec_from_file_location("pixel_heart", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PixelHeartTests(unittest.TestCase):
    def test_ease_range(self):
        self.assertEqual(module.ease(0.0), 0.0)
        self.assertEqual(module.ease(1.0), 1.0)
        self.assertAlmostEqual(module.ease(0.5), 0.5)

    def test_heart_returns_numeric_values(self):
        x, y = module.heart(np.pi / 2)
        self.assertTrue(np.isfinite(x))
        self.assertTrue(np.isfinite(y))

    def test_build_heart_pixels_outputs_arrays(self):
        xs, ys, colors = module.build_heart_pixels(0.0, 48, 48)
        self.assertEqual(len(xs), len(ys))
        self.assertEqual(len(xs), len(colors))
        self.assertGreater(len(xs), 0)

    def test_build_heart_pixels_has_a_top_notch(self):
        xs, ys, _ = module.build_heart_pixels(0.0, 200, 200)
        top_y = ys.max()
        top_x = xs[np.isclose(ys, top_y)]
        self.assertLess(top_x.max() - top_x.min(), 18.0)


if __name__ == "__main__":
    unittest.main()
