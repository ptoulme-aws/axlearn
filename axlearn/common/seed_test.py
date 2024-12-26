import os
from absl.testing import absltest
from axlearn.common import test_utils
import tensorflow as tf
import importlib


class SeedTest(test_utils.TestCase):
    def test_tf_random_seed_from_env(self):
        os.environ["DATA_SEED"] = "42"
        importlib.import_module('axlearn.common.input_lm')
        sequence_1 = [tf.random.uniform((1,)).numpy() for _ in range(5)]

        # Re-import 'input_lm' to reset seed
        importlib.reload(importlib.import_module('axlearn.common.input_lm'))
        sequence_2 = [tf.random.uniform((1,)).numpy() for _ in range(5)]

        # Assert that the two sequences are same
        self.assertEqual(sequence_1, sequence_2)

if __name__ == "__main__":
    absltest.main()