import unittest

import torch

from protein_cnns.model import ProteinCNN


class ModelTest(unittest.TestCase):
    def test_exact_architecture_and_parameter_count(self) -> None:
        model = ProteinCNN()
        self.assertEqual(model.parameter_count, 16_418)
        self.assertEqual(tuple(model.conv1.weight.shape), (32, 21, 7))
        self.assertEqual(tuple(model.conv2.weight.shape), (32, 32, 7))
        self.assertEqual(tuple(model.classifier.weight.shape), (2, 2240))
        self.assertEqual(tuple(model(torch.zeros(3, 21, 70)).shape), (3, 2))


if __name__ == "__main__":
    unittest.main()
