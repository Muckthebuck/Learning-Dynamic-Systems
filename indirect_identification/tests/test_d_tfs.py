import unittest
import numpy as np
import control as ctrl
from scipy.signal import lfilter
from indirect_identification.d_tfs import d_tfs  # Replace with the actual module path

class TestDTFS(unittest.TestCase):
    def setUp(self):
        """Set up test cases with example transfer functions."""
        self.num1 = [1, 2]
        self.den1 = [1, -0.5]
        self.num2 = [2, 1]
        self.den2 = [1, 0.5]
        
        self.tf1 = d_tfs((self.num1, self.den1))
        self.tf2 = d_tfs((self.num2, self.den2))
        
        self.ctrl_tf1 = ctrl.TransferFunction(self.num1, self.den1, True)
        self.ctrl_tf2 = ctrl.TransferFunction(self.num2, self.den2, True)

    def ensure_correct_scaling(self, tf):
        scale = tf.den[0][0][0]
        tf.den /=scale
        tf.num /=scale

    def test_addition(self):
        """Test addition of two transfer functions."""
        result = self.tf1 + self.tf2
        expected = ctrl.minreal(self.ctrl_tf1 + self.ctrl_tf2)
        expected = self.ctrl_tf1 + self.ctrl_tf2
        self.ensure_correct_scaling(expected)
        np.testing.assert_allclose(result.num, expected.num[0][0], atol=1e-6)
        np.testing.assert_allclose(result.den, expected.den[0][0], atol=1e-6)

    def test_subtraction(self):
        """Test subtraction of two transfer functions."""
        result = self.tf1 - self.tf2
        expected = ctrl.minreal(self.ctrl_tf1 - self.ctrl_tf2)
        expected = self.ctrl_tf1 - self.ctrl_tf2
        self.ensure_correct_scaling(expected)
        np.testing.assert_allclose(result.num, expected.num[0][0], atol=1e-6)
        np.testing.assert_allclose(result.den, expected.den[0][0], atol=1e-6)

    def test_multiplication(self):
        """Test multiplication of two transfer functions."""
        result = self.tf1 * self.tf2
        expected = ctrl.minreal(self.ctrl_tf1 * self.ctrl_tf2)
        expected = self.ctrl_tf1 * self.ctrl_tf2
        self.ensure_correct_scaling(expected)
        np.testing.assert_allclose(result.num, expected.num[0][0], atol=1e-6)
        np.testing.assert_allclose(result.den, expected.den[0][0], atol=1e-6)

    def test_division(self):
        """Test division of two transfer functions."""
        result = self.tf1 / self.tf2
        expected = ctrl.minreal(self.ctrl_tf1 / self.ctrl_tf2)
        expected = self.ctrl_tf1 / self.ctrl_tf2
        self.ensure_correct_scaling(expected)
        np.testing.assert_allclose(result.num, expected.num[0][0], atol=1e-6)
        np.testing.assert_allclose(result.den, expected.den[0][0], atol=1e-6)

    def test_inversion(self):
        """Test inversion of a transfer function."""
        result = ~self.tf1
        expected = 1 / self.ctrl_tf1
        np.testing.assert_allclose(result.num, expected.num[0][0], atol=1e-6)
        np.testing.assert_allclose(result.den, expected.den[0][0], atol=1e-6)

    def test_evaluation(self):
        """Test evaluation of transfer function at a given point."""
        z = 1.5
        result = self.tf1(z)
        expected = self.ctrl_tf1(z)
        self.assertAlmostEqual(result, expected, places=6)

    def test_apply_shift_operator(self):
        """Test applying shift operator on a discrete input signal."""
        U_t = np.array([1, 2, 3, 4, 5])
        result = self.tf1._apply_shift_operator(U_t)
        expected = lfilter(self.num1, self.den1, U_t)
        np.testing.assert_allclose(result, expected, atol=1e-6)

if __name__ == '__main__':
    unittest.main()
