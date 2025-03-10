import unittest
import numpy as np
from cupyx.profiler import benchmark
from indirect_identification.tf_methods.deprecated.tfs_methods import _add_tfs, _sub_tfs, _mul_tfs 
from indirect_identification.tf_methods.deprecated.fast_tfs_methods import _add_tfs as _add_tfs_njit, _sub_tfs as _sub_tfs_njit, _mul_tfs as _mul_tfs_njit  
from indirect_identification.tf_methods.fast_tfs_methods_fast_math import _add_tfs as _add_tfs_fastmath, _sub_tfs as _sub_tfs_fastmath, _mul_tfs as _mul_tfs_fastmath  

class TestFunctions(unittest.TestCase):
    
    def setUp(self):
        # Setup some test data for comparison
        self.num1 = np.array([1.0, 2.0, 3.0])
        self.den1 = np.array([1.0, 1.0, 1.0])
        self.num2 = np.array([2.0, 1.0, 0.0])
        self.den2 = np.array([1.0, 2.0, 3.0])

        # Define the functions for each operation
        self.add_functions = [
            (_add_tfs, '_add_tfs'),
            (_add_tfs_njit, '_add_tfs_njit'),
            (_add_tfs_fastmath, '_add_tfs_fastmath')
        ]
        
        self.sub_functions = [
            (_sub_tfs, '_sub_tfs'),
            (_sub_tfs_njit, '_sub_tfs_njit'),
            (_sub_tfs_fastmath, '_sub_tfs_fastmath')
        ]
        
        self.mul_functions = [
            (_mul_tfs, '_mul_tfs'),
            (_mul_tfs_njit, '_mul_tfs_njit'),
            (_mul_tfs_fastmath, '_mul_tfs_fastmath')
        ]

    def _test_operation(self, operation_functions, operation_name):
        """
        Helper function to compare results and benchmark functions for a specific operation.
        
        Parameters:
        - operation_functions: list of tuples containing (function, name)
        - operation_name: name of the operation (for benchmarking and reporting)
        """
        # Compare results for all functions in the operation group
        for func, name in operation_functions:
            for compare_func, compare_name in operation_functions:
                if func != compare_func:
                    with self.subTest(func=func, compare_func=compare_func):
                        print(f"Testing {name} vs {compare_name}")
                        num_1, den_1 = func(self.num1, self.den1, self.num2, self.den2)
                        num_2, den_2 = compare_func(self.num1, self.den1, self.num2, self.den2)
                        np.testing.assert_allclose(num_1, num_2, atol=1e-10)
                        np.testing.assert_allclose(den_1, den_2, atol=1e-10)

        # Benchmarking the operation functions
        for func, name in operation_functions:
            def benchmark_func():
                return func(self.num1, self.den1, self.num2, self.den2)

            print(f"Benchmarking {name}:")
            print(benchmark(benchmark_func))

    def test_add_operations(self):
        self._test_operation(self.add_functions, "addition")

    def test_sub_operations(self):
        self._test_operation(self.sub_functions, "subtraction")

    def test_mul_operations(self):
        self._test_operation(self.mul_functions, "multiplication")

if __name__ == "__main__":
    unittest.main()
