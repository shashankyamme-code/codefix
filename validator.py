"""
Validator for KV-Cached Multi-Head Attention - AI CODEFIX 2025 HARD_2

This script validates your implementation against test cases.

Usage:
    python validator.py --file kv_attention.py
    python validator.py --file kv_attention.py --verbose
    python validator.py --file kv_attention.py --test-file test_cases_hidden.json
"""

import argparse
import json
import sys
import importlib.util
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


class AttentionValidator:
    """Validator for KV-Cached Multi-Head Attention implementation."""

    def __init__(self, module_path: str, test_file: str = "test_cases.json", verbose: bool = False):
        """
        Initialize validator.

        Args:
            module_path: Path to the Python file to test
            test_file: Path to test cases JSON file
            verbose: Whether to print detailed output
        """
        self.module_path = module_path
        self.test_file = test_file
        self.verbose = verbose
        self.module = None
        self.test_cases = []
        self.tolerance = 1e-4  # Tolerance for floating point comparisons

    def load_module(self) -> bool:
        """
        Dynamically load the Python module.

        Returns:
            True if successful, False otherwise
        """
        try:
            spec = importlib.util.spec_from_file_location("kv_attention_module", self.module_path)
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)

            # Verify required class exists
            if not hasattr(self.module, 'KVCachedMultiHeadAttention'):
                print("✗ Error: Module must contain 'KVCachedMultiHeadAttention' class")
                return False

            if self.verbose:
                print(f"✓ Successfully loaded module from {self.module_path}")

            return True
        except Exception as e:
            print(f"✗ Error loading module: {e}")
            return False

    def load_test_cases(self) -> bool:
        """
        Load test cases from JSON file.

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.test_file, 'r') as f:
                data = json.load(f)

            # Handle both single test case and multiple test cases
            if isinstance(data, dict):
                if 'test_cases' in data:
                    self.test_cases = data['test_cases']
                elif 'test_case' in data:
                    self.test_cases = [data['test_case']]
                else:
                    print(f"✗ Error: Invalid test file format")
                    return False
            elif isinstance(data, list):
                self.test_cases = data
            else:
                print(f"✗ Error: Test file must contain dict or list")
                return False

            if self.verbose:
                print(f"✓ Loaded {len(self.test_cases)} test case(s) from {self.test_file}")

            return True
        except FileNotFoundError:
            print(f"✗ Error: Test file '{self.test_file}' not found")
            return False
        except json.JSONDecodeError as e:
            print(f"✗ Error parsing JSON: {e}")
            return False
        except Exception as e:
            print(f"✗ Error loading test cases: {e}")
            return False

    def tensor_from_data(self, data: Any) -> torch.Tensor:
        """
        Convert data (list or dict) to PyTorch tensor.

        Args:
            data: Data to convert (list, nested list, or dict with 'values' key)

        Returns:
            PyTorch tensor
        """
        if isinstance(data, dict):
            if 'values' in data:
                values = data['values']
                shape = data.get('shape', None)
                tensor = torch.tensor(values, dtype=torch.float32)
                if shape:
                    tensor = tensor.view(*shape)
                return tensor

        return torch.tensor(data, dtype=torch.float32)

    def run_test_case(self, test_case: Dict[str, Any], test_id: int) -> Tuple[bool, str, Optional[Dict]]:
        """
        Run a single test case.

        Args:
            test_case: Test case dictionary
            test_id: Test case ID for reporting

        Returns:
            Tuple of (passed, message, results_dict)
        """
        try:
            # Extract test parameters
            config = test_case.get('config', {})
            d_model = config.get('d_model', 64)
            num_heads = config.get('num_heads', 4)
            max_cache_len = config.get('max_cache_len', 2048)
            dropout = config.get('dropout', 0.1)

            # Create model
            model_class = self.module.KVCachedMultiHeadAttention
            model = model_class(
                d_model=d_model,
                num_heads=num_heads,
                max_cache_len=max_cache_len,
                dropout=dropout
            )
            model.eval()  # Set to evaluation mode

            # Set random seed for reproducibility
            seed = test_case.get('seed', 42)
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Load input tensors
            inputs = test_case.get('inputs', {})
            query = self.tensor_from_data(inputs['query'])
            key = self.tensor_from_data(inputs['key'])
            value = self.tensor_from_data(inputs['value'])

            # Handle cache
            cache_data = inputs.get('cache', None)
            cache = None
            if cache_data is not None and cache_data != "null":
                cache = {
                    'key': self.tensor_from_data(cache_data['key']) if cache_data.get('key') else None,
                    'value': self.tensor_from_data(cache_data['value']) if cache_data.get('value') else None
                }

            use_causal_mask = inputs.get('use_causal_mask', True)

            # Run forward pass
            with torch.no_grad():
                output, new_cache = model(query, key, value, cache=cache, use_causal_mask=use_causal_mask)

            # Check if expected outputs exist
            expected = test_case.get('expected', None)
            if expected is None:
                # No expected output - just check it runs without error
                return (
                    True,
                    f"✓ Test #{test_id} '{test_case.get('name', 'unnamed')}' - Executed successfully (no expected output to validate)",
                    {
                        'output_shape': list(output.shape),
                        'cache_key_shape': list(new_cache['key'].shape) if new_cache.get('key') is not None else None,
                        'cache_value_shape': list(new_cache['value'].shape) if new_cache.get('value') is not None else None
                    }
                )

            # Validate output shape
            expected_output = self.tensor_from_data(expected['output'])
            if output.shape != expected_output.shape:
                return (
                    False,
                    f"✗ Test #{test_id} - Output shape mismatch: got {output.shape}, expected {expected_output.shape}",
                    None
                )

            # Validate output values
            output_diff = torch.abs(output - expected_output).max().item()
            if output_diff > self.tolerance:
                return (
                    False,
                    f"✗ Test #{test_id} - Output values mismatch: max diff = {output_diff:.6f} (tolerance = {self.tolerance})",
                    {'max_diff': output_diff}
                )

            # Validate cache shapes
            expected_cache = expected.get('cache', {})
            if expected_cache:
                expected_cache_key = self.tensor_from_data(expected_cache['key'])
                expected_cache_value = self.tensor_from_data(expected_cache['value'])

                if new_cache['key'].shape != expected_cache_key.shape:
                    return (
                        False,
                        f"✗ Test #{test_id} - Cache key shape mismatch: got {new_cache['key'].shape}, expected {expected_cache_key.shape}",
                        None
                    )

                # Validate cache values
                cache_diff = torch.abs(new_cache['key'] - expected_cache_key).max().item()
                if cache_diff > self.tolerance:
                    return (
                        False,
                        f"✗ Test #{test_id} - Cache key values mismatch: max diff = {cache_diff:.6f}",
                        {'cache_max_diff': cache_diff}
                    )

            # All validations passed
            return (
                True,
                f"✓ Test #{test_id} '{test_case.get('name', 'unnamed')}' - PASSED",
                {
                    'output_shape': list(output.shape),
                    'max_output_diff': output_diff,
                    'cache_key_shape': list(new_cache['key'].shape) if new_cache.get('key') is not None else None
                }
            )

        except Exception as e:
            import traceback
            error_msg = str(e)
            if self.verbose:
                error_msg += "\n" + traceback.format_exc()
            return (
                False,
                f"✗ Test #{test_id} - Runtime error: {error_msg}",
                None
            )

    def run_all_tests(self) -> Tuple[int, int, List[str]]:
        """
        Run all test cases.

        Returns:
            Tuple of (passed_count, total_count, messages)
        """
        passed = 0
        total = len(self.test_cases)
        messages = []

        for i, test_case in enumerate(self.test_cases, 1):
            success, message, results = self.run_test_case(test_case, i)
            messages.append(message)

            if self.verbose and results:
                messages.append(f"  Results: {results}")

            if success:
                passed += 1

        return passed, total, messages

    def validate(self) -> bool:
        """
        Run complete validation.

        Returns:
            True if all tests pass, False otherwise
        """
        print("=" * 70)
        print("KV-Cached Multi-Head Attention - Validator")
        print("=" * 70)

        # Load module
        if not self.load_module():
            return False

        # Load test cases
        if not self.load_test_cases():
            return False

        print(f"\nRunning {len(self.test_cases)} test case(s)...\n")

        # Run tests
        passed, total, messages = self.run_all_tests()

        # Print results
        for msg in messages:
            print(msg)

        print("\n" + "=" * 70)
        print(f"Results: {passed}/{total} tests passed")
        print("=" * 70)

        if passed == total:
            print("✓ All tests passed! Great job!")
            return True
        else:
            print(f"✗ {total - passed} test(s) failed. Keep debugging!")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate KV-Cached Multi-Head Attention implementation"
    )
    parser.add_argument(
        '--file',
        type=str,
        default='kv_attention.py',
        help='Path to the Python file to validate (default: kv_attention.py)'
    )
    parser.add_argument(
        '--test-file',
        type=str,
        default='test_cases.json',
        help='Path to test cases JSON file (default: test_cases.json)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed output'
    )

    args = parser.parse_args()

    # Create validator
    validator = AttentionValidator(
        module_path=args.file,
        test_file=args.test_file,
        verbose=args.verbose
    )

    # Run validation
    success = validator.validate()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
