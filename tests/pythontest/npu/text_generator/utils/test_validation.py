# Copyright Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
import unittest

from mindie_llm.utils.validation import (
    ValidationError,
    InconsistencyError,
    OutOfBoundsError,
    UnsupportedTypeError,
    validate_list,
    validate_string,
    SAFE_LIST_LENGTH,
    SAFE_STRING_LENGTH,
)


class TestValidationError(unittest.TestCase):
    """Test ValidationError exception class."""

    def test_validation_error_init(self):
        """Test ValidationError initialization with param_name and detail."""
        error = ValidationError("test_param", "test detail")
        expected_msg = "The parameter `test_param` is invalid: test detail"
        self.assertEqual(str(error), expected_msg)
        self.assertIn("test_param", str(error))
        self.assertIn("test detail", str(error))


class TestInconsistencyError(unittest.TestCase):
    """Test InconsistencyError exception class."""

    def test_inconsistency_error_init(self):
        """Test InconsistencyError initialization."""
        error = InconsistencyError("test_param", "attr_a", "attr_b")
        error_str = str(error)
        self.assertIn("test_param", error_str)
        self.assertIn("attr_a", error_str)
        self.assertIn("attr_b", error_str)
        self.assertIn("is not equal to", error_str)
        self.assertIn("The parameter `test_param` is invalid:", error_str)


class TestOutOfBoundsError(unittest.TestCase):
    """Test OutOfBoundsError exception class."""

    def test_out_of_bounds_error_init_with_int(self):
        """Test OutOfBoundsError initialization with int limit value."""
        error = OutOfBoundsError("test_param", "SAFE_LIMIT", 100)
        error_str = str(error)
        self.assertIn("test_param", error_str)
        self.assertIn("SAFE_LIMIT", error_str)
        self.assertIn("100", error_str)
        self.assertIn("exceeds the safety limit", error_str)
        self.assertIn("mindie_llm.utils.validation", error_str)

    def test_out_of_bounds_error_init_with_float(self):
        """Test OutOfBoundsError initialization with float limit value."""
        error = OutOfBoundsError("temperature", "LOWER_SAFE_TEMPERATURE", 0.0)
        error_str = str(error)
        self.assertIn("0.0", error_str)
        self.assertIn("temperature", error_str)

    def test_out_of_bounds_error_init_with_str(self):
        """Test OutOfBoundsError initialization with str limit value."""
        error = OutOfBoundsError("test_param", "SAFE_LIMIT", "custom_limit")
        error_str = str(error)
        self.assertIn("custom_limit", error_str)


class TestUnsupportedTypeError(unittest.TestCase):
    """Test UnsupportedTypeError exception class."""

    def test_unsupported_type_error_init(self):
        """Test UnsupportedTypeError initialization."""
        error = UnsupportedTypeError("test_param", "List[Any]")
        error_str = str(error)
        self.assertIn("test_param", error_str)
        self.assertIn("List[Any]", error_str)
        self.assertIn("type is not supported", error_str)
        self.assertIn("The parameter `test_param` is invalid:", error_str)


class TestValidateList(unittest.TestCase):
    """Test validate_list function."""

    def test_validate_list_normal_case_empty(self):
        """Test validate_list with empty list - should pass."""
        validate_list("test_list", [])

    def test_validate_list_normal_case_within_limit(self):
        """Test validate_list with valid list within safe length."""
        validate_list("test_list", [1, 2, 3])
        # boundary value: exactly at limit
        validate_list("test_list", [None] * SAFE_LIST_LENGTH)

    def test_validate_list_non_list_type_string(self):
        """Test validate_list with string type should raise UnsupportedTypeError."""
        with self.assertRaises(UnsupportedTypeError) as context:
            validate_list("test_param", "not a list")
        self.assertIn("test_param", str(context.exception))
        self.assertIn("List[Any]", str(context.exception))

    def test_validate_list_non_list_type_tuple(self):
        """Test validate_list with tuple type should raise UnsupportedTypeError."""
        with self.assertRaises(UnsupportedTypeError):
            validate_list("test_param", (1, 2, 3))

    def test_validate_list_non_list_type_dict(self):
        """Test validate_list with dict type should raise UnsupportedTypeError."""
        with self.assertRaises(UnsupportedTypeError):
            validate_list("test_param", {"key": "value"})

    def test_validate_list_exceeds_safe_length(self):
        """Test validate_list with list exceeding SAFE_LIST_LENGTH."""
        too_long_list = [1] * (SAFE_LIST_LENGTH + 1)
        with self.assertRaises(OutOfBoundsError) as context:
            validate_list("test_param", too_long_list)
        error_msg = str(context.exception)
        self.assertIn("test_param", error_msg)
        self.assertIn("SAFE_LIST_LENGTH", error_msg)
        self.assertIn(str(SAFE_LIST_LENGTH), error_msg)
        self.assertIn("mindie_llm.utils.validation", error_msg)


class TestValidateString(unittest.TestCase):
    """Test validate_string function."""

    def test_validate_string_normal_case_empty(self):
        """Test validate_string with empty string - should pass."""
        validate_string("test_str", "")

    def test_validate_string_normal_case_within_limit(self):
        """Test validate_string with valid string within safe length."""
        validate_string("test_str", "hello")
        # boundary value: exactly at limit
        validate_string("test_str", "a" * SAFE_STRING_LENGTH)

    def test_validate_string_exceeds_safe_length(self):
        """Test validate_string with string exceeding SAFE_STRING_LENGTH."""
        too_long_str = "a" * (SAFE_STRING_LENGTH + 1)
        with self.assertRaises(OutOfBoundsError) as context:
            validate_string("test_param", too_long_str)
        error_msg = str(context.exception)
        self.assertIn("test_param", error_msg)
        self.assertIn("SAFE_STRING_LENGTH", error_msg)
        self.assertIn(str(SAFE_STRING_LENGTH), error_msg)
        self.assertIn("mindie_llm.utils.validation", error_msg)


if __name__ == "__main__":
    unittest.main()