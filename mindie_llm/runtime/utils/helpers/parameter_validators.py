# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import argparse
from enum import Enum
from typing import Any, Dict, List, Type, Union, Tuple, Iterable, Optional

from mindie_llm.runtime.utils.helpers.safety.file import check_file_safety
from mindie_llm.runtime.utils.helpers.safety.path import standardize_path
from mindie_llm.utils.log.logging import logger


class Field:
    def __init__(self, ge: Optional[float] = None, gt: Optional[float] = None, le: Optional[float] = None,
                lt: Optional[float] = None, min_length: Optional[int] = None, max_length: Optional[int] = None):
        constraints: Dict[str, Optional[Any]] = {
            'ge': ge, 'gt': gt, 'le': le, 'lt': lt,
            'min_length': min_length, 'max_length': max_length
        }
        self._validate_not_all_none(constraints)
        self._validate_range_constraints(constraints)
        self._validate_length_constraints(constraints)
        self._validate_combined_constraints(constraints)
        self.ge, self.gt, self.le, self.lt = ge, gt, le, lt
        self.min_length, self.max_length = min_length, max_length

    @staticmethod
    def _validate_not_all_none(constraints: Dict[str, Optional[Any]]) -> None:
        if all(value is None for value in constraints.values()):
            raise ValueError("At least one constraint must be specified")

    @staticmethod
    def _validate_range_constraints(constraints: Dict[str, Optional[Any]]) -> None:
        ge, gt, le, lt = constraints['ge'], constraints['gt'], constraints['le'], constraints['lt']
        if ge is not None and gt is not None:
            raise ValueError("Cannot specify both 'ge' and 'gt'")
        if le is not None and lt is not None:
            raise ValueError("Cannot specify both 'le' and 'lt'")
        if ge is not None and le is not None and ge > le:
            raise ValueError("'ge' cannot be greater than 'le'")
        if gt is not None and lt is not None and gt > lt:
            raise ValueError("'gt' cannot be greater than 'lt'")

    @staticmethod
    def _validate_length_constraints(constraints: Dict[str, Optional[Any]]) -> None:
        min_length, max_length = constraints['min_length'], constraints['max_length']
        if min_length is not None and max_length is not None and min_length > max_length:
            raise ValueError("'min_length' cannot be greater than 'max_length'")

    @staticmethod
    def _validate_combined_constraints(constraints: Dict[str, Optional[Any]]) -> None:
        if any(constraints[key] is not None for key in ['ge', 'gt', 'le', 'lt']) and any(
                constraints[key] is not None for key in ['min_length', 'max_length']):
            raise ValueError("Cannot specify both range constraints and length constraints")

    def extract_constraint(self) -> str:
        """
        Extracts the constraints of the Field instance as a string.

        This method iterates over the instance's dictionary and constructs a string
        representation of the constraints that are not None.

        Returns:
            str: A string representation of the constraints in the format "{key: value, ...}".
        Returns:

        """
        return "{" + ", ".join(f"{k}: {v}" for k, v in self.__dict__.items() if v is not None) + "}"


class ValidationPipelineStatus(Enum):
    """
    Enum representing the status of the validation pipeline.

    Attributes:
        CONTINUE: Indicates that the validation should continue to the next step in the pipeline.
        BREAK: Indicates that the validation should stop and no further steps should be executed.
    """
    CONTINUE = "continue"
    BREAK = "break"


class ValidationStrategies:
    @staticmethod
    def validate_none(value: Any, allow_none: bool) -> ValidationPipelineStatus:
        if value is None:
            if allow_none:
                return ValidationPipelineStatus.BREAK
            raise ValueError("None value is not allowed.")
        return ValidationPipelineStatus.CONTINUE

    @staticmethod
    def validate_special_value(value: Any, special_values: List[Any]) -> ValidationPipelineStatus:
        return ValidationPipelineStatus.BREAK if value in special_values else ValidationPipelineStatus.CONTINUE

    @staticmethod
    def validate_range_list(value: Any, range_list: List[Any]) -> ValidationPipelineStatus:
        if value in range_list:
            return ValidationPipelineStatus.BREAK
        else:
            raise ValueError(f"Value should be in one of these: {range_list}")

    @staticmethod
    def validate_type(value: Any, expected_types: Union[Type, Tuple[Type, ...]]) -> ValidationPipelineStatus:
        if not isinstance(value, expected_types):
            expected_type_names = ', '.join(
                t.__name__ for t in (expected_types if isinstance(expected_types, tuple) else (expected_types,)))
            raise ValueError(f"Value must be one of the following types: {expected_type_names}")
        return ValidationPipelineStatus.CONTINUE

    @staticmethod
    def validate_range(value: Any, field_validators: List[Field], constraints: str = None) -> ValidationPipelineStatus:
        if constraints is None:
            constraints = ValidationStrategies.extract_constraints(field_validators)
        for field in field_validators:
            if field.ge is not None and value < field.ge:
                continue
            if field.gt is not None and value <= field.gt:
                continue
            if field.le is not None and value > field.le:
                continue
            if field.lt is not None and value >= field.lt:
                continue
            if field.min_length is not None and len(value) < field.min_length:
                continue
            if field.max_length is not None and len(value) > field.max_length:
                continue
            return ValidationPipelineStatus.CONTINUE
        raise ValueError(f"Value should be in one of these: {constraints}")

    @staticmethod
    def validate_file(value: Any) -> ValidationPipelineStatus:
        standard_value = standardize_path(value)
        check_file_safety(standard_value)
        return ValidationPipelineStatus.CONTINUE

    @staticmethod
    def extract_constraints(fields: Iterable[Field]) -> str:
        """
        Extracts the constraints from a list of Field instances.

        This method iterates over the provided Field instances and constructs a string
        representation of their constraints by calling the `extract_constraint` method
        on each Field instance.

        Args:
            fields (Iterable[Field]): An iterable of Field instances.

        Returns:
            str: A string representation of the constraints for all provided Field instances.
                Example: "{ge: 0, lt: 1}; {ge: 2, lt: 3}"
        """
        return "; ".join(field.extract_constraint() for field in fields)


class ValidationPipeline:
    def __init__(self):
        self.pipeline: List[Tuple[callable, Tuple[Any, ...]]] = []

    def add(self, strategy_method: callable, *args: Any, position: int = None) -> None:
        if position is None:
            self.pipeline.append((strategy_method, args))
        else:
            self.pipeline.insert(position, (strategy_method, args))

    def remove(self, strategy_method: callable, position: int = None) -> None:
        if position is None:
            self.pipeline = [item for item in self.pipeline if item[0] != strategy_method]
        else:
            if 0 <= position < len(self.pipeline) and self.pipeline[position][0] == strategy_method:
                self.pipeline.pop(position)

    def update(self, strategy_method: callable, *args: Any, position: int = None) -> None:
        if position is not None:
            if 0 <= position < len(self.pipeline):
                self.pipeline[position] = (strategy_method, args)
        else:
            for i, (method, _) in enumerate(self.pipeline):
                if method == strategy_method:
                    self.pipeline[i] = (strategy_method, args)
                    break

    def refresh(self) -> None:
        self.pipeline.clear()

    def validate(self, value: Any) -> None:
        for strategy_method, args in self.pipeline:
            result = strategy_method(value, *args)
            if result == ValidationPipelineStatus.BREAK:
                break


class ParameterValidator:
    requires_fields: bool = True

    def __init__(self, value_type: Union[Type, Tuple[Type, ...]] | None, *fields: Field, allow_none: bool = False,
                special_values: List[Any] = None):
        if self.requires_fields and not fields:
            raise ValueError("At least one Field parameter must be provided.")
        self.allow_none = allow_none
        self.special_values = special_values if special_values is not None else []
        self.fields = fields
        self.constraints = ValidationStrategies.extract_constraints(self.fields)
        self.pipeline = ValidationPipeline()
        self.create_validation_pipeline(value_type)

    def validate(self, value: Any, key: Any = None) -> None:
        try:
            self.pipeline.validate(value)
        except ValueError as e:
            if key is not None:
                raise ValueError(f"Validation failed for argument '{key}'") from e
            else:
                raise ValueError("Validation failed for argument") from e

    def create_validation_pipeline(self, value_type: Union[Type, Tuple[Type, ...]] | None):
        self.pipeline.refresh()
        self.pipeline.add(ValidationStrategies.validate_none, self.allow_none)
        self.pipeline.add(ValidationStrategies.validate_special_value, self.special_values)
        if value_type:
            self.pipeline.add(ValidationStrategies.validate_type, value_type)
        self.pipeline.add(ValidationStrategies.validate_range, self.fields, self.constraints)


class StringParameterValidator(ParameterValidator):
    def __init__(self, *fields: Field, special_values: List[str] = None, allow_none: bool = False):
        super().__init__(str, *fields, allow_none=allow_none, special_values=special_values)


class IntParameterValidator(ParameterValidator):
    def __init__(self, *fields: Field, special_values: List[int] = None, allow_none: bool = False):
        super().__init__(int, *fields, allow_none=allow_none, special_values=special_values)


class FloatParameterValidator(ParameterValidator):
    def __init__(self, *fields: Field, special_values: List[float] = None, allow_none: bool = False):
        super().__init__((float, int), *fields, allow_none=allow_none, special_values=special_values)


class BooleanParameterValidator(ParameterValidator):
    requires_fields: bool = False

    def __init__(self, allow_none: bool = False):
        super().__init__(bool, allow_none=allow_none)

    def create_validation_pipeline(self, value_type: Union[Type, Tuple[Type, ...]] | None):
        self.pipeline.refresh()
        self.pipeline.add(ValidationStrategies.validate_none, self.allow_none)
        self.pipeline.add(ValidationStrategies.validate_type, value_type)


class RangeParamaterValidator(ParameterValidator):
    requires_fields: bool = False

    def __init__(self, range_list: List[Any] = None, allow_none: bool = False):
        self.range_list = range_list
        super().__init__(Any, allow_none=allow_none)

    def create_validation_pipeline(self, value_type: Union[Type, Tuple[Type, ...]] | None):
        self.pipeline.refresh()
        self.pipeline.add(ValidationStrategies.validate_none, self.allow_none)
        self.pipeline.add(ValidationStrategies.validate_range_list, self.range_list)


class ListParameterValidator(ParameterValidator):
    def __init__(self, element_validator: ParameterValidator, *fields: Field, allow_none: bool = False):
        super().__init__(list, *fields, allow_none=allow_none)
        self.element_validator = element_validator

    def validate_elements(self, value: list) -> ValidationPipelineStatus:
        for element in value:
            self.element_validator.validate(element)
        return ValidationPipelineStatus.CONTINUE

    def create_validation_pipeline(self, value_type: Union[Type, Tuple[Type, ...]] | None):
        self.pipeline.refresh()
        self.pipeline.add(ValidationStrategies.validate_none, self.allow_none)
        self.pipeline.add(ValidationStrategies.validate_type, value_type)
        self.pipeline.add(ValidationStrategies.validate_range, self.fields, self.constraints)
        self.pipeline.add(self.validate_elements)


class DictionaryParameterValidator(ParameterValidator):
    requires_fields: bool = False

    def __init__(
            self,
            element_validator_mapping: Dict[Any, ParameterValidator],
            allow_none: bool = False,
            allow_addition_key: bool = True
        ):
        self.allow_addition_key = allow_addition_key
        super().__init__(dict, allow_none=allow_none)
        self.element_validator_mapping = element_validator_mapping

    def validate_keys(self, value: dict) -> ValidationPipelineStatus:
        if set(value.keys()) != set(self.element_validator_mapping.keys()):
            raise ValueError("The keys of the dictionary do not match the expected keys.")
        return ValidationPipelineStatus.CONTINUE

    def validate_elements(self, value: dict) -> ValidationPipelineStatus:
        for ele_key, ele_value in value.items():
            if ele_key in self.element_validator_mapping:
                self.element_validator_mapping.get(ele_key).validate(ele_value, key=ele_key)
        return ValidationPipelineStatus.CONTINUE

    def create_validation_pipeline(self, value_type: Union[Type, Tuple[Type, ...]] | None):
        self.pipeline.refresh()
        self.pipeline.add(ValidationStrategies.validate_none, self.allow_none)
        self.pipeline.add(ValidationStrategies.validate_type, value_type)
        if not self.allow_addition_key:
            self.pipeline.add(self.validate_keys)
        self.pipeline.add(self.validate_elements)


class FileParameterValidator(ParameterValidator):
    requires_fields: bool = False

    def __init__(self, allow_none: bool = False):
        super().__init__(str, allow_none=allow_none)
    
    def create_validation_pipeline(self, value_type: Union[Type, Tuple[Type, ...]] | None):
        self.pipeline.refresh()
        self.pipeline.add(ValidationStrategies.validate_none, self.allow_none)
        self.pipeline.add(ValidationStrategies.validate_type, value_type)
        self.pipeline.add(ValidationStrategies.validate_file)


class CompositeParameterValidator(ParameterValidator):
    """
    A validator for composite parameters that can validate multiple types of values.

    This validator allows for the validation of different types of values using a dictionary
    of type-specific validators. It supports validation for types such as int, str, list, etc.

    Attributes:
        validators (Dict[Type, ParameterValidator]): A dictionary mapping types to their respective validators.
    """
    requires_fields: bool = False

    def __init__(self, validators: Dict[Type, ParameterValidator], allow_none: bool = False):
        super().__init__(None, allow_none=allow_none)
        self.validators = validators

    def validate(self, value: Any, key: Any = None) -> None:
        """
        Validates the provided value using the appropriate validator based on its type.

        Args:
            value (Any): The value to be validated.
            key (Any): The key associated with the value, used for error messages. Defaults to None.

        Raises:
            ValueError: If no validator is found for the type of the value or if validation fails.
        """
        if ValidationStrategies.validate_none(value, self.allow_none) == ValidationPipelineStatus.CONTINUE:
            value_type = type(value)
            if value_type in self.validators:
                self.validators[value_type].validate(value, key=key)
            else:
                raise ValueError(f"No validator found for type {value_type.__name__}")


class ArgumentAction(Enum):
    STORE = 'store'
    STORE_TRUE = 'store_true'
    STORE_FALSE = 'store_false'
    STORE_CONST = 'store_const'
    APPEND = 'append'
    COUNT = 'count'


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.argument_validators: Dict[str, Union[ParameterValidator, Dict[Any, ParameterValidator]]] = {}
        super().__init__(*args, **kwargs)

    def add_argument(self, *args: Any, validator: Union[ParameterValidator, Dict[Any, ParameterValidator]] = None,
                    **kwargs: Any) -> argparse.Action:
        arguments = super().add_argument(*args, **kwargs)
        if validator is not None:
            self.argument_validators.update({arguments.dest: validator})
        return arguments

    def parse_args(self, args: List[str] = None, namespace: argparse.Namespace = None) -> argparse.Namespace:
        args_all = super().parse_args(args, namespace)
        ParameterValidator.context = vars(args_all)
        self._validate_arguments(args_all)
        return args_all

    def update_argument(self, old_name: str, new_dest: str = None, new_option_strings: List[str] = None,
                        **kwargs: Any) -> None:
        action = self._find_action(old_name)
        self._update_action(action, new_dest, new_option_strings, **kwargs)

    def _find_action(self, name: str) -> argparse.Action:
        for action in self._actions:
            if name in action.option_strings or name == action.dest:
                return action
        raise ValueError(f"Action for argument '{name}' not found.")

    def _update_action(self, action: argparse.Action, new_dest: str = None, new_option_strings: List[str] = None,
                    **kwargs: Any) -> None:
        if new_dest:
            action.dest = new_dest
        if new_option_strings:
            self._update_option_strings(action, new_option_strings)
        self._update_kwargs(action, **kwargs)

    def _update_option_strings(self, action: argparse.Action, new_option_strings: List[str]) -> None:
        for option_string in action.option_strings:
            if option_string in self._option_string_actions:
                del self._option_string_actions[option_string]
        action.option_strings = new_option_strings
        for option_string in new_option_strings:
            self._option_string_actions[option_string] = action

    def _update_kwargs(self, action: argparse.Action, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(action, key):
                setattr(action, key, value)
            else:
                raise AttributeError(f"Argument '{action.dest}' does not have attribute '{key}'.")

    def _validate_arguments(self, args_all: argparse.Namespace) -> None:
        for arg, value in vars(args_all).items():
            if arg in self.argument_validators:
                self._validate_argument(arg, value)

    def _validate_argument(self, arg: str, value: Any) -> None:
        validator = self.argument_validators.get(arg, None)
        if validator is None:
            raise argparse.ArgumentTypeError(f"The validator of {arg} has not been set")
        try:
            validator.validate(value, key=arg)
        except ValueError as e:
            err_msg = f"Validation error for argument '{arg}': {e}"
            logger.error(err_msg)
            raise argparse.ArgumentError(f"{err_msg}") from e
