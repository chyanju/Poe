from abc import ABC, abstractmethod
from typing import Callable, Iterable, Optional, Any, Tuple, List

from ..dsl import Node

class InterpreterError(Exception):

    @abstractmethod
    def __init__(self):
        super().__init__()

class GeneralError(InterpreterError):
    def __init__(self, msg: str=""):
        super().__init__()
        self._msg = msg

    def __str__(self):
        return "GeneralError:{}".format(self._msg)

# this is specifically defined to pair with the lazy combination iterator in LineSkeletonEnumerator
# fixme: the context here is the interpreter combination context, not the original visitor context
#        try to use another name to resolve this ambiguity
class EnumAssertion(InterpreterError):
    def __init__(self, context: Tuple[int, ...], extlen: int, condition: Callable[[Any], bool], tag: Any = None):
        super().__init__()
        self._context = context
        self._extlen = extlen # extended length, the length of combination
        self._condition = condition
        self._tag = tag

    def __str__(self):
        return "EnumAssertion:{}".format(self._tag)

# this is used for skeleton level deduction and CDCL
# fixme: this is an initial version
class SkeletonAssertion(InterpreterError):
    def __init__(self, example: Any = None, prog: Any = None, tag: Any = None):
        super().__init__()
        # prog: the program that triggers the analysis
        self._example = example
        self._prog = prog
        self._tag = tag

    def __str__(self):
        return "SkeletonAssertion:{}".format(self._tag)

class EqualityAssertion(InterpreterError):
    def __init__(self, context: Tuple[int, ...], condition: Callable[[Any], bool], tag: Any = None):
        super().__init__()
        self._context = context
        self._condition = condition
        self._tag = tag

    def __str__(self):
        return "EqualityAssertion:{}".format(self._tag)

class ComponentError(InterpreterError):
    def __init__(self, context: Tuple[int, ...], condition: Callable[[Any], bool], tag: Any = None):
        super().__init__()
        self._context = context
        self._condition = condition
        self._tag = tag

    def __str__(self):
        return "ComponentError:{}".format(self._tag)