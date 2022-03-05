from abc import ABC, abstractmethod
from typing import Callable, Iterable, List, Any, Tuple

from .. import dsl as D
from ..dsl import Node

from .error import EnumAssertion

class Interpreter(ABC):

    @abstractmethod
    def eval(self, prog: Node, inputs: List[Any]) -> Any:
        '''
        Evaluate a DSL `prog` on input `inputs`. The output is returned.
        This is a covenient wrapper over `eval_step` that repeatedly invoke the generator until we get the final result.
        '''
        raise NotImplementedError

    # def assertEnum(self,
    #     node: Node,
    #     context: Tuple[int, ...], # current combination context, filled with production id
    #     comb: Tuple[int, ...], # current combination, filled with produxtion id, usually with more slots filled than context
    #     cond: Callable[[Any], bool], # condition function, callable
    #     tag: Any = None, # used to identify additional information
    # ) -> None:
    #     if node.is_leaf():
    #         raise RuntimeError("assertEnum() cannot be called from within a leaf node: {}".format(node))
    #     if not cond(comb):
    #         raise EnumAssertion(context, cond, tag=tag)

    def assertEnum(self,
        node: Node,
        context: Tuple[int, ...], # current combination context, filled with production id
        comb: Tuple[int, ...], # current combination, filled with produxtion id, usually with more slots filled than context
        cond: Callable[[Any], bool], # condition function, callable
        tag: Any = None, # used to identify additional information
    ) -> None:
        if node.is_leaf():
            raise RuntimeError("assertEnum() cannot be called from within a leaf node: {}".format(node))
        if not cond(comb):
            raise EnumAssertion(context=context, extlen=len(comb), condition=cond, tag=tag)