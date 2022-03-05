from typing import Iterator
from itertools import product, permutations
from ..spec import TrinitySpec, Type
from ..spec.production import Production, FunctionProduction, ParamProduction, EnumProduction
from ..dsl import Node, Builder
from .enumerator import Enumerator

class ExhaustiveEnumerator(Enumerator):

    def __init__(self, spec: TrinitySpec, sizes: List[int] = list(range(1,20))):
        assert len(sizes) > 0, "You need to provide at least one target size for exhaustive enumerator, got: {}.".format(sizes)
        for p in sizes:
            assert p >= 1, "Every size in provided size list must be >= 1, got: {}.".format(sizes)
        self._spec = spec
        self._sizes = sizes

        self._curr_sk_size_ind = -1
        self._curr_sk_size = None
        self._curr_sk_iterator = None

    def function_production_to_line_skeleton_command(self, arg_fp: FunctionProduction):
        pass

    def next_valid_line_skeleton_iterator(self):
        self._curr_sk_size_ind += 1
        if self._curr_sk_size_ind >= len(self._sizes):
            return None




