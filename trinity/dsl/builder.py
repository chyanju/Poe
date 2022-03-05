from typing import Union
import sexpdata
from .node import *
from ..spec import TrinitySpec, Production, EnumType
from ..visitor import GenericVisitor


class ProductionVisitor(GenericVisitor):
    _children: List[Node]

    def __init__(self, children: List[Node]):
        self._children = children

    def visit_enum_production(self, prod, **kwargs) -> Node:
        return AtomNode(prod, **kwargs)

    def visit_param_production(self, prod, **kwargs) -> Node:
        return ParamNode(prod, **kwargs)

    def visit_function_production(self, prod, **kwargs) -> Node:
        return ApplyNode(prod, self._children, **kwargs)


class Builder:
    '''A factory class to build AST node'''

    _spec: TrinitySpec

    def __init__(self, spec: TrinitySpec):
        self._spec = spec

    def _make_node(self, prod: Production, children: List[Node] = [], **kwargs) -> Node:
        return ProductionVisitor(children).visit(prod, **kwargs)

    def make_node(self, src: Union[int, Production], children: List[Node] = [], **kwargs) -> Node:
        '''
        Create a node with the given production index and children.
        Raise `KeyError` or `ValueError` if an error occurs
        '''
        if isinstance(src, int):
            return self._make_node(self._spec.get_production_or_raise(src), children, **kwargs)
        elif isinstance(src, Production):
            # Sanity check first
            prod = self._spec.get_production_or_raise(src.id)
            if src != prod:
                raise ValueError(
                    'DSL Builder found inconsistent production instance')
            return self._make_node(prod, children, **kwargs)
        else:
            raise ValueError(
                'make_node() only accepts int or production, but found {}'.format(src))

    def make_enum(self, name: str, value: str) -> Node:
        '''
        Convenient method to create an enum node.
        Raise `KeyError` or `ValueError` if an error occurs
        '''
        ty = self.get_type_or_raise(name)
        prod = self.get_enum_production_or_raise(ty, value)
        return self.make_node(prod.id)

    def make_param(self, index: int) -> Node:
        '''
        Convenient method to create a param node.
        Raise `KeyError` or `ValueError` if an error occurs
        '''
        prod = self.get_param_production_or_raise(index)
        return self.make_node(prod.id)

    def make_apply(self, name: str, args: List[Node]) -> Node:
        '''
        Convenient method to create an apply node.
        Raise `KeyError` or `ValueError` if an error occurs
        '''
        prod = self.get_function_production_or_raise(name)
        return self.make_node(prod.id, args)

    def from_jsonexp(self, jsonexp) -> Node:
        '''
        Convert json expression (from ReMorpheus based benchmark json, "solution" field) into Trinity node.
        Note that for ReMorpheus deduction, you still need to provide tag information.
        '''
        if isinstance(jsonexp, list):
            if isinstance(jsonexp[0], list):
                if jsonexp[0][0] == "function":
                    tmp_function_prod = self._spec.get_function_production(jsonexp[0][1])
                    tmp_arg_nodes = [self.from_jsonexp(p) for p in jsonexp[1:]]
                    tmp_function_node = self.make_node( tmp_function_prod, tmp_arg_nodes )
                    return tmp_function_node
                else:
                    raise Exception("Invalid: {}".format(jsonexp))
            elif jsonexp[0] == "param":
                return self.make_node( self._spec.get_param_production(jsonexp[1]) )
            elif jsonexp[0] == "enum":
                return self.make_node(
                    self._spec.get_enum_production(
                        self._spec.get_type(jsonexp[1]),
                        jsonexp[2]
                    )
                )
            # elif jsonexp[0] == "function": this should be captured by parent layer
            else:
                raise Exception("Invalid: {}".format(jsonexp))
        else:
            # do not process non-list unit alone
            raise Exception("Invalid: {}".format(jsonexp))

    def _from_sexp(self, sexp) -> Node:
        if not isinstance(sexp, list) or len(sexp) < 2 or not isinstance(sexp[0].value(), str):
            # None of our nodes serializes to atom
            msg = 'Cannot parse sexp into dsl.Node: {}'.format(sexp)
            raise ValueError(msg)
        sym = sexp[0].value()

        # First check for param node
        if sym == '@param':
            index = int(sexp[1])
            return self.make_param(index)

        # Next, check for atom node
        ty = self.get_type(sym)
        if ty is not None and ty.is_enum():
            if isinstance(sexp[1], list):
                # Could be a enum list
                value = [str(x) for x in sexp[1]]
                return self.make_enum(ty.name, value)
            else:
                value = str(sexp[1])
                return self.make_enum(ty.name, value)

        # Finally, check for apply node
        args = [self._from_sexp(x) for x in sexp[1:]]
        return self.make_apply(sym, args)

    def from_sexp_string(self, sexp_str: str) -> Node:
        '''
        Convenient method to create an AST from an sexp string.
        Raise `KeyError` or `ValueError` if an error occurs
        '''
        try:
            sexp = sexpdata.loads(sexp_str)
        # This library is liberal on its exception raising...
        except Exception as e:
            raise ValueError('Sexp parsing error: {}'.format(e))
        return self._from_sexp(sexp)

    # For convenience, expose all methods in TrinitySpec so that the client do not need to keep a reference of it
    def __getattr__(self, attr):
        return getattr(self._spec, attr)
