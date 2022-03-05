from typing import cast, List, Any
from abc import ABC, abstractmethod
from sexpdata import Symbol
from ..spec import Type, Production, EnumProduction, ParamProduction, FunctionProduction

class Mutable():
    '''
    A helper class that is mutable.
    '''
    def __init__(self, v):
        self.v = v
    

class Node(ABC):
    '''Generic and abstract AST Node'''

    _prod: Production
    _tag: Any

    @abstractmethod
    def __init__(self, prod: Production, tag: Any = None):
        self._prod = prod
        # _tag here is used to store any additional information
        # that you want to track on a specific node
        self._tag = tag

    @property
    def production(self) -> Production:
        return self._prod

    @property
    def tag(self) -> Any:
        return self._tag

    @property
    def type(self) -> Type:
        return self._prod.lhs

    @abstractmethod
    def is_leaf(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def is_enum(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def is_param(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def is_apply(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def children(self) -> List['Node']:
        raise NotImplementedError

    @abstractmethod
    def to_sexp(self):
        raise NotImplementedError

    @abstractmethod
    def to_jsonexp(self):
        raise NotImplementedError


class LeafNode(Node):
    '''Generic and abstract class for AST nodes that have no children'''
    @abstractmethod
    def __init__(self, prod: Production, **kwargs):
        super().__init__(prod, **kwargs)
        if prod.is_function():
            raise ValueError(
                'Cannot construct an AST leaf node from a FunctionProduction')

    def is_leaf(self) -> bool:
        return True

    def is_apply(self) -> bool:
        return False

class HoleNode(LeafNode):
    '''Special node that keeps the un-derived status of a program, i.e., a hole. No production is determined yet'''

    _type: Type

    def __init__(self, type: Type, **kwargs):
        self._prod = None
        self._tag = None
        self._type = type

    def make_copy(self):
        return HoleNode(type=self._type)

    @property
    def production(self) -> Production:
        # note: a hole node's production is not determined yet
        raise NotImplementedError("Invalid method on a hole node")

    @property
    def tag(self) -> Any:
        raise NotImplementedError("Invalid method on a hole node")

    @property
    def type(self) -> Type:
        return self._type

    @property
    def children(self) -> List['Node']:
        raise NotImplementedError("Invalid method on a hole node")

    def is_leaf(self) -> bool:
        return True

    def is_apply(self) -> bool:
        return False

    def is_enum(self) -> bool:
        return False

    def is_param(self) -> bool:
        return False

    def to_sexp(self):
        raise NotImplementedError("Invalid method on a collapsed node")

    def to_jsonexp(self):
        raise NotImplementedError("Invalid method on a collapsed node")

    def __repr__(self) -> str:
        return "HoleNode<{}>".format(self._type)

    def __str__(self) -> str:
        return "Hole<{}>".format(self._type)

class CollapsedNode(LeafNode):
    '''Special node that stores intermediate results for partial evaluation'''

    _node: Any
    _res: Any

    def __init__(self, node: Any, **kwargs):
        # note: don't call any parent's __init__ function, just set all the variables to None if they are not needed
        #       since this is a special kind of node and it should never appears in normal enumerator's flows
        self._prod = None
        self._tag = None
        self._node = node
        self._res = None # this should be set by external interpreter

    @property
    def node(self):
        return self._node

    @property
    def res(self):
        return self._res

    def make_copy(self):
        raise NotImplementedError("Invalid method on a collapsed node")

    @property
    def production(self) -> Production:
        raise NotImplementedError("Invalid method on a collapsed node")

    @property
    def tag(self) -> Any:
        raise NotImplementedError("Invalid method on a collapsed node")

    @property
    def type(self) -> Type:
        raise NotImplementedError("Invalid method on a collapsed node")

    @property
    def children(self) -> List['Node']:
        raise NotImplementedError("Invalid method on a collapsed node")

    def is_leaf(self) -> bool:
        return True

    def is_apply(self) -> bool:
        return False

    def is_enum(self) -> bool:
        return False

    def is_param(self) -> bool:
        return False

    def to_sexp(self):
        raise NotImplementedError("Invalid method on a collapsed node")

    def to_jsonexp(self):
        raise NotImplementedError("Invalid method on a collapsed node")

    def __repr__(self) -> str:
        return "CollapsedNode<{}>".format(self._node.__repr__())

    def __str__(self) -> str:
        return "Collapsed<{}>".format(self._node.__str__())


class AtomNode(LeafNode):
    '''Leaf AST node that holds string data'''

    def __init__(self, prod: Production, **kwargs):
        if not prod.is_enum():
            raise ValueError(
                'Cannot construct an AST atom node from a non-enum production')
        super().__init__(prod, **kwargs)

    def make_copy(self):
        '''
        Make a "deep" copy of the current node.
        Note that production and tag are still shared (not deep copied).
        '''
        return AtomNode(self._prod, tag=self._tag)

    @property
    def data(self) -> Any:
        prod = cast(EnumProduction, self._prod)
        return prod.rhs[0]

    @property
    def children(self) -> List[Node]:
        return []

    def is_enum(self) -> bool:
        return True

    def is_param(self) -> bool:
        return False

    def to_sexp(self):
        return [Symbol(self.type.name), self.data]

    def to_jsonexp(self):
        return ["enum", self.type.name, self.data]

    def deep_eq(self, other) -> bool:
        '''
        Test whether this node is the same with ``other``. This function performs deep comparison rather than just comparing the object identity.
        '''
        if isinstance(other, AtomNode):
            return self.type == other.type and self.data == other.data
        return False

    def deep_hash(self) -> int:
        '''
        This function performs deep hash rather than just hashing the object identity.
        '''
        return hash((self.type, str(self.data)))

    def __repr__(self) -> str:
        return 'AtomNode({})'.format(self.data)

    def __str__(self) -> str:
        return '{}'.format(self.data)


class ParamNode(LeafNode):
    '''Leaf AST node that holds a param'''

    def __init__(self, prod: Production, **kwargs):
        if not prod.is_param():
            raise ValueError(
                'Cannot construct an AST param node from a non-param production')
        super().__init__(prod, **kwargs)

    def make_copy(self):
        '''
        Make a "deep" copy of the current node.
        Note that production and tag are still shared (not deep copied).
        '''
        return ParamNode(self._prod, tag=self._tag)

    @property
    def index(self) -> int:
        prod = cast(ParamProduction, self._prod)
        return prod.rhs[0]

    @property
    def children(self) -> List[Node]:
        return []

    def is_enum(self) -> bool:
        return False

    def is_param(self) -> bool:
        return True

    def to_sexp(self):
        return [Symbol('@param'), self.index]

    def to_jsonexp(self):
        return ["param", self.index]

    def deep_eq(self, other) -> bool:
        '''
        Test whether this node is the same with ``other``. This function performs deep comparison rather than just comparing the object identity.
        '''
        if isinstance(other, ParamNode):
            return self.index == other.index
        return False

    def deep_hash(self) -> int:
        '''
        This function performs deep hash rather than just hashing the object identity.
        '''
        return hash(self.index)

    def __repr__(self) -> str:
        return 'ParamNode({})'.format(self.index)

    def __str__(self) -> str:
        return '@param{}'.format(self.index)


class ApplyNode(Node):
    '''Internal AST node that represent function application'''
    _args: List[Node]

    def __init__(self, prod: Production, args: List[Node], validation=True, **kwargs):
        super().__init__(prod, **kwargs)
        if validation:
            if not prod.is_function():
                raise ValueError(
                    'Cannot construct an AST internal node from a non-function production')
            if len(prod.rhs) != len(args):
                msg = 'Argument count mismatch: expected {} but found {}'.format(
                    len(prod.rhs), len(args))
                raise ValueError(msg)
            for index, (decl_ty, node) in enumerate(zip(prod.rhs, args)):
                actual_ty = node.type
                if decl_ty != actual_ty:
                    msg = 'Argument {} type mismatch: expected {} but found {}'.format(
                        index, decl_ty, actual_ty)
                    raise ValueError(msg)
        self._args = args

    def make_copy(self):
        '''
        Make a "deep" copy of the current node.
        Note that production and tag are still shared (not deep copied).
        '''
        new_args = [ p.make_copy() for p in self._args ]
        return ApplyNode(self._prod, new_args, tag=self._tag)

    def collapse(self, n) -> Node:
        '''
        Collapse the ApplyNode n times in a depth-first left-first way; replace the collapsed node with CollapsedNode.
        This will return a copy of a collapsed version of the node/program.
        '''
        dn = None
        if isinstance(n, int):
            # wrap into a mutable instance
            dn = Mutable(n)
        elif isinstance(n, Mutable):
            dn = n
        else:
            raise NotImplementedError("Number of collapsed nodes should be integer or mutable integer, got: {}".format(n))

        new_args = []
        for i in range(len(self._args)):
            if isinstance(self._args[i], ApplyNode):
                new_args.append( self._args[i].collapse(dn) )
            else:
                new_args.append( self._args[i].make_copy() )

        # note: turn off validation since there may be CollapsedNode inside
        new_node = ApplyNode(self._prod, new_args, validation=False, tag=self._tag)
        if dn.v>0:
            if dn.v == 1:
                # collapse self
                dn.v -= 1
                return CollapsedNode(new_node)
            else:
                # don't collapse, but record it
                dn.v -= 1
                return new_node
        else:
            # run out, don't collapse
            return new_node

    @property
    def name(self) -> str:
        prod = cast(FunctionProduction, self._prod)
        return prod.name

    @property
    def args(self) -> List[Node]:
        return self._args

    @property
    def children(self) -> List[Node]:
        return self._args

    def is_leaf(self) -> bool:
        return False

    def is_enum(self) -> bool:
        return False

    def is_param(self) -> bool:
        return False

    def is_apply(self) -> bool:
        return True

    def to_sexp(self):
        return [Symbol(self.name)] + [x.to_sexp() for x in self.args]

    def to_jsonexp(self):
        return [["function", self.name]] + [p.to_jsonexp() for p in self.args]

    def deep_eq(self, other) -> bool:
        '''
        Test whether this node is the same with ``other``. This function performs deep comparison rather than just comparing the object identity.
        '''
        if isinstance(other, ApplyNode):
            return self.name == other.name and \
                len(self.args) == len(other.args) and \
                all(x.deep_eq(y)
                    for x, y in zip(self.args, other.args))
        return False

    def deep_hash(self) -> int:
        '''
        This function performs deep hash rather than just hashing the object identity.
        '''
        return hash((self.name, tuple([x.deep_hash() for x in self.args])))

    def __repr__(self) -> str:
        return 'ApplyNode({}, {})'.format(self.name, self._args)

    def __str__(self) -> str:
        return '{}({})'.format(self.name, ', '.join([str(x) for x in self._args]))
