import xxhash
import pandas as pd
import numpy as np

from typing import Tuple, List, Iterator, Any

from .. import spec as S
from ..dsl import Node, AtomNode, ParamNode, ApplyNode
from ..visitor import GenericVisitor
from ..logger import get_logger
from ..profiler import get_profiler

from .interpreter import Interpreter
from .post_order import PostOrderInterpreter
from .error import InterpreterError, GeneralError, EqualityAssertion, ComponentError

logger = get_logger('trinity.interpreter.watson4vis')
profiler = get_profiler("trinity.interpreter.watson4vis")

class NodeVisitor(GenericVisitor):
    _interp: PostOrderInterpreter
    _inputs: Any

    def __init__(self, interp, inputs):
        self._interp = interp
        self._inputs = inputs

    # note: for atom node, to support parameter level conflict-driven learning, 
    #       use ??? to get the value, not the original eval_??? methods in Trinity
    #       and eventually return the node itself
    # note: in this version, every atom node is required to have tag and "cpos" field
    def visit_atom_node(self, atom_node: AtomNode):
        tmp_prod_id = atom_node.production.id
        # note: use self._interp to refer to the self in eval
        self._interp._current_combination += (tmp_prod_id,)
        # note: sanity checking temporarily for new deduction, remove at some point
        if atom_node.tag["cpos"] != self._interp._last_cpos+1:
            raise Exception("Panic: last cpos is: {}, current cpos is: {}".format(
                self._interp._last_cpos, atom_node.tag["cpos"]))
        self._interp._last_cpos = atom_node.tag["cpos"]
        return atom_node

    def visit_param_node(self, param_node: ParamNode):
        param_index = param_node.index
        if param_index >= len(self._inputs):
            msg = 'Input parameter access({}) out of bound({})'.format(
                param_index, len(self._inputs))
            raise GeneralError(msg)
        return self._inputs[param_index]

    def visit_apply_node(self, apply_node: ApplyNode):
        in_values = [self.visit(x) for x in apply_node.args]
        method_name = self._eval_method_name(apply_node.name)
        method = getattr(self._interp, method_name,
                         self._method_not_found)
        return method(apply_node, in_values)

    def _method_not_found(self, apply_node: ApplyNode, arg_values: List[Any]):
        msg = 'Cannot find required eval method: "{}"'.format(
            self._eval_method_name(apply_node.name))
        raise NotImplementedError(msg)

    @staticmethod
    def _eval_method_name(name):
        return 'eval_' + name

class Watson4VisInterpreter(PostOrderInterpreter):

    def __init__(self, spec: S.TrinitySpec, config=None, *args, **kwargs):
        super(Watson4VisInterpreter, self).__init__(*args, **kwargs)

        self._spec = spec
        self._default_config = {
            "aggressive_mode": True
        }
        self._config = None
        if config is None:
            self._config = self._default_config
        else:
            self._config = config

        # if you are debugging, turn this to False
        self._suppress_pandas_exception = True
        # self._suppress_pandas_exception = False

        self._colname_count = 0

        # fixme: this is used to compute the abstract values for content() and head()
        #        you will need to manually set it
        # self._example_input0 = None

        # ======== shared infrastructure in line with LineSkeletonEnumerator ======== #
        # note: a stateful variable that keeps track of interpretation combination
        #       which is connected to LineSkeletonEnumerator
        #       typical combination can be (1, 4, 1, 2, None, None)
        #       where None indicates anything and integers indicate production id
        #       LineSkeletonEnumerator will capture and utilize it to speed up enumeration
        self._current_iter_ptr = None
        self._current_nslot = None
        self._current_context = None
        self._current_combination = None
        # helper class for partial evaluation
        self.visitor_class = NodeVisitor
        # ======== shared infrastructure in line with LineSkeletonEnumerator ======== #

        # note: a sanity testing variable temporarily for the new deduction design
        #       need to make sure that the cpos values visited are increasing
        #       i.e., the order the atom node is composed is the same as the atom node is visited
        # note: sanity checking temporarily for new deduction, remove at some point
        self._last_cpos = -1

        # a numpy vectorize method to perform normalization for every cell value
        # self.normalize_cell_value = np.vectorize(lambda x: self._normalize_cell_value(x))
        # self.np_type_map = np.vectorize(lambda x: self._np_value_type(x))
        # self.np_hash_map = np.vectorize(lambda x: self._np_value_hash(x))
        self.np_mixed_hash_map = np.vectorize(
            lambda x: self._np_value_hash(x)*10 + self._np_value_type(x)
        )

        # given input and program, quickly identify output if computed before
        # use `hash_tb` to generate hash value for table, which is the first layer key
        # use str(prog) to generate hash value for program, which is the second layer key
        # fixme: currently this feature is not yet implemented
        self._value_caching = True # turn on value caching
        self._cached_outputs = {}

        # for hashing table (numpy) values
        self.strhash = lambda v: xxhash.xxh32(v, seed=4869).intdigest()
        self.inthash = hash
        self.tuphash = hash

        # caching for equality checking
        self._cached_eq_signatures = {}

    # def hash_tb(self, arg_tb):
    #     # helper for value caching
    #     return tuple(pd.util.hash_pandas_object(arg_tb).tolist())

    def flatten_index(self, arg_tb):
        # test whether the index is default or not
        # default index will be something like: FrozenList([None])
        arg_drop = all(p is None for p in arg_tb.index.names)
        d0 = arg_tb.reset_index(drop=arg_drop).to_dict()
        d1 = {}
        for k,v in d0.items():
            if isinstance(k, tuple):
                k0 = [p for p in k if len(p)>0][-1]
                d1[k0] = v
            else:
                d1[k] = v
        return pd.DataFrame.from_dict(d1)

    def fresh_colname(self):
        self._colname_count += 1
        return "COL{}".format(self._colname_count-1)

    # note: do NOT call me directly, call self.np_type_map
    def _np_value_type(self, v):
        if v is None:
            return 0
        elif isinstance(v, str):
            return 1
        elif isinstance(v, float):
            return 2
        elif isinstance(v, int):
            return 3
        elif np.issubdtype(type(v), np.inexact):
            return 4
        elif np.issubdtype(type(v), np.integer):
            return 5
        elif np.issubdtype(type(v), np.bool_):
            return 6
        else:
            raise NotImplementedError("Unsupported cell type for comparison, got: {}.".format(type(v)))

    # note: do NOT call me directly, call self.np_hash_map
    def _np_value_hash(self, v):
        if v is None:
            return 0
        elif isinstance(v, bool):
            return 1 if v else 0
        elif isinstance(v, str):
            return self.strhash(v)
        elif isinstance(v, float):
            return round(v, 4) * 10000
        elif isinstance(v, int):
            return v
        elif np.True_ == v:
            return 1
        elif np.False_ == v:
            return 0
        elif np.issubdtype(type(v), np.inexact):
            return round(v, 4) * 10000
        elif np.issubdtype(type(v), np.integer):
            return v
        else:
            raise NotImplementedError("Unsupported cell type for comparison, got: {}.".format(type(v)))

    def get_tb_signature(self, tb):
        # note: this is not consistent hashing (built-in Python hash)
        return self.tuphash(tuple(tb.values.flatten().tolist()))

    @profiler.ctimer("interpreter.equal_tb")
    def equal_tb(self, actual, expected):
        # logger.debug("actual: {}".format(actual))
        if actual.shape != expected.shape:
            raise EqualityAssertion(context=self._current_context, condition=lambda comb: False, tag="shape")

        eq_signature_actual = self.get_tb_signature(actual)
        eq_signature_expected = self.get_tb_signature(expected)

        if eq_signature_actual not in self._cached_eq_signatures.keys():
            # establish profile
            self._cached_eq_signatures[eq_signature_actual] = {}
        av_sigs = self._cached_eq_signatures[eq_signature_actual] # shorthand
        if eq_signature_expected not in self._cached_eq_signatures.keys():
            # establish profile
            self._cached_eq_signatures[eq_signature_expected] = {}
        ev_sigs = self._cached_eq_signatures[eq_signature_expected] # shorthand
        
        # normalize the array for comparison
        av = actual.values
        ev = expected.values
        try:
            av_mixed_map = None
            if "mixed_map" in av_sigs.keys():
                av_mixed_map = av_sigs["mixed_map"]
            else:
                av_mixed_map = self.np_mixed_hash_map(av)
                av_sigs["mixed_map"] = av_mixed_map
        except:
            raise EqualityAssertion(context=self._current_context, condition=lambda comb: False, tag="mixedmap")

        ev_mixed_map = None
        if "mixed_map" in ev_sigs.keys():
            ev_mixed_map = ev_sigs["mixed_map"]
        else:
            ev_mixed_map = self.np_mixed_hash_map(ev)
            ev_sigs["mixed_map"] = ev_mixed_map

        # ================================
        # column-wise signature comparison
        av_mixed_columns = None
        if "mixed_columns" in av_sigs.keys():
            av_mixed_columns = av_sigs["mixed_columns"]
        else:
            # then store every column as a set and compare
            av_mixed_columns = [ frozenset(av_mixed_map[:,i].tolist()) for i in range(av_mixed_map.shape[1]) ]
            # sort
            av_mixed_columns = sorted(av_mixed_columns, key=lambda x: hash(x), reverse=False)
            av_sigs["mixed_columns"] = av_mixed_columns

        ev_mixed_columns = None
        if "mixed_columns" in ev_sigs.keys():
            ev_mixed_columns = ev_sigs["mixed_columns"]
        else:
            # then store every column as a set and compare
            ev_mixed_columns = [ frozenset(ev_mixed_map[:,i].tolist()) for i in range(ev_mixed_map.shape[1]) ]
            # sort
            ev_mixed_columns = sorted(ev_mixed_columns, key=lambda x: hash(x), reverse=False)
            ev_sigs["mixed_columns"] = ev_mixed_columns
        # compare
        # logger.debug("av_mixed_columns: {}".format(av_mixed_columns))
        # logger.debug("ev_mixed_columns: {}".format(ev_mixed_columns))
        res_column = all([av_mixed_columns[i]==ev_mixed_columns[i] for i in range(len(av_mixed_columns))])
        if not res_column:
            raise EqualityAssertion(context=self._current_context, condition=lambda comb: False, tag="column")

        # =============================
        # row-wise signature comparison
        av_mixed_rows = None
        if "mixed_rows" in av_sigs.keys():
            av_mixed_rows = av_sigs["mixed_rows"]
        else:
            # then store every row as a set and compare
            av_mixed_rows = [ frozenset(av_mixed_map[i,:].tolist()) for i in range(av_mixed_map.shape[0]) ]
            # sort
            av_mixed_rows = sorted(av_mixed_rows, key=lambda x: hash(x), reverse=False)
            av_sigs["mixed_rows"] = av_mixed_rows

        ev_mixed_rows = None
        if "mixed_rows" in ev_sigs.keys():
            ev_mixed_rows = ev_sigs["mixed_rows"]
        else:
            # then store every row as a set and compare
            ev_mixed_rows = [ frozenset(ev_mixed_map[i,:].tolist()) for i in range(ev_mixed_map.shape[0]) ]
            # sort
            ev_mixed_rows = sorted(ev_mixed_rows, key=lambda x: hash(x), reverse=False)
            ev_sigs["mixed_rows"] = ev_mixed_rows
        # compare
        # logger.debug("av_mixed_rows: {}".format(av_mixed_rows))
        # logger.debug("ev_mixed_rows: {}".format(ev_mixed_rows))
        res_row = all([av_mixed_rows[i]==ev_mixed_rows[i] for i in range(len(av_mixed_rows))])
        if not res_row:
            raise EqualityAssertion(context=self._current_context, condition=lambda comb: False, tag="row")

        # if you are here, you are good to go
        return

    # hijack the original eval method to add detection of context nslot
    def eval(self, prog: Node, inputs: List[Any]) -> Any:
        '''
        Interpret the Given AST in post-order. Assumes the existence of `eval_XXX` method where `XXX` is the name of a function defined in the DSL.
        '''
        node_visitor = NodeVisitor(self, inputs)
        # note: sanity checking temporarily for new deduction, remove at some point
        self._last_cpos = -1
        try:
            # try if this node is a root node ("skeleton" field only exists in root node)
            if prog.tag is not None:
                if "skeleton" in prog.tag:
                    # print("DEBUG: {}".format(prog.tag))
                    # yes it's root
                    # then initialize set the _current_combination
                    self._current_iter_ptr = prog.tag["iter_ptr"]
                    self._current_nslot = prog.tag["nslot"]
                    self._current_context = ()
                    self._current_combination = ()
            return node_visitor.visit(prog)
        except InterpreterError as e:
            raise

    # ================================== #
    # ======== enum productions ======== #
    # ================================== #

    # fixme: merge with NodeVisitor later
    def _eval_method_name(self, name):
        return 'eval_' + name

    # main entrance of evaluating an atom node
    def _eval_atom_node(self, node):
        node_type = node.type.name
        method_name = self._eval_method_name(node_type)
        method = getattr(self, method_name)
        return method(node.data)

    # note: use this method in EnumAssertion
    def _eval_enum_prod(self, prod):
        prod_type = prod.lhs.name
        method_name = self._eval_method_name(prod_type)
        method = getattr(self, method_name)
        return method(prod.rhs[0])

    # can only be called by _eval_atom_node
    def eval_ColInt(self, v):
        return int(v)

    # can only be called by _eval_atom_node
    # def eval_SmallInt(self, v):
    #     return int(v)
    def eval_ConstVal(self, v):
        if v.endswith("@Float"):
            return float(v[:-6])
        elif v.endswith("@Int"):
            return int(v[:-4])
        elif v.endswith("@Str"):
            return v[:-4]
        elif v.endswith("@Set"):
            return eval(v[:-4])
        elif v=="<NULL>":
            # special placeholder
            # note-important: don't throw here, since enumerator's update method may also call this
            #                 and if it throws again in update function, no one catches this exception
            #                 instead, use the following check_ConstVal to process it
            return v
        else:
            raise InterpreterError("Exception evaluating ConstVal.")
    # helper function, should be called inside a component evaluation
    def check_ConstVal(self, v):
        if v=="<NULL>":
            raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="ConstVal:NULL")

    # can only be called by _eval_atom_node
    def eval_ColList(self, v):
        return [int(p) for p in v]

    # can only be called by _eval_atom_node
    def eval_AggrFunc(self, v):
        return v

    # can only be called by _eval_atom_node
    def eval_NumFunc(self, v):
        return v

    # can only be called by _eval_atom_node
    def eval_BoolFunc(self, v):
        return v

    # can only be called by _eval_atom_node
    def eval_IndFunc(self, v):
        return v

    # ====================================== #
    # ======== function productions ======== #
    # ====================================== #

    # interpret collist into column ints
    # note-important: based on validate_collist, the list should either be all positive or all negative
    #                 otherwise this function won't work as expected
    # fixme: maybe add an assertion?
    def explain_collist(self, arg_ncol, arg_collist):
        # print("# explain: arg_ncol={}, arg_collist={}".format(arg_ncol, arg_collist))

        ret_collist = list(range(arg_ncol))
        if arg_collist[0] >= 0:
            # positive list
            ret_collist = [p for p in arg_collist]
        else:
            # negative list
            for p in arg_collist:
                if p == -99:
                    ret_collist.remove(0)
                else:
                    ret_collist.remove(-p)

        # print("# YES!")
        return ret_collist

    # this validates that the collist is not stupid
    # note: this should be called with in assertEnum, and before you call the explain function
    def validate_collist(self, arg_ncol, arg_collist):
        # print("# validate: arg_ncol={}, arg_collist={}".format(arg_ncol, arg_collist))
        # arg_collist is the original collist (before explanation)

        if len(arg_collist) != len(list(set(arg_collist))):
            # don't include duplicates
            # e.g., -1, -1, will cause ValueError in .remove(x) in explain_collist
            return False

        # note-important: don't mix positive and negative ints
        if max(arg_collist) >= 0 and min(arg_collist) < 0:
            return False

        for p in arg_collist:
            if p == 0:
                if -99 in arg_collist:
                    return False
                elif 0 >= arg_ncol:
                    return False
                else:
                    continue
            elif p == -99:
                if 0 in arg_collist:
                    return False
                if 0 >= arg_ncol:
                    return False
                else:
                    continue
            elif p > 0:
                if p >= arg_ncol:
                    return False
                elif -p in arg_collist:
                    return False
                else:
                    continue
            elif p < 0:
                if -p >= arg_ncol:
                    return False
                elif -p in arg_collist:
                    return False
                else:
                    continue
        # print("# YES~")
        return True

    @profiler.ctimer("interpreter.eval_SelectCol")
    def eval_SelectCol(self, node, args):
        profiler.add1("interpreter.eval_SelectCol")
        arg_tb, node_collist = args
        arg_collist = self._eval_atom_node(node_collist)
        arg_nrow, arg_ncol = arg_tb.shape

        # value caching
        if self._value_caching:
            # compute signature
            # arg_sig = (str(node), self.hash_tb(arg_tb))
            arg_sig = (self._current_iter_ptr, self._current_combination)
            if arg_sig in self._cached_outputs:
                profiler.add1("cache_hit.SelectCol")
                if self._cached_outputs[arg_sig] is None:
                    # special token for ComponentError
                    raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="SelectCol:cached")
                else:
                    # good to set context if you are here
                    self._current_context += (node_collist.production.id,)
                    return self._cached_outputs[arg_sig]
        
        self.assertEnum(
            node=node, 
            context=self._current_context, 
            comb=self._current_combination,
            # this makes sure the original colist is not stupid
            cond=lambda comb: ( 
                lambda x: self.validate_collist(arg_ncol, x) 
            )(
                self._eval_enum_prod( self._spec.get_production( comb[node_collist.tag["cpos"]] ) )
            ),
            tag="SelectCol:0",
        )

        # explain collist after the previous assertion holds
        arg_collist = self.explain_collist(arg_ncol, arg_collist)

        try:
            tmp_cols = arg_tb.columns[arg_collist]
            ret_tb = arg_tb.loc[:, tmp_cols]
        except:
            if self._suppress_pandas_exception:
                if self._value_caching:
                    self._cached_outputs[arg_sig] = None
                raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="SelectCol")
            else:
                raise

        if self._value_caching:
            self._cached_outputs[arg_sig] = ret_tb

        # good to set context if you are here
        self._current_context += (node_collist.production.id,)

        return ret_tb

    # # info: benchmarks/test/4
    # @profiler.ctimer("interpreter.eval_gather")
    # def eval_gather(self, node, args):
    #     profiler.add1("interpreter.eval_gather")
    #     arg_tb, node_collist = args
    #     arg_collist = self._eval_atom_node(node_collist)
    #     arg_nrow, arg_ncol = arg_tb.shape

    #     # value caching
    #     if self._value_caching:
    #         # compute signature
    #         # arg_sig = (str(node), self.hash_tb(arg_tb))
    #         arg_sig = (self._current_iter_ptr, self._current_combination)
    #         if arg_sig in self._cached_outputs:
    #             profiler.add1("cache_hit.gather")
    #             if self._cached_outputs[arg_sig] is None:
    #                 # special token for ComponentError
    #                 raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="gather:cached")
    #             else:
    #                 # good to set context if you are here
    #                 self._current_context += (node_collist.production.id,)
    #                 return self._cached_outputs[arg_sig]

    #     _current_dtypes = arg_tb.dtypes.tolist()
    #     self.assertEnum(
    #         node=node, 
    #         context=self._current_context, 
    #         comb=self._current_combination,
    #         # x0: this makes sure the original colist is not stupid
    #         # self.explain_collist(x0): normal gather check
    #         # note-important: to ensure x0 comes before self.explain_collist(x0) checks, merge them into one assertEnum
    #         cond=lambda comb: ( 
    #             lambda x0: self.validate_collist(arg_ncol, x0) and \
    #                 len(set([_current_dtypes[p] for p in self.explain_collist(arg_ncol, x0)]))==1 
    #             )(
    #                 self._eval_enum_prod( self._spec.get_production( comb[node_collist.tag["cpos"]] ) )
    #         ),
    #         tag="gather:0",
    #     )

    #     # explain collist after the previous assertion holds
    #     arg_collist = self.explain_collist(arg_ncol, arg_collist)

    #     try:
    #         tmp_cols = arg_tb.columns[arg_collist]
    #         tmp_rcols = [p for p in arg_tb.columns if p not in tmp_cols]
    #         ret_tb = pd.melt(
    #             arg_tb, 
    #             id_vars=tmp_rcols, 
    #             value_vars=tmp_cols, 
    #             var_name=self.fresh_colname(), 
    #             value_name=self.fresh_colname(),
    #         )
    #         # print("# ret_tb: {}".format(ret_tb))
    #     except:
    #         if self._suppress_pandas_exception:
    #             if self._value_caching:
    #                 self._cached_outputs[arg_sig] = None
    #             raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="gather")
    #         else:
    #             raise

    #     if self._value_caching:
    #         self._cached_outputs[arg_sig] = ret_tb

    #     # good to set context if you are here
    #     self._current_context += (node_collist.production.id,)

    #     return ret_tb

    @profiler.ctimer("interpreter.eval_Spread")
    def eval_Spread(self, node, args):
        profiler.add1("interpreter.eval_Spread")
        arg_tb, node_col0, node_col1, node_col2 = args
        arg_col0 = self._eval_atom_node(node_col0)
        arg_col1 = self._eval_atom_node(node_col1)
        arg_col2 = self._eval_atom_node(node_col2)
        arg_nrow, arg_ncol = arg_tb.shape

        # logger.debug("# spread gets:\n{}".format(arg_tb))

        # value caching
        if self._value_caching:
            # compute signature
            # arg_sig = (str(node), self.hash_tb(arg_tb))
            arg_sig = (self._current_iter_ptr, self._current_combination)
            if arg_sig in self._cached_outputs:
                profiler.add1("cache_hit.Spread")
                if self._cached_outputs[arg_sig] is None:
                    # special token for ComponentError
                    raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="Spread:cached")
                else:
                    # good to set context if you are here
                    self._current_context += (node_col0.production.id, node_col1.production.id, node_col2.production.id)
                    return self._cached_outputs[arg_sig]

        _current_dtypes = arg_tb.dtypes.tolist()
        self.assertEnum(
            node=node, 
            context=self._current_context, 
            comb=self._current_combination,
            # note: nested lambda to store temp variable
            cond=lambda comb: (
                lambda x0, x1, x2: (x0 < arg_ncol and x1 < arg_ncol and x2 < arg_ncol and \
                                    x0 != x1 and x1 != x2 and x2 != x0 and \
                                    pd.api.types.is_string_dtype(_current_dtypes[x1])
                )
            )(
                self._eval_enum_prod( self._spec.get_production( comb[node_col0.tag["cpos"]] ) ),
                self._eval_enum_prod( self._spec.get_production( comb[node_col1.tag["cpos"]] ) ),
                self._eval_enum_prod( self._spec.get_production( comb[node_col2.tag["cpos"]] ) )
            ),
            tag="Spread:0",
        )

        try:
            tmp_col0 = arg_tb.columns[arg_col0]
            tmp_col1 = arg_tb.columns[arg_col1]
            tmp_col2 = arg_tb.columns[arg_col2]

            # the normal pivot way
            ret_tb = pd.pivot(arg_tb, index=[tmp_col0], columns=[tmp_col1], values=[tmp_col2])

            # flatten multiple indices
            ret_tb = self.flatten_index(ret_tb)
        except:
            if self._suppress_pandas_exception:
                if self._value_caching:
                    self._cached_outputs[arg_sig] = None
                raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="Spread")
            else:
                raise

        if self._value_caching:
            self._cached_outputs[arg_sig] = ret_tb

        # good to set context if you are here
        self._current_context += (node_col0.production.id, node_col1.production.id, node_col2.production.id)

        # logger.debug("# spread returns:\n{}".format(ret_tb))

        return ret_tb

    @profiler.ctimer("interpreter.eval_Mutate")
    def eval_Mutate(self, node, args):
        profiler.add1("interpreter.eval_Mutate")
        arg_tb, node_op, node_col0, node_col1 = args
        arg_op = self._eval_atom_node(node_op)
        arg_col0 = self._eval_atom_node(node_col0)
        arg_col1 = self._eval_atom_node(node_col1)
        arg_nrow, arg_ncol = arg_tb.shape

        # logger.debug("# mutate gets:\n{}".format(arg_tb))

        # value caching
        if self._value_caching:
            # compute signature
            # arg_sig = (str(node), self.hash_tb(arg_tb))
            arg_sig = (self._current_iter_ptr, self._current_combination)
            if arg_sig in self._cached_outputs:
                profiler.add1("cache_hit.Mutate")
                if self._cached_outputs[arg_sig] is None:
                    # special token for ComponentError
                    raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="Mutate:cached")
                else:
                    # good to set context if you are here
                    self._current_context += (node_op.production.id, node_col0.production.id, node_col1.production.id)
                    return self._cached_outputs[arg_sig]

        _current_dtypes = arg_tb.dtypes.tolist()
        self.assertEnum(
            node=node, 
            context=self._current_context, 
            comb=self._current_combination,
            # note: nested lambda to store temp variable
            cond=lambda comb: (
                lambda x0, x1: x0 < arg_ncol and \
                    x1 < arg_ncol and \
                    x0 != x1 and \
                    pd.api.types.is_numeric_dtype(_current_dtypes[x0]) and \
                    pd.api.types.is_numeric_dtype(_current_dtypes[x1])
            )(
                self._eval_enum_prod( self._spec.get_production( comb[node_col0.tag["cpos"]] ) ),
                self._eval_enum_prod( self._spec.get_production( comb[node_col1.tag["cpos"]] ) )
            ),
            tag="Mutate:0",
        )

        tmp_op = None
        if arg_op == "/":
            tmp_op = lambda x,y: x/y
        elif arg_op == "+":
            tmp_op = lambda x,y: x+y
        elif arg_op == "-":
            tmp_op = lambda x,y: x-y
        elif arg_op == "*":
            tmp_op = lambda x,y: x*y
        elif arg_op == "diff":
            tmp_op = lambda x,y: abs(x-y)
        elif arg_op == "<NULL>":
            raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="NumFunc:NULL")
        else:
            raise NotImplementedError("Unsupported NumFunc, got: {}.".format(arg_op))

        try:
            tmp_colname = self.fresh_colname()
            tmp_col0 = arg_tb.columns[arg_col0]
            tmp_col1 = arg_tb.columns[arg_col1]
            ret_tb = arg_tb.assign(**{tmp_colname:tmp_op(arg_tb[tmp_col0], arg_tb[tmp_col1])})
            # print("# ret_tb: {}".format(ret_tb))
        except:
            if self._suppress_pandas_exception:
                if self._value_caching:
                    self._cached_outputs[arg_sig] = None
                raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="Mutate")
            else:
                raise

        if self._value_caching:
            self._cached_outputs[arg_sig] = ret_tb

        # good to set context if you are here
        self._current_context += (node_op.production.id, node_col0.production.id, node_col1.production.id)

        return ret_tb

    @profiler.ctimer("interpreter.eval_SelectRow0")
    def eval_SelectRow0(self, node, args):
        profiler.add1("interpreter.eval_SelectRow0")
        arg_tb, node_op, node_col = args
        arg_op = self._eval_atom_node(node_op)
        arg_col = self._eval_atom_node(node_col)
        arg_nrow, arg_ncol = arg_tb.shape

        # value caching
        if self._value_caching:
            # compute signature
            # arg_sig = (str(node), self.hash_tb(arg_tb))
            arg_sig = (self._current_iter_ptr, self._current_combination)
            if arg_sig in self._cached_outputs:
                profiler.add1("cache_hit.SelectRow0")
                if self._cached_outputs[arg_sig] is None:
                    # special token for ComponentError
                    raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="SelectRow0:cached")
                else:
                    # good to set context if you are here
                    self._current_context += (node_op.production.id, node_col.production.id)
                    return self._cached_outputs[arg_sig]

        _current_dtypes = arg_tb.dtypes.tolist()
        self.assertEnum(
            node=node, 
            context=self._current_context, 
            comb=self._current_combination,
            # note: nested lambda to store temp variable
            cond=lambda comb: (
                lambda col, op: col<arg_ncol and \
                                (pd.api.types.is_numeric_dtype(_current_dtypes[col]) if op in {"eqmax","eqmin"} else True)
            ) (
                self._eval_enum_prod( self._spec.get_production( comb[node_col.tag["cpos"]] ) ),
                self._eval_enum_prod( self._spec.get_production( comb[node_op.tag["cpos"]] ) ),
            ),
            tag="SelectRow0:0",
        )

        # note: putting this outside the `try` block to help debugging and keep coding style uniformed
        tmp_op = None
        if arg_op == "eqmin":
            tmp_op = lambda x,y: x==y
        elif arg_op == "eqmax":
            tmp_op = lambda x,y: x==y
        elif arg_op == "<NULL>":
            raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="IndFunc:NULL")
        else:
            raise NotImplementedError("Unsupported IndFunc, got: {}.".format(arg_op))

        try:
            tmp_colname = self.fresh_colname()
            tmp_col = arg_tb.columns[arg_col]
            if arg_op == "eqmin":
                ret_tb = arg_tb[tmp_op(arg_tb[tmp_col], min(arg_tb[tmp_col]))]
            elif arg_op == "eqmax":
                ret_tb = arg_tb[tmp_op(arg_tb[tmp_col], max(arg_tb[tmp_col]))]
            else:
                raise NotImplementedError("You should not reach here.")
        except:
            if self._suppress_pandas_exception:
                if self._value_caching:
                    self._cached_outputs[arg_sig] = None
                raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="SelectRow0")
            else:
                raise

        if self._value_caching:
            self._cached_outputs[arg_sig] = ret_tb

        # good to set context if you are here
        self._current_context += (node_op.production.id, node_col.production.id)

        return ret_tb

    def instantiate_BoolFunc(self, arg_token):
        if arg_token == "==":
            return lambda x,y: x==y
        elif arg_token == ">":
            return lambda x,y: x>y
        elif arg_token == "<":
            return lambda x,y: x<y
        elif arg_token == "!=":
            return lambda x,y: x!=y
        elif arg_token == ">=":
            return lambda x,y: x>=y
        elif arg_token == "<=":
            return lambda x,y: x<=y
        elif arg_token == "setin":
            return lambda x,y: x.isin(y)
        elif arg_token == "<NULL>":
            raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="BoolFunc:NULL")
        else:
            raise NotImplementedError("Unsupported BoolFunc, got: {}.".format(arg_token))

    @profiler.ctimer("interpreter.eval_SelectRow1")
    def eval_SelectRow1(self, node, args):
        profiler.add1("interpreter.eval_SelectRow1")
        arg_tb, node_op, node_col, node_int = args
        arg_op = self._eval_atom_node(node_op)
        arg_col = self._eval_atom_node(node_col)
        arg_int = self._eval_atom_node(node_int)
        arg_nrow, arg_ncol = arg_tb.shape

        # check for <NULL>
        self.check_ConstVal(arg_int)

        # value caching
        if self._value_caching:
            # compute signature
            # arg_sig = (str(node), self.hash_tb(arg_tb))
            arg_sig = (self._current_iter_ptr, self._current_combination)
            if arg_sig in self._cached_outputs:
                profiler.add1("cache_hit.SelectRow1")
                if self._cached_outputs[arg_sig] is None:
                    # special token for ComponentError
                    raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="SelectRow1:cached")
                else:
                    # good to set context if you are here
                    self._current_context += (node_op.production.id, node_col.production.id, node_int.production.id)
                    return self._cached_outputs[arg_sig]

        _current_dtypes = arg_tb.dtypes.tolist()
        self.assertEnum(
            node=node, 
            context=self._current_context, 
            comb=self._current_combination,
            # note: nested lambda to store temp variable
            cond=lambda comb: (
                lambda col, op, y: col<arg_ncol and \
                                   (isinstance(y, set) if op=="isin" else True) and \
                                   (pd.api.types.is_numeric_dtype(_current_dtypes[col]) if op in {"<","<=",">",">="} else True) and \
                                   ( (isinstance(y, float) or isinstance(y, int)) if op in {"<","<=",">",">="} else True)
            ) (
                self._eval_enum_prod( self._spec.get_production( comb[node_col.tag["cpos"]] ) ),
                self._eval_enum_prod( self._spec.get_production( comb[node_op.tag["cpos"]] ) ),
                self._eval_enum_prod( self._spec.get_production( comb[node_int.tag["cpos"]] ) ),
            ),
            tag="SelectRow1:0",
        )

        tmp_op = self.instantiate_BoolFunc(arg_op)
        try:
            tmp_colname = self.fresh_colname()
            tmp_col = arg_tb.columns[arg_col]
            ret_tb = arg_tb[tmp_op(arg_tb[tmp_col], arg_int)]
        except:
            if self._suppress_pandas_exception:
                if self._value_caching:
                    self._cached_outputs[arg_sig] = None
                raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="SelectRow1")
            else:
                raise

        if self._value_caching:
            self._cached_outputs[arg_sig] = ret_tb

        # good to set context if you are here
        self._current_context += (node_op.production.id, node_col.production.id, node_int.production.id)

        return ret_tb

    @profiler.ctimer("interpreter.eval_GroupSum")
    def eval_GroupSum(self, node, args):
        profiler.add1("interpreter.eval_GroupSum")
        arg_tb, node_collist, node_op, node_col = args
        arg_collist = self._eval_atom_node(node_collist)
        arg_op = self._eval_atom_node(node_op)
        arg_col = self._eval_atom_node(node_col)
        arg_nrow, arg_ncol = arg_tb.shape

        # value caching
        if self._value_caching:
            # compute signature
            # arg_sig = (str(node), self.hash_tb(arg_tb))
            arg_sig = (self._current_iter_ptr, self._current_combination)
            if arg_sig in self._cached_outputs:
                profiler.add1("cache_hit.GroupSum")
                if self._cached_outputs[arg_sig] is None:
                    # special token for ComponentError
                    raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="GroupSum:cached")
                else:
                    # good to set context if you are here
                    self._current_context += (node_collist.production.id, node_op.production.id, node_col.production.id)
                    return self._cached_outputs[arg_sig]

        _current_dtypes = arg_tb.dtypes.tolist()
        self.assertEnum(
            node=node, 
            context=self._current_context, 
            comb=self._current_combination,
            # this makes sure the original colist is not stupid
            # note-important: to ensure x0 comes before self.explain_collist(x0)/y checks, merge them into one assertEnum
            cond=lambda comb: (
                # morpheus aggressive semantics
                lambda x0, op, y: self.validate_collist(arg_ncol, x0) and \
                    y < arg_ncol and \
                    # only count can deal with string, others should only be dealing with numbers
                    (pd.api.types.is_numeric_dtype(_current_dtypes[y]) if op in {"min","max","sum","mean"} else True) and \
                    all(map(lambda z:z>=0, x0)) \
                if self._config["aggressive_mode"] else \
                lambda x0, op, y: self.validate_collist(arg_ncol, x0) and \
                    (pd.api.types.is_numeric_dtype(_current_dtypes[y]) if op in {"min","max","sum","mean"} else True) and \
                    y < arg_ncol
            )(
                self._eval_enum_prod( self._spec.get_production( comb[node_collist.tag["cpos"]] ) ),
                self._eval_enum_prod( self._spec.get_production( comb[node_op.tag["cpos"]] ) ),
                self._eval_enum_prod( self._spec.get_production( comb[node_col.tag["cpos"]] ) ),
            ),
            tag="GroupSum:0",
        )

        # explain collist after the previous assertion holds
        arg_collist = self.explain_collist(arg_ncol, arg_collist)

        try:
            tmp_colname = self.fresh_colname()
            tmp_collist = [arg_tb.columns[p] for p in arg_collist]
            tmp_col = arg_tb.columns[arg_col]
            if arg_op == "min":
                ret_tb = arg_tb.groupby(by=tmp_collist)[tmp_col].min().to_frame(tmp_colname)
            elif arg_op == "max":
                ret_tb = arg_tb.groupby(by=tmp_collist)[tmp_col].max().to_frame(tmp_colname)
            elif arg_op == "sum":
                ret_tb = arg_tb.groupby(by=tmp_collist)[tmp_col].sum().to_frame(tmp_colname)
            elif arg_op == "mean":
                ret_tb = arg_tb.groupby(by=tmp_collist)[tmp_col].mean().to_frame(tmp_colname)
            elif arg_op == "count":
                ret_tb = arg_tb.groupby(by=tmp_collist)[tmp_col].count().to_frame(tmp_colname)
            elif arg_op == "<NUlL>":
                raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="AggrFunc:NULL")
            else:
                raise NotImplementedError("Unsupported AggrFunc, got: {}".format(arg_op))
            ret_tb = self.flatten_index(ret_tb)
            # don't rename, it's already done in the to_frame call above
            # ref: https://stackoverflow.com/questions/48770035/adding-a-count-column-to-the-result-of-a-groupby-in-pandas
            # ret_tb = ret_tb.rename(columns={tmp_col:tmp_colname})
            # print("# ret_tb: {}".format(ret_tb))
        except:
            if self._suppress_pandas_exception:
                if self._value_caching:
                    self._cached_outputs[arg_sig] = None
                raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="GroupSum")
            else:
                raise

        if self._value_caching:
            self._cached_outputs[arg_sig] = ret_tb

        # good to set context if you are here
        self._current_context += (node_collist.production.id, node_op.production.id, node_col.production.id)

        return ret_tb

    @profiler.ctimer("interpreter.eval_Summarize")
    def eval_Summarize(self, node, args):
        profiler.add1("interpreter.eval_Summarize")
        arg_tb, node_op, node_col = args
        arg_op = self._eval_atom_node(node_op)
        arg_col = self._eval_atom_node(node_col)
        arg_nrow, arg_ncol = arg_tb.shape

        # value caching
        if self._value_caching:
            # compute signature
            # arg_sig = (str(node), self.hash_tb(arg_tb))
            arg_sig = (self._current_iter_ptr, self._current_combination)
            if arg_sig in self._cached_outputs:
                profiler.add1("cache_hit.Summarize")
                if self._cached_outputs[arg_sig] is None:
                    # special token for ComponentError
                    raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="Summarize:cached")
                else:
                    # good to set context if you are here
                    self._current_context += (node_op.production.id, node_col.production.id)
                    return self._cached_outputs[arg_sig]

        _current_dtypes = arg_tb.dtypes.tolist()
        self.assertEnum(
            node=node, 
            context=self._current_context, 
            comb=self._current_combination,
            # this makes sure the original colist is not stupid
            # note-important: to ensure x0 comes before self.explain_collist(x0)/y checks, merge them into one assertEnum
            cond=lambda comb: (
                # morpheus aggressive semantics
                lambda op, y: y < arg_ncol and \
                              (pd.api.types.is_numeric_dtype(_current_dtypes[y]) if op in {"min","max","sum","mean"} else True)
            )(
                self._eval_enum_prod( self._spec.get_production( comb[node_op.tag["cpos"]] ) ),
                self._eval_enum_prod( self._spec.get_production( comb[node_col.tag["cpos"]] ) ),
            ),
            tag="Summarize:0",
        )

        try:
            tmp_colname = self.fresh_colname()
            tmp_col = arg_tb.columns[arg_col]
            # fixme: implement a* series
            if arg_op == "min":
                ret_tb = pd.DataFrame.from_records([[arg_tb[tmp_col].min()]], columns=[tmp_colname])
            elif arg_op == "max":
                ret_tb = pd.DataFrame.from_records([[arg_tb[tmp_col].max()]], columns=[tmp_colname])
            elif arg_op == "sum":
                ret_tb = pd.DataFrame.from_records([[arg_tb[tmp_col].sum()]], columns=[tmp_colname])
            elif arg_op == "mean":
                ret_tb = pd.DataFrame.from_records([[arg_tb[tmp_col].mean()]], columns=[tmp_colname])
            elif arg_op == "count":
                ret_tb = pd.DataFrame.from_records([[arg_tb[tmp_col].count()]], columns=[tmp_colname])
            elif arg_op == "<NULL>":
                raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="AggrFunc:NULL")
            else:
                raise NotImplementedError("Unsupported AggrFunc, got: {}".format(arg_op))
            ret_tb = self.flatten_index(ret_tb)
            # don't rename, it's already done in the to_frame call above
            # ref: https://stackoverflow.com/questions/48770035/adding-a-count-column-to-the-result-of-a-groupby-in-pandas
            # ret_tb = ret_tb.rename(columns={tmp_col:tmp_colname})
            # print("# ret_tb: {}".format(ret_tb))
        except:
            if self._suppress_pandas_exception:
                if self._value_caching:
                    self._cached_outputs[arg_sig] = None
                raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="Summarize")
            else:
                raise

        if self._value_caching:
            self._cached_outputs[arg_sig] = ret_tb

        # good to set context if you are here
        self._current_context += (node_op.production.id, node_col.production.id)

        return ret_tb

    @profiler.ctimer("interpreter.eval_Contrast")
    def eval_Contrast(self, node, args):
        profiler.add1("interpreter.eval_Contrast")
        arg_tb, node_op, node_kcol, node_const0, node_const1, node_vcol = args
        arg_op = self._eval_atom_node(node_op)
        arg_kcol = self._eval_atom_node(node_kcol)
        arg_const0 = self._eval_atom_node(node_const0)
        arg_const1 = self._eval_atom_node(node_const1)
        arg_vcol = self._eval_atom_node(node_vcol)
        arg_nrow, arg_ncol = arg_tb.shape

        # check for <NULL>
        self.check_ConstVal(arg_const0)
        self.check_ConstVal(arg_const1)

        # value caching
        if self._value_caching:
            # compute signature
            # arg_sig = (str(node), self.hash_tb(arg_tb))
            arg_sig = (self._current_iter_ptr, self._current_combination)
            if arg_sig in self._cached_outputs:
                profiler.add1("cache_hit.Contrast")
                if self._cached_outputs[arg_sig] is None:
                    # special token for ComponentError
                    raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="Contrast:cached")
                else:
                    # good to set context if you are here
                    self._current_context += (node_op.production.id, node_kcol.production.id, node_const0.production.id, node_const1.production.id, node_vcol.production.id)
                    return self._cached_outputs[arg_sig]

        _current_dtypes = arg_tb.dtypes.tolist()
        self.assertEnum(
            node=node, 
            context=self._current_context, 
            comb=self._current_combination,
            # this makes sure the original colist is not stupid
            # note-important: to ensure x0 comes before self.explain_collist(x0)/y checks, merge them into one assertEnum
            cond=lambda comb: (
                # morpheus aggressive semantics
                lambda op, kc, vc: kc<arg_ncol and vc<arg_ncol and kc!=vc and \
                                   pd.api.types.is_numeric_dtype(_current_dtypes[vc])
            )(
                self._eval_enum_prod( self._spec.get_production( comb[node_op.tag["cpos"]] ) ),
                self._eval_enum_prod( self._spec.get_production( comb[node_kcol.tag["cpos"]] ) ),
                self._eval_enum_prod( self._spec.get_production( comb[node_vcol.tag["cpos"]] ) ),
            ),
            tag="Contrast:0",
        )

        tmp_op = None
        if arg_op == "/":
            tmp_op = lambda x,y: x/y
        elif arg_op == "+":
            tmp_op = lambda x,y: x+y
        elif arg_op == "-":
            tmp_op = lambda x,y: x-y
        elif arg_op == "*":
            tmp_op = lambda x,y: x*y
        elif arg_op == "diff":
            tmp_op = lambda x,y: abs(x-y)
        elif arg_op == "<NULL>":
            raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="NumFunc:NULL")
        else:
            raise NotImplementedError("Unsupported NumFunc, got: {}.".format(arg_op))

        try:
            # first filter by two consts and we'll have two tables
            tmp_kcol = arg_tb.columns[arg_kcol]

            tmp_tb0 = arg_tb[arg_tb[tmp_kcol]==arg_const0]
            # assert tmp_tb0.shape[0]>0 # at least one row should be there
            assert tmp_tb0.shape[0]==1 # stricter

            tmp_tb1 = arg_tb[arg_tb[tmp_kcol]==arg_const1]
            # assert tmp_tb1.shape[0]>0 # at least one row should be there
            assert tmp_tb1.shape[0]==1 # stricter

            tmp_vcol = arg_tb.columns[arg_vcol]
            ret_tb = pd.DataFrame.from_records(
                [[ tmp_op( tmp_tb0.iloc[0,arg_vcol], tmp_tb1.iloc[0,arg_vcol] ) ]], columns=[tmp_vcol]
            )
        except:
            if self._suppress_pandas_exception:
                if self._value_caching:
                    self._cached_outputs[arg_sig] = None
                raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="Contrast")
            else:
                raise

        if self._value_caching:
            self._cached_outputs[arg_sig] = ret_tb

        # good to set context if you are here
        self._current_context += (node_op.production.id, node_kcol.production.id, node_const0.production.id, node_const1.production.id, node_vcol.production.id)

        return ret_tb




