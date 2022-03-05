import xxhash
import pandas as pd
import numpy as np

from typing import Tuple, List, Iterator, Any

from .. import spec as S
from ..dsl import Node, AtomNode, ParamNode, ApplyNode
from ..visitor import GenericVisitor

from .interpreter import Interpreter
from .post_order import PostOrderInterpreter
from .error import InterpreterError, GeneralError, EqualityAssertion, ComponentError


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
        return (self._inputs[param_index], "input@{}".format(param_index))

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

class Watson4VisEvalInterpreter(PostOrderInterpreter):

    def __init__(self, spec: S.TrinitySpec, *args, **kwargs):
        super(Watson4VisEvalInterpreter, self).__init__(*args, **kwargs)

        self._spec = spec

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
        else:
            raise InterpreterError("Exception evaluating ConstVal.")

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

        return ret_collist

    def eval_SelectCol(self, node, args):
        (arg_tb, str_tb), node_collist = args
        arg_collist = self._eval_atom_node(node_collist)
        arg_nrow, arg_ncol = arg_tb.shape

        # explain collist after the previous assertion holds
        arg_collist = self.explain_collist(arg_ncol, arg_collist)

        try:
            tmp_cols = arg_tb.columns[arg_collist]
            ret_tb = arg_tb.loc[:, tmp_cols]
        except:
            if self._suppress_pandas_exception:
                raise Exception("You should not reach here.")
            else:
                raise

        # good to set context if you are here
        self._current_context += (node_collist.production.id,)

        return (ret_tb, ["SelectCol", str_tb, list(tmp_cols)])

    def eval_Spread(self, node, args):
        (arg_tb, str_tb), node_col0, node_col1, node_col2 = args
        arg_col0 = self._eval_atom_node(node_col0)
        arg_col1 = self._eval_atom_node(node_col1)
        arg_col2 = self._eval_atom_node(node_col2)
        arg_nrow, arg_ncol = arg_tb.shape

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
                raise Exception("You should not reach here.")
            else:
                raise

        # good to set context if you are here
        self._current_context += (node_col0.production.id, node_col1.production.id, node_col2.production.id)

        return (ret_tb, ["Spread", str_tb, tmp_col0, tmp_col1, tmp_col2])

    def eval_Mutate(self, node, args):
        (arg_tb, str_tb), node_op, node_col0, node_col1 = args
        arg_op = self._eval_atom_node(node_op)
        arg_col0 = self._eval_atom_node(node_col0)
        arg_col1 = self._eval_atom_node(node_col1)
        arg_nrow, arg_ncol = arg_tb.shape

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
        else:
            raise NotImplementedError("Unsupported NumFunc, got: {}.".format(arg_op))

        try:
            tmp_colname = self.fresh_colname()
            tmp_col0 = arg_tb.columns[arg_col0]
            tmp_col1 = arg_tb.columns[arg_col1]
            ret_tb = arg_tb.assign(**{tmp_colname:tmp_op(arg_tb[tmp_col0], arg_tb[tmp_col1])})
        except:
            if self._suppress_pandas_exception:
                raise Exception("You should not reach here.")
            else:
                raise

        # good to set context if you are here
        self._current_context += (node_op.production.id, node_col0.production.id, node_col1.production.id)

        return (ret_tb, ["Mutate", str_tb, arg_op, tmp_col0, tmp_col1])

    def eval_SelectRow0(self, node, args):
        (arg_tb, str_tb), node_op, node_col = args
        arg_op = self._eval_atom_node(node_op)
        arg_col = self._eval_atom_node(node_col)
        arg_nrow, arg_ncol = arg_tb.shape

        # note: putting this outside the `try` block to help debugging and keep coding style uniformed
        tmp_op = None
        if arg_op == "eqmin":
            tmp_op = lambda x,y: x==y
        elif arg_op == "eqmax":
            tmp_op = lambda x,y: x==y
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
                raise Exception("You should not reach here.")
            else:
                raise

        # good to set context if you are here
        self._current_context += (node_op.production.id, node_col.production.id)

        return (ret_tb, ["SelectRow0", str_tb, arg_op, tmp_col])

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
        else:
            raise NotImplementedError("Unsupported BoolFunc, got: {}.".format(arg_token))

    def eval_SelectRow1(self, node, args):
        (arg_tb, str_tb), node_op, node_col, node_int = args
        arg_op = self._eval_atom_node(node_op)
        arg_col = self._eval_atom_node(node_col)
        arg_int = self._eval_atom_node(node_int)
        arg_nrow, arg_ncol = arg_tb.shape

        tmp_op = self.instantiate_BoolFunc(arg_op)
        try:
            tmp_colname = self.fresh_colname()
            tmp_col = arg_tb.columns[arg_col]
            ret_tb = arg_tb[tmp_op(arg_tb[tmp_col], arg_int)]
        except:
            if self._suppress_pandas_exception:
                raise Exception("You should not reach here.")
            else:
                raise

        # good to set context if you are here
        self._current_context += (node_op.production.id, node_col.production.id, node_int.production.id)

        return (ret_tb, ["SelectRow1", str_tb, arg_op, tmp_col, arg_int])

    def eval_GroupSum(self, node, args):
        (arg_tb, str_tb), node_collist, node_op, node_col = args
        arg_collist = self._eval_atom_node(node_collist)
        arg_op = self._eval_atom_node(node_op)
        arg_col = self._eval_atom_node(node_col)
        arg_nrow, arg_ncol = arg_tb.shape

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
            else:
                raise NotImplementedError("Unsupported AggrFunc, got: {}".format(arg_op))
            ret_tb = self.flatten_index(ret_tb)
        except:
            if self._suppress_pandas_exception:
                raise Exception("You should not reach here.")
            else:
                raise

        # good to set context if you are here
        self._current_context += (node_collist.production.id, node_op.production.id, node_col.production.id)

        return (ret_tb, ["GroupSum", str_tb, list(tmp_collist), arg_op, tmp_col])

    def eval_Summarize(self, node, args):
        (arg_tb, str_tb), node_op, node_col = args
        arg_op = self._eval_atom_node(node_op)
        arg_col = self._eval_atom_node(node_col)
        arg_nrow, arg_ncol = arg_tb.shape

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
            else:
                raise NotImplementedError("Unsupported AggrFunc, got: {}".format(arg_op))
            ret_tb = self.flatten_index(ret_tb)
        except:
            if self._suppress_pandas_exception:
                raise Exception("You should not reach here.")
            else:
                raise

        # good to set context if you are here
        self._current_context += (node_op.production.id, node_col.production.id)

        return (ret_tb, ["Summarize", str_tb, arg_op, tmp_col])

    def eval_Contrast(self, node, args):
        (arg_tb, str_tb), node_op, node_kcol, node_const0, node_const1, node_vcol = args
        arg_op = self._eval_atom_node(node_op)
        arg_kcol = self._eval_atom_node(node_kcol)
        arg_const0 = self._eval_atom_node(node_const0)
        arg_const1 = self._eval_atom_node(node_const1)
        arg_vcol = self._eval_atom_node(node_vcol)
        arg_nrow, arg_ncol = arg_tb.shape

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
                raise Exception("You should not reach here.")
            else:
                raise

        # good to set context if you are here
        self._current_context += (node_op.production.id, node_kcol.production.id, node_const0.production.id, node_const1.production.id, node_vcol.production.id)

        return (ret_tb, ["Contrast", str_tb, arg_op, tmp_kcol, arg_const0, arg_const1, tmp_vcol])





