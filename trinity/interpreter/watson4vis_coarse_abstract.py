import pandas as pd
import numpy as np

from collections import namedtuple
from typing import Tuple, List, Iterator, Any

from ..dsl import Node, AtomNode, ParamNode, ApplyNode, CollapsedNode
from ..visitor import GenericVisitor
from ..spec.interval import *
from ..utils.table import make_abs, assemble_abstract_table, abs_intersected

from .interpreter import Interpreter
from .post_order import PostOrderInterpreter, CoarseAbstractInterpreter
from .error import InterpreterError, GeneralError

class NodeVisitor(GenericVisitor):
    _interp: CoarseAbstractInterpreter
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

# fixme: currently abstract interpretation rules are hard coded
#        need to dynamically load it from the spec in the next version
class Watson4VisCoarseAbstractInterpreter(CoarseAbstractInterpreter):
    '''
    A basic abstract interpreter that works on full skeleton level.
    '''

    def __init__(self, config=None, *args, **kwargs):
        super(Watson4VisCoarseAbstractInterpreter, self).__init__(*args, **kwargs)

        self._default_config = {
            "aggressive_mode": True
        }
        self._config = None
        if config is None:
            self._config = self._default_config
        else:
            self._config = config
        
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

    # fixme: forward to utils.table.make_abs
    def make_abs(self):
        return make_abs()

    # fixme: forward to utils.table.assemble_abstract_table
    def assemble_abstract_table(self, arg_tb0, arg_tb1):
        return assemble_abstract_table(arg_tb0, arg_tb1)

    # fixme: forward to utils.table.abs_intersected
    def abs_intersected(self, abs0, abs1):
        return abs_intersected(abs0, abs1)

    # hijack eval function: transform the inputs to abstract values before feeding
    def eval(self, prog: Node, inputs: List[Any], is_concrete=True) -> Any:
        if is_concrete:
            abstract_inputs = [
                self.assemble_abstract_table(inputs[0], p)
                for p in inputs
            ]
        else:
            abstract_inputs = inputs

        node_visitor = NodeVisitor(self, abstract_inputs)
        # note: sanity checking temporarily for new deduction, remove at some point
        self._last_cpos = -1
        try:
            # try if this node is a root node ("skeleton" field only exists in root node)
            if prog.tag is not None:
                if "skeleton" in prog.tag:
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

    def eval_ColInt(self, v):
        return None

    def eval_ColList(self, v):
        return None

    def eval_AggrFunc(self, v):
        return None

    def eval_NumFunc(self, v):
        return None

    def eval_BoolFunc(self, v):
        return None

    def eval_IndFunc(self, v):
        return None

    # ====================================== #
    # ======== function productions ======== #
    # ====================================== #

    def eval_SelectCol(self, node, args):
        arg_tb, arg_collist = args
        out = self.make_abs()
        out["row"] = interval_binary_op("==", out["row"], arg_tb["row"])
        out["col"] = interval_binary_op("<", out["col"], arg_tb["col"])
        out["head"] = interval_binary_op("<=", out["head"], arg_tb["head"])
        out["content"] = interval_binary_op("<=", out["content"], arg_tb["content"])
        return out

    # def eval_gather(self, node, args):
    #     arg_tb, arg_collist = args
    #     out = self.make_abs()
    #     out["row"] = interval_binary_op(">=", out["row"], arg_tb["row"])
    #     out["col"] = interval_binary_op("<=", out["col"], arg_tb["col"])
    #     out["head"] = interval_binary_op(
    #         "<=",
    #         out["head"],
    #         interval_binary_op("+", arg_tb["head"], Interval(2,2)),
    #     )
    #     out["content"] = interval_binary_op(
    #         "<=",
    #         out["content"],
    #         interval_binary_op("+", arg_tb["content"], Interval(2,2)),
    #     )
    #     return out

    def eval_Spread(self, node, args):
        arg_tb, arg_col0, arg_col1, arg_col2 = args
        out = self.make_abs()
        out["row"] = interval_binary_op("<=", out["row"], arg_tb["row"])
        out["col"] = interval_binary_op(">=", out["col"], arg_tb["col"])
        out["head"] = interval_binary_op("<=", out["head"], arg_tb["content"])
        out["content"] = interval_binary_op("<=", out["content"], arg_tb["content"])
        return out

    def eval_Mutate(self, node, args):
        arg_tb, arg_op, arg_col0, arg_col1 = args
        out = self.make_abs()
        out["row"] = interval_binary_op("==", out["row"], arg_tb["row"])
        out["col"] = interval_binary_op(
            "==",
            out["col"],
            interval_binary_op("+", arg_tb["col"], Interval(1,1)),
        )
        out["head"] = interval_binary_op(
            "==",
            out["head"],
            interval_binary_op("+", arg_tb["head"], Interval(1,1)),
        )
        out["content"] = interval_binary_op(">", out["content"], arg_tb["content"])
        out["content"] = interval_binary_op(
            "<=",
            out["content"],
            interval_binary_op("+", arg_tb["content"], arg_tb["row"]),
        )
        return out

    def eval_SelectRow0(self, node, args):
        arg_tb, arg_op, arg_col = args
        out = self.make_abs()
        out["row"] = interval_binary_op("<", out["row"], arg_tb["row"])
        out["col"] = interval_binary_op("==", out["col"], arg_tb["col"])
        out["head"] = interval_binary_op("==", out["head"], arg_tb["head"])
        out["content"] = interval_binary_op("<=", out["content"], arg_tb["content"])
        return out

    def eval_SelectRow1(self, node, args):
        arg_tb, arg_op, arg_col, arg_int = args
        out = self.make_abs()
        out["row"] = interval_binary_op("<", out["row"], arg_tb["row"])
        out["col"] = interval_binary_op("==", out["col"], arg_tb["col"])
        out["head"] = interval_binary_op("==", out["head"], arg_tb["head"])
        out["content"] = interval_binary_op("<=", out["content"], arg_tb["content"])
        return out

    def eval_GroupSum(self, node, args):
        arg_tb, arg_col0, arg_op, arg_col1 = args
        out = self.make_abs()
        out["row"] = interval_binary_op("<=", out["row"], arg_tb["row"])

        if self._config["aggressive_mode"]:
            # morpheus aggressive semantics
            out["col"] = interval_binary_op(
                "<=",
                out["col"],
                Interval(3,3),
            )
        else:
            out["col"] = interval_binary_op(
                "<=",
                out["col"],
                interval_binary_op("+", arg_tb["col"], Interval(1,1)),
            )

        out["head"] = interval_binary_op(">", out["head"], Interval(0,0))
        out["head"] = interval_binary_op(
            "<=",
            out["head"],
            interval_binary_op("+", arg_tb["head"], Interval(1,1)),
        )
        # note: originally it's: out.content <= in.content + in.group + 1
        #       since we don't track group, it becomes: out.content <= in.content + in.row + 1
        out["content"] = interval_binary_op(
            "<=",
            out["content"],
            interval_binary_op(
                "+",
                arg_tb["content"],
                interval_binary_op("+", arg_tb["row"], Interval(1,1)),
            ),
        )
        return out

    def eval_Summarize(self, node, args):
        arg_tb, arg_op, arg_col1 = args
        out = self.make_abs()
        out["row"] = interval_binary_op("==", out["row"], Interval(1,1))
        out["col"] = interval_binary_op("==", out["col"], Interval(1,1))
        out["head"] = interval_binary_op("==", out["head"], Interval(1,1))
        out["content"] = interval_binary_op("<=", out["content"], Interval(1,1))
        return out

    def eval_Contrast(self, node, args):
        arg_tb, node_op, node_kcol, node_const0, node_const1, node_vcol = args
        out = self.make_abs()
        out["row"] = interval_binary_op("==", out["row"], Interval(1,1))
        out["col"] = interval_binary_op("==", out["col"], Interval(1,1))
        out["head"] = interval_binary_op("==", out["head"], Interval(0,0))
        out["content"] = interval_binary_op("<=", out["content"], Interval(1,1))
        return out
       










