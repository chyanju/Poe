import pandas as pd
import numpy as np

from typing import Tuple, List, Iterator, Any

from .. import spec as S
from ..dsl import Node, AtomNode, ParamNode, ApplyNode, CollapsedNode
from ..visitor import GenericVisitor
from ..logger import get_logger
from ..profiler import get_profiler

from .interpreter import Interpreter
from .post_order import PostOrderInterpreter, PartialInterpreter, CoarseAbstractInterpreter
from .error import InterpreterError, GeneralError

logger = get_logger("trinity.watson4vis_partial_interpreter")
profiler = get_logger("trinity.watson4vis_partial_interpreter")

class NodeVisitor(GenericVisitor):
    '''
    Special visitor for partial evaluation.
    Note: We assume here that there's *ONLY ONE* CollapsedNode, so that the concrete interpreter state
          doesn't get update outside itself. Partial evaluation can prune a program with concrete prefix (condition)
          that corresponds to a CollapsedNode, which is always evaluated by a concrete interpreter.
    '''
    _partinterp: PartialInterpreter
    _interp: PostOrderInterpreter
    _absinterp: CoarseAbstractInterpreter
    _inputs: Any
    _abstract_inputs: Any

    def __init__(self, partinterp, interp, absinterp, inputs, abstract_inputs):
        self._partinterp = partinterp
        self._interp = interp
        self._absinterp = absinterp

        self._inputs = inputs
        self._abstract_inputs = abstract_inputs

        # create corresponding visitors
        self._interp_visitor = self._interp.visitor_class(self._interp, self._inputs)
        self._absinterp_visitor = self._absinterp.visitor_class(self._absinterp, self._abstract_inputs)

    def state_sync(self, from_interp, to_interp):
        to_interp._current_combination = tuple([p for p in from_interp._current_combination])
        to_interp._current_context = tuple([p for p in from_interp._current_context])
        to_interp._last_cpos = from_interp._last_cpos

    # ==================================== #
    # ======== visit calls series ======== #
    # ==================================== #
    # as per the spirit of partial evaluation, the default is_concrete should be False
    # except for collapsed nodes, because the top node should be abstract, and concretization
    # starts from bottom

    # handled by mixed interpreters
    def visit_collapsed_node(self, collapsed_node: CollapsedNode, is_concrete: bool=True):
        # concrete interpret the collapsed program and get the output
        tmp_result = None
        if isinstance(collapsed_node._node, ApplyNode):
            # inside of collapsed node must be concrete
            in_values = [self.visit(x, is_concrete=True) for x in collapsed_node._node.args]
            method_name = self._eval_method_name(collapsed_node._node.name)
            method = getattr(self._interp, method_name, self._method_not_found)
            tmp_result = method(collapsed_node._node, in_values)
        else:
            raise Exception("Unsupported innser type of CollapsedNode, got: {}.".format(type(collapsed_node._node)))
        # need to manually set the intermediate result
        collapsed_node._res = tmp_result
        # turn into abstract form and return
        tmp_abs_res = self._absinterp.assemble_abstract_table( self._inputs[0], tmp_result )
        # sync
        self.state_sync(self._interp, self._partinterp)
        self.state_sync(self._interp, self._absinterp)
        # print("# [DEBUG] visit_collapsed_node, comb: {}".format(self._partinterp._current_combination))
        return tmp_abs_res

    def visit_atom_node(self, atom_node: AtomNode, is_concrete: bool=True):
        if is_concrete:
            tmp_result = self._interp_visitor.visit_atom_node(atom_node)
            # sync
            self.state_sync(self._interp, self._partinterp)
            self.state_sync(self._interp, self._absinterp)
        else:
            tmp_result = self._absinterp_visitor.visit_atom_node(atom_node)
            # sync
            self.state_sync(self._absinterp, self._partinterp)
            # don't update self._interp, see class notes
            # self.state_sync(self._absinterp, self._interp)
        # print("# [DEBUG] (concrete?:{}) visit_atom_node, comb: {}".format(is_concrete, self._partinterp._current_combination))
        return tmp_result

    # handled by abstract interpreter
    def visit_param_node(self, param_node: ParamNode, is_concrete: bool=True):
        if is_concrete:
            tmp_result = self._interp_visitor.visit_param_node(param_node)
            # sync
            self.state_sync(self._interp, self._partinterp)
            self.state_sync(self._interp, self._absinterp)
        else:
            tmp_result = self._absinterp_visitor.visit_param_node(param_node)
            # sync
            self.state_sync(self._absinterp, self._partinterp)
            # don't update self._interp, see class notes
            # self.state_sync(self._absinterp, self._interp)
        # print("# [DEBUG] (concrete?:{}) visit_param_node, comb: {}".format(is_concrete, self._partinterp._current_combination))
        return tmp_result

    # handled by abstract interpreter
    def visit_apply_node(self, apply_node: ApplyNode, is_concrete: bool=True):
        # inherit is_concrete status for calling the children
        in_values = [self.visit(x, is_concrete=is_concrete) for x in apply_node.args]
        method_name = self._eval_method_name(apply_node.name)
        if is_concrete:
            method = getattr(self._interp, method_name, self._method_not_found)
            tmp_result = method(apply_node, in_values)
            # sync
            self.state_sync(self._interp, self._partinterp)
            self.state_sync(self._interp, self._absinterp)
        else:
            method = getattr(self._absinterp, method_name, self._method_not_found)
            tmp_result = method(apply_node, in_values)
            # sync
            self.state_sync(self._absinterp, self._partinterp)
            # don't update self._interp, see class notes
            # self.state_sync(self._absinterp, self._interp)
        # print("# [DEBUG] (concrete?:{}) visit_apply_node, comb: {}".format(is_concrete, self._partinterp._current_combination))
        return tmp_result

    def _method_not_found(self, apply_node: ApplyNode, arg_values: List[Any]):
        msg = 'Cannot find required eval method: "{}"'.format(
            self._eval_method_name(apply_node.name))
        raise NotImplementedError(msg)

    @staticmethod
    def _eval_method_name(name):
        return 'eval_' + name

class Watson4VisPartialInterpreter(PartialInterpreter):
    '''
    Note that the current partial interpreter only works with CoarseAbstractInterpreter.
    '''
    interpreter: PostOrderInterpreter
    abstract_interpreter: CoarseAbstractInterpreter

    def __init__(self, interpreter, abstract_interpreter, *args, **kwargs):
        super(Watson4VisPartialInterpreter, self).__init__(*args, **kwargs)
        self.interpreter = interpreter
        self.abstract_interpreter = abstract_interpreter
        
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
        # self.visitor_class = NodeVisitor
        # ======== shared infrastructure in line with LineSkeletonEnumerator ======== #

        # note: a sanity testing variable temporarily for the new deduction design
        #       need to make sure that the cpos values visited are increasing
        #       i.e., the order the atom node is composed is the same as the atom node is visited
        # note: sanity checking temporarily for new deduction, remove at some point
        self._last_cpos = -1

    # hijack the original eval method to perform partial evaluation
    def eval(self, prog: Node, inputs: List[Any], is_concrete=True) -> Any:
        if is_concrete:
            abstract_inputs = [
                self.abstract_interpreter.assemble_abstract_table(inputs[0], p)
                for p in inputs
            ]
        else:
            abstract_inputs = inputs

        node_visitor = NodeVisitor(self, self.interpreter, self.abstract_interpreter, inputs, abstract_inputs)
        try:
            # try if this node is a root node ("skeleton" field only exists in root node)
            tmp_tag = None
            if isinstance(prog, CollapsedNode):
                tmp_tag = prog._node.tag
            else:
                tmp_tag = prog.tag

            if tmp_tag is not None:
                if "skeleton" in tmp_tag:
                    # yes it's root
                    # then initialize set the _current_combination
                    self.interpreter._current_iter_ptr = tmp_tag["iter_ptr"]
                    self.interpreter._current_nslot = tmp_tag["nslot"]
                    self.interpreter._current_context = ()
                    self.interpreter._current_combination = ()
                    self.interpreter._last_cpos = -1

                    self.abstract_interpreter._current_iter_ptr = tmp_tag["iter_ptr"]
                    self.abstract_interpreter._current_nslot = tmp_tag["nslot"]
                    self.abstract_interpreter._current_context = ()
                    self.abstract_interpreter._current_combination = ()
                    self.abstract_interpreter._last_cpos = -1
            # top node should be abstract as per the spirits of partial evaluation
            return node_visitor.visit(prog, is_concrete=False)
        except InterpreterError as e:
            raise