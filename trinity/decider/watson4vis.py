from typing import Callable, NamedTuple, List, Any

from ..interpreter import Interpreter, PostOrderInterpreter, InterpreterError, EnumAssertion, SkeletonAssertion, CoarseAbstractInterpreter, FineAbstractInterpreter
from ..profiler import get_profiler
from ..recorder import Recorder

from .decider import Decider
from .example_base import Example

profiler = get_profiler("trinity.decider.watson4vis")
# =========================================== #
# ======== deduction recording rules ======== #
# =========================================== #
# general format: 
#   - if unsat: (<error_str>, prog)
#     - unsat results are recorded in synthesizer
#   - if sat: ("FullSkeleton"/"PartialEvaluation"/"Equality", prog)
#     - sat results are recorded in decider
# examples:
#   - unsat:
#     - ("EnumAssertion:partial:step1.", prog)
#     - ("EqualityAssertion:shape", prog)
#   - sat:
#     - ("FullSkeleton", prog)
#     - ("Equality", prog)

class Watson4VisDecider(Decider):
    interpreter: PostOrderInterpreter
    coarse_abstract_interpreter: CoarseAbstractInterpreter # throws SkeletonAssertion
    # fine_abstract_interpreter: FineAbstractInterpreter # throws EnumAssertion
    partial_interpreter: PostOrderInterpreter
    examples: List[Example]
    equal_output: Callable[[Any, Any], bool]

    def __init__(self,
                 interpreter: PostOrderInterpreter,
                 coarse_abstract_interpreter: CoarseAbstractInterpreter,
                 # fine_abstract_interpreter: FineAbstractInterpreter,
                 partial_interpreter: PostOrderInterpreter,
                 examples: List[Example],
                 equal_output: Callable[[Any, Any], bool] = lambda x, y: x == y,
                 recorder: Recorder=None):
        self.interpreter = interpreter
        self.coarse_abstract_interpreter = coarse_abstract_interpreter
        # self.fine_abstract_interpreter = fine_abstract_interpreter
        self.partial_interpreter = partial_interpreter
        if len(examples) == 0:
            raise ValueError(
                'ExampleDecider cannot take an empty list of examples')
        self.examples = examples
        self.equal_output = equal_output
        self.recorder = recorder

    # note: ref https://www.toptal.com/python/top-10-mistakes-that-python-programmers-make
    def __del__(self):
        if self.recorder:
            self.recorder.serialize()

    @profiler.ctimer("analyze.equality")
    def analyze_equality(self, prog):
        for x in self.examples:
            profiler.add1("analyze.equality")
            self.equal_output( self.interpreter.eval(prog, x.input), x.output )

        return

    @profiler.ctimer("analyze.total")
    def analyze(self, prog, **kwargs):
        try:
            if "fc" in kwargs and kwargs["fc"]:
                self.analyze_full_skeleton(prog)
                if self.recorder:
                    self.recorder.record(("FullSkeleton", str(prog)))

            # perform partial evaluation
            self.analyze_partial_evaluation(prog)
            if self.recorder:
                self.recorder.record(("PartialEvaluation", str(prog)))

            # perform concrete evaluation
            self.analyze_equality(prog)
            if self.recorder:
                self.recorder.record(("Equality", str(prog)))

            # if you make here, you are good to go
            return
        except InterpreterError as e:
            if self.recorder:
                self.recorder.record((str(e), str(prog)))
            raise

    @profiler.ctimer("analyze.full_skeleton")
    def analyze_full_skeleton(self, prog):
        '''
        This calls abstract interpretation to determine whether the current skeleton is feasible.
        Note that this method doesn't have CDCL; it only generalizes to itself (one skeleton, the current one).
        '''
        for ex in self.examples:
            profiler.add1("analyze.full_skeleton")
            res_expected = self.coarse_abstract_interpreter.assemble_abstract_table(ex.input[0], ex.output)
            res_actual = self.coarse_abstract_interpreter.eval(prog, ex.input)
            result = self.coarse_abstract_interpreter.abs_intersected(res_expected, res_actual)
            if not result:
                # infeasible, throw InterpreterError
                # fixme: currently the whole speculative output is to blame (guard)
                # fixme: currently the whole program is added to KB, ideally only partial programs are to add
                raise SkeletonAssertion(example=ex, prog=prog, tag=prog._tag["skeleton"])
        # if you reach here, you are good to go
        return

    @profiler.ctimer("analyze.partial_evaluation")
    def analyze_partial_evaluation(self, prog):
        '''
        This is partial evaluation.
        '''
        # in principle, there can only be at most one CollapsedNode in a program
        # first count the number of CollapsedNode
        ac = prog.__repr__().count("ApplyNode")
        for ex in self.examples:
            for i in range(1, ac):
                profiler.add1("analyze.partial_evaluation")
                # if i==0, then it's sketch level deduction
                # if i==ac, then it's concrete evaluation
                # so here for partial evaluation, we only do [1, ac-1]
                iprog = prog.collapse(i)
                # print("# [DEBUG] pe, i={}, prog={}".format(i, iprog))
                res_expected = self.partial_interpreter.abstract_interpreter.assemble_abstract_table(ex.input[0], ex.output)
                res_actual = self.partial_interpreter.eval(iprog, ex.input)
                result = self.partial_interpreter.abstract_interpreter.abs_intersected(res_expected, res_actual)
                if not result:
                    # print("# [DEBUG] expected: {}".format(res_expected))
                    # print("# [DEBUG] actual: {}".format(res_actual))
                    # here what we raise is EnumAssertion
                    raise EnumAssertion(
                        context=self.partial_interpreter.interpreter._current_combination,
                        extlen=len(self.partial_interpreter.interpreter._current_combination),
                        condition=lambda comb:False,
                        tag="partial:step{}".format(i),
                    )
        pass