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

class NaiveDecider(Decider):
    interpreter: PostOrderInterpreter

    def __init__(self,
                 interpreter: PostOrderInterpreter,
                 examples: List[Example],
                 equal_output: Callable[[Any, Any], bool] = lambda x, y: x == y,
                 recorder: Recorder=None):
        self.interpreter = interpreter
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
        return

    @profiler.ctimer("analyze.partial_evaluation")
    def analyze_partial_evaluation(self, prog):
        return