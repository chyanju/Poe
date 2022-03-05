from abc import ABC, abstractmethod
from typing import Any, List

from ..interpreter import InterpreterError
from ..enumerator import Enumerator
from ..decider import Decider
from ..dsl import Node
from ..logger import get_logger
from ..profiler import get_profiler

logger = get_logger("trinity.synthesizer.watson4vis")
profiler = get_profiler("trinity.synthesizer.watson4vis")

class Watson4VisSynthesizer(ABC):

    _enumerator: Enumerator
    _decider: Decider

    def __init__(self, enumerator: Enumerator, decider: Decider):
        self._enumerator = enumerator
        self._decider = decider

    @property
    def enumerator(self):
        return self._enumerator

    @property
    def decider(self):
        return self._decider

    @profiler.ctimer("synthesize")
    def synthesize(self):
        '''
        A convenient method to enumerate ASTs until the result passes the analysis.
        Returns the synthesized program, or `None` if the synthesis failed.
        '''
        fc, prog = self._enumerator.next()
        # print("# fc={}, prog={}".format(fc, prog))
        while prog is not None:

            profiler.add1("concrete.programs")
            logger.debug('Proposed: {}.'.format(prog))
            # print("{}".format(str(prog.to_jsonexp()).replace("'",'"')))
            try:
                # decide whether there's skeleton level deduction
                self._decider.analyze(prog, fc=fc)
                # if you can make here without throwing any exceptions, you are good
                logger.info("Accepted: {}.".format(prog))
                return prog
            except InterpreterError as e:
                logger.debug("Rejected: {}.".format(e))
                self._enumerator.update(e)
                fc, prog = self._enumerator.next()
        logger.info('Exhausted.')
        return None
