import time

from abc import ABC, abstractmethod
from typing import Any, List

from ..interpreter import InterpreterError
from ..enumerator import Enumerator
from ..decider import Decider
from ..dsl import Node
from ..logger import get_logger
from ..profiler import get_profiler

logger = get_logger("trinity.synthesizer.watson4vis_pause")
profiler = get_profiler("trinity.synthesizer.watson4vis_pause")

class Watson4VisPauseSynthesizer(ABC):

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
    def synthesize(self, timeout=None):
        '''
        A convenient method to enumerate ASTs until the result passes the analysis.
        Returns the synthesized program, or `None` if the synthesis failed.
        '''
        start_time = time.time()

        accepted_list = []
        fc, prog = self._enumerator.next()
        # print("# fc={}, prog={}".format(fc, prog))
        while prog is not None:
            if timeout is not None:
                if timeout<0:
                    # already timeout
                    break
                if time.time()-start_time>timeout:
                    break
            profiler.add1("concrete.programs")
            logger.debug('Proposed: {}.'.format(prog))
            # print("{}".format(str(prog.to_jsonexp()).replace("'",'"')))
            try:
                # decide whether there's skeleton level deduction
                self._decider.analyze(prog, fc=fc)
                # if you can make here without throwing any exceptions, you are good
                logger.info("Accepted: {}.".format(prog))
                accepted_list.append(prog)
                # input("FOUND ONE, CHECK")
                # return prog
                fc, prog = self._enumerator.next()
                continue
            except InterpreterError as e:
                logger.debug("Rejected: {}.".format(e))
                self._enumerator.update(e)
                fc, prog = self._enumerator.next()
        logger.info('Exhausted or Timeout.')
        # return None
        return accepted_list
