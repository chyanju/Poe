import math
import copy

from typing import Set, Optional, List, Any, Iterator, Tuple
from random import Random
from itertools import product
from functools import reduce

from .. import dsl as D
from .. import spec as S
from ..interpreter import EnumAssertion, SkeletonAssertion, ComponentError, EqualityAssertion
from ..spec.production import Production, FunctionProduction, ParamProduction, EnumProduction
from ..spec import TrinitySpec, Type, EnumType, ValueType
from ..dsl import Node, Builder
from ..logger import get_logger
from ..profiler import get_profiler

from .enumerator import Enumerator

logger = get_logger("trinity.enumerator.line_skeleton")
profiler = get_profiler("trinity.enumerator.line_skeleton")

class Pool():
    '''
    A special class to store a set for combinations/enumerations.
    https://stackoverflow.com/questions/1132941/least-astonishment-and-the-mutable-default-argument
    '''
    def __init__(self, pool):
        self._pool = pool

    def __len__(self):
        return len(self._pool)

    def __getitem__(self, i):
        return self._pool[i]

    def append(self, p):
        self._pool.append(p)

    def __iadd__(self, other):
        if isinstance(other, Pool):
            self._pool += other._pool
        elif isinstance(other, list):
            self._pool += other
        else:
            raise NotImplementedError("Pool += operator doesn't support type {}.".format(type(other)))
        return self

class BlockTracker():
    def __init__(self, choices):
        for p in choices:
            assert len(p) == len(set(p)), "Duplicate choices in one pool detected."
        self._vchoices = [set(p) for p in choices] # for validation, you need set to make it faster
        self._cchoices = choices # for choice, you need the original order
        self._rchoices = [
            {choices[i][j]:j for j in range(len(choices[i]))}
            for i in range(len(choices))
        ] # reverse indexing, given a pcomb, you can easily know its index comb
        self._caps = [len(choices[i]) for i in range(len(choices))]
        self._tracker = {}

    def make_copy(self):
        tmp_tracker = BlockTracker(copy.deepcopy(self._cchoices))
        tmp_tracker._tracker = copy.deepcopy(self._tracker)
        return tmp_tracker

    def validate_pcomb(self, pcomb):
        assert len(pcomb) <= len(self._caps),\
            "Length of partial combination is too long, expected: <={}, got: {}.".format(len(self._caps), len(pcomb))
        for i in range(len(pcomb)):
            assert isinstance(pcomb[i], int),\
                "Unsupported type of choice at position {}, expected int, got: {}.".format(i, type(pcomb[i]))
            assert pcomb[i] in self._vchoices[i],\
                "Choice of posiiton {} of combination is invalid, expected one of: {}, got: {}.".format(i, self._vchoices[i], pcomb[i])
        # if you are here, you are good to go

    def add(self, pcomb):
        # pcomb: partial combination, always contain concrete choice,
        #        e.g., for caps of 6 positions, the followings are valid (within range):
        #          - (2, 3): block pattern (2, 3, None, None, None, None)
        #          - (2, 3, 5, 1, 5, 7): block specific combination
        #        and the followings are incorrect:
        #          - (2, 3)
        #          - (1, 2, 3)
        #          - (1, 2, 3, 4, 5, 6, 7, 8)
        self.validate_pcomb(pcomb)
        tmp_tracker = self._tracker
        for i in range(len(pcomb)):

            if tmp_tracker is None:
                # up to this level it's already pruned
                # no need to insert anymore
                return

            tmp_key = pcomb[i]
            if i==len(pcomb)-1:
                # the last position in combination, block at this level
                tmp_tracker[tmp_key] = None
            else:
                # not the last position, move on
                if tmp_key not in tmp_tracker.keys():
                    tmp_tracker[tmp_key] = {}
                tmp_tracker = tmp_tracker[tmp_key]
            
        # try to see whether anything can be merged
        self.update(self._tracker, pcomb, 0)
        self.update_root()

    def update_root(self):
        # note that if self._tracker is None, then the whole skeleton is blocked
        if len(self._tracker.keys()) >= self._caps[0]:
            can_merge = True
            for dkey in self._tracker.keys():
                if self._tracker[dkey] is not None:
                    can_merge = False
            if can_merge:
                self._tracker = None
        return

    def update(self, ptracker, pcomb, pos):
        # recursive method, merge from bottom to top
        # this queyr the same partial combination 
        # and try to check whether any tracker state can be merged or not

        # at this level, this call is trying to merge ptracker[pcomb[pos]]'s children
        if pos+1 >= len(pcomb):
            return

        pkey = pcomb[pos]
        self.update(ptracker[pkey], pcomb, pos+1)
        if len(ptracker[pkey].keys()) >= self._caps[pos+1]:
            # ready to merge since all the choices are explored
            can_merge = True
            for dkey in ptracker[pkey].keys():
                if ptracker[pkey][dkey] is not None:
                    can_merge = False
            if can_merge:
                # all of the keys are None (blocked), which means the parent should be blocked
                # merge!
                ptracker[pkey] = None
        # else: cannot merge since not all the choices are explored

        # done and return
        return

    def check(self, pcomb, return_root_cause_pos=False):
        # check whether pcomb is feasible or not
        # return True if feasible (not blocked), False if blocked

        # print("checking pcomb={}, tracker={}".format(pcomb, self._tracker))
        self.validate_pcomb(pcomb)

        # start checking
        if self._tracker is None:
            # the whole skeleton is blocked
            if return_root_cause_pos:
                return (False, -1)
            else:
                return False

        tmp_tracker = self._tracker
        for i in range(len(pcomb)):
            ikey = pcomb[i]
            if ikey not in tmp_tracker.keys():
                # no blocking record halfway, which means it's good
                if return_root_cause_pos:
                    return (True, i)
                else:
                    return True
            else:
                if tmp_tracker[ikey] is None:
                    # has blocking record halfway, which means it's already blocked
                    if return_root_cause_pos:
                        return (False, i)
                    else:
                        return False
                else:
                    if i==len(pcomb)-1:
                        # pcomb being checked is actually more abstract than dict
                        # since tmp_tracker[ikey] is not None, it means not all of the next level are pruned
                        # so here you should return True
                        if return_root_cause_pos:
                            return (True, i)
                        else:
                            return True
                    else:
                        # don't know at this level, move to next level
                        tmp_tracker = tmp_tracker[ikey]

        # if you return here, something is wrong
        raise Exception("You should not reach here. This happens usually because the tail position is not None. Check your impl.")

    def vcomb_advance(self, vcomb, pos):
        # advance 1 step for the designated position
        # if running out of choices, back track to previous position, respectively
        curr_comb = list(vcomb)
        curr_pos = pos
        curr_val = curr_comb[curr_pos]
        curr_c = self._rchoices[curr_pos][curr_val]
        next_c = curr_c + 1
        while next_c >= len(self._cchoices[curr_pos]):
            # run out of choices at this level, back track to previous position
            curr_pos -= 1
            if curr_pos <= -1:
                # run out of all choices at all positions
                return None
            curr_val = curr_comb[curr_pos]
            curr_c = self._rchoices[curr_pos][curr_val]
            next_c = curr_c + 1
        # if you are here, you are good to move on
        curr_comb[curr_pos] = self._cchoices[curr_pos][next_c]
        for i in range(curr_pos+1, len(vcomb)):
            # set to the 1st choice
            curr_comb[i] = self._cchoices[i][0]
        return curr_comb

    def next_available(self, vcomb):
        # returns the next available vcomb (starting from the given vcomb, including)
        # or None if none is available.
        # note: need to call update() before you call next_available(), since its efficiency
        #       relies on the updated/sorted state of the tracker
        curr_comb = list(vcomb) # make mutable
        while True:
            res, block_pos = self.check(curr_comb, return_root_cause_pos=True)
            if block_pos==-1:
                # the whole space is blocked, don't bother, just return
                return None
            if block_pos<-1 or block_pos>=len(curr_comb):
                raise Exception("You should not reach here: block_pos is invalid, got: {}, curr_comb is {}.".format(block_pos, curr_comb))
            if res:
                # good to go
                return tuple(curr_comb)
            else:
                # otherwise check the next choice in the block level and try
                # move down 1 choice in the block pos, and reset all afterwards to initial choices
                curr_comb = self.vcomb_advance(curr_comb, block_pos)
                if curr_comb is None:
                    # run out of choices
                    return None

class LineSkeletonIterator():
    '''
    A true lazy iterator for concretizing full skeleton. Note that this requires full skeleton where only the following tokens are allowed:
    - FunctionProduction
    - ParamProduction
    - EnumType
    - Int (which corresponds to reference token)
    This constructs a lazy iterator for a single full line skeleton.
    '''
    def __init__(self, builder, name, skeleton):
        self._skeleton = skeleton
        self._name = name # identifiable name of skeleton
        self._builder = builder

        # initialization
        # first identify all the EnumTypes and construct a list of pools in a depth-first post-order way
        # -- the same way we construct a program in the future
        self._pools = self.identify_enum_types(self._skeleton)
        self._pools_rinds = [
            {p[i].id:i for i in range(len(p))}
            for p in self._pools
        ]
        # this is the index list of self._pools
        # the lazy iterator will use this indices instead of direct sampling on self._pools
        self._pseqs = [list(range(len(self._pools[i]))) for i in range(len(self._pools))]
        self._size = reduce(lambda x,y: x*len(y), self._pseqs, 1)
        self._ptr = 0

        # precompute lazy product helpers
        # refer to: https://github.com/tylerburdsall/lazy-cartesian-product-python
        self._divs = []
        self._mods = []
        factor = 1
        for i in range(len(self._pseqs)-1, -1, -1):
            items = len(self._pseqs[i])
            self._divs.insert(0, factor)
            self._mods.insert(0, items)
            factor = factor * items

        # note: argument level conflict-driven learning
        #       key is context, value is condition predicate (lambda function)
        # self._predicates = {}
        # self._imported_predicates = {} # separating imported and collected
        self._block_tracker = BlockTracker(choices=[
            [
                q.id
                for q in p._pool
            ]
            for p in self._pools
        ])

    def reset_ptr(self):
        self._ptr = 0

    def __next__(self):
        return self.next()

    # @profiler.ctimer("iterator.next")
    # def next(self):
    #     if self._ptr >= self._size:
    #         raise StopIteration()
    #     self._ptr += 1
    #     # return self.__getitem__(self._ptr-1)
    #     return (
    #         self._ptr-1 == 0, # detect whether this is the first candidate in the skeleton
    #         self.__getitem__(self._ptr-1) # the program itself
    #     )

    @profiler.ctimer("iterator.next")
    def next(self):
        # note-important: normally spaces before self._ptr are all blocked (densely blocked),
        #                 but in Keystroke setting, this is not always the case, since there are
        #                 programs that can't be verified true nor false, but skipped
        nx_vcomb = self._block_tracker.next_available(self.icomb_to_vcomb(self.ptr_to_icomb(self._ptr)))
        if nx_vcomb is None:
            raise StopIteration()
        nx_ptr = self.icomb_to_ptr(self.vcomb_to_icomb(nx_vcomb))
        self._ptr = nx_ptr + 1
        return (
            nx_ptr == 0, # detect whether this is the first candidate in the skeleton
            self.__getitem__(nx_ptr) # the program itself
        )

    def identify_enum_types(self, sunit):
        ret_pools = []
        if isinstance(sunit, list):
            for i in range(len(sunit)):
                ret_pools += self.identify_enum_types(sunit[i])
        elif isinstance(sunit, FunctionProduction):
            pass
        elif isinstance(sunit, int):
            pass
        elif isinstance(sunit, ParamProduction):
            pass
        elif isinstance(sunit, EnumType):
            prods = self._builder.get_productions_with_lhs(sunit)
            # only use EnumProds
            enum_prods = [p for p in prods if p.is_enum()]
            assert len(enum_prods) > 0, "No EnumProduction is available for EnumType {}. You need to refine your dsl or skeleton.".format(sunit.name)
            ret_pools.append(Pool(enum_prods))
        else:
            raise NotImplementedError("Unsupported type for a full skeleton, got: {}.".format(type(sunit)))
        return ret_pools

    def __len__(self):
        return self._size

    def icomb_to_ptr(self, icomb):
        ptr = 0
        for i in range(len(icomb)-1, -1, -1):
            ptr += self._divs[i] * icomb[i]
        return ptr

    def ptr_to_icomb(self, ptr):
        combination = []
        for i in range(0, len(self._pseqs)):
            combination.append(self._pseqs[i][ int(math.floor(ptr / self._divs[i])) % self._mods[i] ])
        return tuple(combination)

    # vcomb stores production id
    # icomb stores combination index
    def icomb_to_vcomb(self, icomb):
        return tuple([self._pools[i][icomb[i]].id for i in range(len(icomb))])

    def vcomb_to_icomb(self, vcomb):
        return tuple([self._pools_rinds[i][vcomb[i]] for i in range(len(vcomb))])

    def __getitem__(self, n):
        if n<0 or n>=self._size:
            raise IndexError("i={}, size={}".format(n, self._size))

        # first compute combination choices for every Pool
        icomb = self.ptr_to_icomb(n)
        # then construct the program recursively in a depth-first post-order way
        # argument level conflict-driven learning: before construction, query the stored predicates
        # note: convert the combination (ptr based) here into combination (prod id based)
        # note-important:
        #     a combination in interpreters is production id based
        #     a combination here is pointer-to-iter based (see usage down in construct_ast)
        #     so you need to covert it here to fit the predicate function returned from interpreter
        vcomb = self.icomb_to_vcomb(icomb)
        if self.check_combination(vcomb):
            return self.construct_ast(self._skeleton, icomb)
        else:
            return None

    @profiler.ctimer("iterator.check_combination")
    def check_combination(self, vcomb):
        return self._block_tracker.check(vcomb)

    @profiler.ctimer("iterator.construct_ast")
    def construct_ast(self, cmds, comb):
        top_level_outputs = [] # used for reference tracking

        def rec_make_node(sunit, pos):
            curr_pos = None
            curr_node = None
            if isinstance(sunit, list):
                # first pos is FunctionProduction, presumably validated by rec_validate
                tmp_pos = pos
                tmp_args = []
                for i in range(1, len(sunit)):
                    # skip the first pos, since we will make it from this level
                    tmp_pos, tmp_node = rec_make_node(sunit[i], tmp_pos)
                    tmp_args.append(tmp_node)
                curr_pos = tmp_pos
                curr_node = self._builder.make_node(sunit[0], tmp_args)
            elif isinstance(sunit, FunctionProduction):
                raise ValueError("FunctionProduction should be assembled from within the previous level.")
            elif isinstance(sunit, int):
                curr_pos = pos
                # refer to a top level output
                curr_node = top_level_outputs[sunit]
            elif isinstance(sunit, ParamProduction):
                curr_pos = pos
                curr_node = self._builder.make_node(sunit)
            elif isinstance(sunit, EnumType):
                curr_pos = pos + 1
                # note: track the current position in combination as cpos, which will be used to blame nodes in the future
                curr_node = self._builder.make_node( self._pools[pos][comb[pos]], tag={"cpos": pos} )
            else:
                raise NotImplementedError("Unsupported type for a full skeleton, got: {}.".format(type(sunit)))
            return curr_pos, curr_node

        top_pos = 0
        for k in range(len(cmds)):
            top_pos, top_node = rec_make_node(cmds[k], top_pos)
            top_level_outputs.append(top_node)
        assert top_pos == len(comb), "Numbers of combination elements used don't match, expected {}, but got {}.".format(len(comb), top_pos)
        # print("# [debug] number of outputs: {}".format(len(top_level_outputs)))
        return top_level_outputs[-1]

    def get_combination_suite(self, cntx, extlen):
        '''
        Generate a target testing suite of partial combinations for given context and extended length.
        Returns an iterator
        '''
        tmp_clist = []
        for i in range(len(cntx)):
            # one context position has one choice only
            tmp_clist.append([cntx[i]])
        for i in range(len(cntx), extlen):
            tmp_clist.append([p.id for p in self._pools[i]._pool])
        return product(*tmp_clist)

    def update(self, core):
        # print("update: context={}, extlen={}".format(core._context, core._extlen))
        if isinstance(core, EnumAssertion):
            combs = self.get_combination_suite(core._context, core._extlen)
            # tmp_combs = self.get_combination_suite(core._context, core._extlen)
            # logger.debug("suite is: {}".format(list(tmp_combs)))
            for c in combs:
                if self._block_tracker.check(c):
                    # only block those good ones, for bad ones, they are already bad, can skip
                    if not core._condition(c):
                        # logger.debug("blocking: {}".format(c))
                        # print("# comb={}, rejected")
                        # unsat, add to tracker
                        self._block_tracker.add(c)
        elif isinstance(core, ComponentError) or isinstance(core, EqualityAssertion):
            combs = self.get_combination_suite(core._context, len(core._context))
            for c in combs:
                if self._block_tracker.check(c):
                    # only block those good ones, for bad ones, they are already bad, can skip
                    if not core._condition(c):
                        # print("# comb={}, rejected")
                        # unsat, add to tracker
                        self._block_tracker.add(c)
        else:
            raise NotImplementedError("Unsupported core type, got: {}.".format(type(core)))

class LineSkeletonEnumerator(Enumerator):

    def __init__(self, spec: S.TrinitySpec, cands: List[Any]):
        '''
        Initialize an enumerator that takes as input a list of candidate programs written in line skeleton format.
        '''
        self._spec = spec
        self._builder = D.Builder(spec)
        # if you see any slot that require this type, raise a warning
        # because a type that is not terminal could introduce infinite derivations
        # if it both appears in a production's LHS and RHS
        self._warning_types = set([ p.lhs for p in self._builder.productions() if not p.is_enum() ])
        # fixme: add cyclic substitution detection later
        logger.info("The following types contain non-terminal production rules: {}. ".format(", ".join([str(p) for p in self._warning_types])) + \
            "Try not to leave them as holes in your skeleton. " + \
            "Even though a spawning procedure can try to complete them, a cyclic substitution may still happen.")

        # note: in Trinity, EnumType is terminal type, ValueType is non-terminal type
        # note-important: in Trinity, a Type can only either be terminal or non-terminal; it can't be both
        # EnumType <- EnumProduction
        # ValueType <- ParamProduction
        # ValueType <- FunctionProduction
        self._all_types = self._spec._type_spec.types()
        self._terminal_types = [p for p in self._all_types if p.is_enum()]
        self._nonterminal_types = [p for p in self._all_types if p.is_value()]

        # store the original skeletons
        self._skeletons = cands
        # prepare candidate full line skeletons
        self._cands = []
        for sk in cands:
            self._cands += self.canonicalize(sk)
        # prepare iterators for every full line skeleton
        self._iters = [ 
            LineSkeletonIterator(self._builder, self.rec_list_to_tuple(self.pretty_print_ir1_cmd(sk)), sk)
            for sk in self._cands 
        ]

        self._iter_ptr = 0

        # note: skeleton level CDCL, list elements are lambda functions that match skeleton name
        #       e.g., the following represents a predicate (skeleton pattern):
        #       (('gather|select', 'Param@0', 'EnumType@ColList'), 
        #        ('separate', 'ref@0', 'EnumType@ColInt'), 
        #        (None, 'ref@1', 'EnumType@ColList'))
        #       where 'a|b' indicates a set, None indicates anything
        # note: call `skeleton_pattern_matching` to match between a predicate and a skeleton
        self._skeleton_patterns = []

    def export_knowledge_base(self):
        # export reusable knowledge base
        # the exported knowledge base can only be used for the same enumerator for ks/recovery purpose
        tmp_kb = {
            "skeleton_patterns": copy.deepcopy( self._skeleton_patterns ),
            "iterator_trackers": [self._iters[i]._block_tracker.make_copy() for i in range(len(self._iters))]
        }
        return tmp_kb

    def import_knowledge_base(self, arg_kb):
        # import only one knowledge base
        # if you want to import multiple, consider using the merge method
        self._skeleton_patterns = arg_kb["skeleton_patterns"]
        for i in range(len(self._iters)):
            self._iters[i]._block_tracker = arg_kb["iterator_trackers"][i].make_copy()

    def rec_skeleton_pattern_matching(self, pred, tgt):
        # match a skeleton pattern (predicate) with a specific skeleton
        # see comments for self._predicates for formatting details
        # pred: predicate, can be a concrete skeleton, or a patterned skeleton
        # tgt: target, should be concrete skeleton
        # return True if matched
        if isinstance(pred, tuple) and isinstance(tgt, tuple):
            if len(pred) != len(tgt):
                return False
            for i in range(len(pred)):
                if not self.rec_skeleton_pattern_matching(pred[i], tgt[i]):
                    return False
            return True
        elif isinstance(pred, str) and isinstance(tgt, str):
            if "|" in pred:
                tmp_toks = pred.split("|")
                if tgt in tmp_toks:
                    return True
                else:
                    return False
            else:
                if pred == tgt:
                    return True
                else:
                    return False
        elif pred is None and isinstance(tgt, str):
            return True
        elif pred is None and isinstance(tgt, tuple):
            return True
        else:
            raise NotImplementedError("Unsupported types for skeleton pattern matching, got: {} and {}.".format(pred, tgt))
        
    def parse_reference(self, refstr):
        '''
        Extract reference value from reference token
        '''
        assert refstr.startswith("ref@")
        try:
            refint = int(refstr[4:])
            assert refint >= 0
            return refint
        except ValueError as e:
            raise ValueError("Error parsing reference value for: {}".format(refstr))

    def rec_tuple_to_list(self, ir):
        # convert tuple from ir to list, recursively
        # this is helpful for post-processing for results returned by product
        if isinstance(ir, tuple) or isinstance(ir, list):
            return [self.rec_tuple_to_list(p) for p in ir]
        else:
            return ir

    def rec_list_to_tuple(self, ir):
        # convert list to tuple, recursively
        # this is helpful for maintaining skeleton name
        if isinstance(ir, tuple) or isinstance(ir, list):
            return tuple([self.rec_list_to_tuple(p) for p in ir])
        else:
            return ir

    def pretty_print_ir0_cmd(self, cmd):
        # this function is only meant for ir0
        retcmd = []
        for i in range(len(cmd)):
            if isinstance(cmd[i], list):
                retcmd.append(self.pretty_print_ir0_cmd(cmd[i]))
            elif isinstance(cmd[i], int):
                retcmd.append("ref@{}".format(cmd[i]))
            elif isinstance(cmd[i], FunctionProduction):
                retcmd.append(cmd[i].name)
            elif isinstance(cmd[i], ParamProduction):
                retcmd.append("Param@{}".format(cmd[i]._param_id))
            elif cmd[i] is None:
                retcmd.append(cmd[i])
            else:
                raise NotImplementedError("Invalid type: {}.".format(type(cmd[i])))
        return retcmd

    def pretty_print_ir1_cmd(self, cmd):
        # this function is only meant for ir1 (somehow you can also use it to print ir2)
        retcmd = []
        for i in range(len(cmd)):
            if isinstance(cmd[i], list):
                retcmd.append(self.pretty_print_ir1_cmd(cmd[i]))
            elif isinstance(cmd[i], int):
                retcmd.append("ref@{}".format(cmd[i]))
            elif isinstance(cmd[i], FunctionProduction):
                retcmd.append(cmd[i].name)
            elif isinstance(cmd[i], ParamProduction):
                retcmd.append("Param@{}".format(cmd[i]._param_id))
            elif isinstance(cmd[i], Type):
                if isinstance(cmd[i], EnumType):
                    retcmd.append("EnumType@{}".format(cmd[i].name))
                elif isinstance(cmd[i], ValueType):
                    retcmd.append("ValueType@{}".format(cmd[i].name))
                else:
                    raise NotImplementedError("Invalid type: {}.".format(type(cmd[i])))
            else:
                raise NotImplementedError("Invalid type: {}.".format(type(cmd[i])))
        return retcmd

    def canonicalize(self, cand: List[Any]):
        '''
        Translate a json skeleton with string/reference/int/null into internal skeleton with dsl components;
        also validate the number of holes in the json skeleton and type-check the concrete assignments in the skeletons
        '''
        def rec_transform(sunit):
            # this transforms a json command into IR0 format
            # note: this is recursive, so a sunit can be a full command or a token
            if isinstance(sunit, list):
                return [rec_transform(p) for p in sunit]
            else:
                if isinstance(sunit, str):
                    if sunit.startswith("ref@"):
                        # special token: reference token
                        # parse it into int
                        # note: param becomes ParamProduction, and reference token get parsed into int
                        #       both are fine anr there's no conflict in this phase
                        return self.parse_reference(sunit)
                    else:
                        # function production
                        return self._builder.get_function_production_or_raise(sunit)
                elif isinstance(sunit, int):
                    # param production
                    return self._builder.get_param_production_or_raise(sunit)
                elif sunit is None:
                    # hole
                    return None
                else:
                    raise ValueError("Cannot recognize skeleton unit: {}.".format(sunit))

        def rec_validate(cmds, curr_cmd):
            # this validates IR0s one by one
            # cmds is actually a skeleton
            for i in range(len(cmds)):
                # validate commands one by one
                # note: if this is in a recursive call stack with specific command assigned
                #       process this command and break the loop after finishing the assignment (no loop)
                # note: we always need cmds because curr_cmd (even nested) can also refer to previous outputs
                # note: if curr_cmd is None, we are in top level, otherwise in recursive level
                cmd = cmds[i] if curr_cmd is None else curr_cmd

                curr_prod = cmd[0]
                assert isinstance(curr_prod, FunctionProduction), "First token of every command should be FunctionProduction, got: {}.".format(curr_prod)
                curr_rhs = curr_prod.rhs
                # validate number of holes
                assert len(curr_rhs) == len(cmd)-1, \
                    "Command arity doesn't match the declared function production arity. " + \
                    "FunctionProduction is {}. Required {} rhs, but got {}.".format(curr_prod.name, len(curr_rhs), len(cmd)-1)
                # valid type of holes
                for j in range(len(curr_rhs)):
                    if cmd[j+1] is None:
                        # hole
                        if curr_rhs[j] in self._warning_types:
                            logger.warning("Non-terminal type {} is left as a hole in {}th token of command {}, which *may* cause cyclic substitution.".format(curr_rhs[j], j+1, self.pretty_print_ir0_cmd(cmd)))
                    elif isinstance(cmd[j+1], int):
                        # special token: reference token
                        # note: we also need to do semantic checking here (don't refer to future steps)
                        assert cmd[j+1] < i, "You can't refer to future/current outputs: current output step is {}, referred output step is {}.".format(i, cmd[j+1])
                        # reference production: still need to check type
                        assert cmds[cmd[j+1]][0].lhs == curr_rhs[j], "Type mismatch for command: {} at {}th token.".format(cmd, j+1)
                    elif isinstance(cmd[j+1], list):
                        # function production
                        # note: FunctionProduction should always be wrapped in a list
                        assert curr_rhs[j] == cmd[j+1][0].lhs, "Type mismatch for command: {} at {}th token.".format(self.pretty_print_ir0_cmd(cmd), j+1)
                        # then move on to next level: depth first validation
                        rec_validate(cmds, cmd[j+1])
                    elif isinstance(cmd[j+1], EnumProduction) or isinstance(cmd[j+1], ParamProduction):
                        # enum production or param production
                        assert curr_rhs[j] == cmd[j+1].lhs
                    else:
                        raise ValueError("Can't validate the unit {}: unknown components.".format(cmd[j+1]))

                # note: see above
                if curr_cmd is not None:
                    break

        def rec_filltype(sunit):
            # this transforms IR0s to IR1s by filling in actual type instances
            # note: this is recursive, so a sunit can be a full command or a token
            curr_prod = sunit[0]
            curr_rhs = curr_prod.rhs
            # note-important: a skeleton's sunit[0] is always a FunctionProduction
            #                 which should always be concretely specified
            runit = [sunit[0]]
            for i in range(len(curr_rhs)):
                if sunit[i+1] is None:
                    # a hole
                    # append Type
                    runit.append(curr_rhs[i])
                elif isinstance(sunit[i+1], int):
                    # a reference token: don't modify just keep the reference
                    # note: since this is well typed by rec_validate, so it's fine
                    runit.append(sunit[i+1])
                elif isinstance(sunit[i+1], list):
                    # a function production
                    # recursive call
                    runit.append(rec_filltype(sunit[i+1]))
                elif isinstance(sunit[i+1], EnumProduction) or isinstance(sunit[i+1], ParamProduction):
                    # a enum production or param production
                    # similar to rec_valid same branch
                    # just append
                    runit.append(sunit[i+1])
                else:
                    raise ValueError("Can't fill in type for the unit: {}.".format(sunit[i+1]))
            return runit

        def rec_spawn(cmds, curr_cmd):
            # this creates a generator that iterates all (potential) full skeletons, given a (potential) non-full skeleton
            # essentially this removes all ValueTypes by subsituting them with Productions and EnumTypes
            # note: this returns a generator, and you need to iterate it to get all the full skeletons
            # note: this may get stuck if there's introduced cyclic substitution
            # note: this has similar calling stacks with rec_validate
            # cmds is actually a skeleton
            tmp_ir = [] # temporary container
            for i in range(len(cmds)):
                cmd = cmds[i] if curr_cmd is None else curr_cmd
                curr_prod = cmd[0]
                curr_rhs = curr_prod.rhs
                tmp_cmd = [] # temporary container
                tmp_cmd.append(Pool([curr_prod]))
                for j in range(len(curr_rhs)):
                    # note: if it's a Type, then it's from a hole and may need spawn
                    #       if it's a Production, no need to do anything
                    #       if it's an int (reference), no need to do anything
                    if cmd[j+1] in self._nonterminal_types:
                        # if this type is non-terminal, then try to derive to full
                        prods = self._builder.get_productions_with_lhs(cmd[j+1])
                        enum_prods, param_prods, func_prods = [], [], []
                        for p in prods:
                            if p.is_enum():
                                enum_prods.append(p)
                            elif p.is_param():
                                param_prods.append(p)
                            elif p.is_function():
                                func_prods.append(p)
                            else:
                                raise ValueError("Unknown Production subtype for type {}, got: {}.".format(cmd[j+1], p))
                        # note: according to Trinity definition, this type is non-terminal type, and it can't be a EnumType
                        assert len(enum_prods) == 0
                        tmp_pool = Pool([])
                        for p in param_prods:
                            tmp_pool.append(self._builder.make_node(p))
                        for p in func_prods:
                            tmp_pool += rec_spawn(cmds, [p]+p.rhs)

                        tmp_cmd.append(tmp_pool)
                    elif isinstance(cmd[j+1], list):
                        tmp_cmd.append(rec_spawn(cmds, cmd[j+1]))
                    else:
                        tmp_cmd.append(Pool([cmd[j+1]]))
                # then expand tmp_cmd into combinations
                # note: need to unwrap into list
                cmd_collection = list(product(*[tmp_cmd[k]._pool for k in range(len(tmp_cmd))]))

                # note: see above
                if curr_cmd is not None:
                    return Pool(cmd_collection)
                else:
                    tmp_ir.append(Pool(cmd_collection))

            # if you reach here, generate the collection (product) of the full ir
            # print("# [debug] sizes for cmds: {}".format([len(tmp_ir[k]._pool) for k in range(len(tmp_ir))]))
            ir_collection = list(product(*[tmp_ir[k]._pool for k in range(len(tmp_ir))]))
            return Pool(ir_collection)

        # typical process of parsing a line skeleton
        # input / original json: 
        #     [
        #         ["Select", 0, null, null],
        #         ["Select", "ref@0", null, ["Str2Val", null]],
        #         ["Extract", "ref@1", null, null]
        #     ]
        # cand_tmp0 / internal representation 0, IR0 / rec_transform:
        #     [
        #         [SelectProd, ParamProd@0, None, None],
        #         [SelectProd, 0, None, [Str2ValProd, None]],
        #         [ExtractProd, 1, None, None]
        #     ]
        # cand_tmp1 / internal representation 1, IR1 / rec_filltype:
        #     [
        #         [SelectProd, ParamProd@0, EnumType@Pp, ValueType@Val],
        #         [SelectProd, 0, EnumType@Pp, [Str2ValProd, EnumType@Str]],
        #         [ExtractProd, 1, EnumType@Pp, ValueType@Val]
        #     ]
        # cand_tmp2 / internal representation 2, IR2 / full (line) skeleton / :
        #     ...

        # step1: replace all valid productions in every command
        cand_tmp0 = [rec_transform(cand[i]) for i in range(len(cand))]
        # step2: validate the number of holes and the position of concrete assignment
        rec_validate(cand_tmp0, None)
        # step3: fill in types for those slots
        cand_tmp1 = [rec_filltype(cand_tmp0[i]) for i in range(len(cand_tmp0))]
        # step4: spawn a skeleton into (potential) full skeleton(s)
        cand_tmp2 = rec_spawn(cand_tmp1, None)
        # cand_tmp2 = Pool(self.rec_tuple_to_list(cand_tmp2._pool))
        cand_tmp2 = self.rec_tuple_to_list(cand_tmp2._pool)
        # everything's fine
        return cand_tmp2

    def get_ir2_nslot(self, cmd):
        ret_nslot = 0
        for i in range(len(cmd)):
            if isinstance(cmd[i], list):
                ret_nslot += self.get_ir2_nslot(cmd[i])
            elif isinstance(cmd[i], int):
                pass
            elif isinstance(cmd[i], FunctionProduction):
                pass
            elif isinstance(cmd[i], ParamProduction):
                pass
            elif isinstance(cmd[i], Type):
                if isinstance(cmd[i], EnumType):
                    ret_nslot += 1
                # elif isinstance(cmd[i], ValueType):
                #     retcmd.append("ValueType@{}".format(cmd[i].name))
                else:
                    raise NotImplementedError("Invalid type: {}.".format(type(cmd[i])))
            else:
                raise NotImplementedError("Invalid type: {}.".format(type(cmd[i])))
        return ret_nslot

    # def convert_ir2_to_sketch(self, cmds):
    #     return [cmds[i][0].name for i in range(len(cmds))]

    @profiler.ctimer("check_iterator")
    def check_iterator(self, it):
        # for p in self._imported_skeleton_patterns:
        #     # print("# [debug] imported checking: p={}, it={}".format(p, it._name))
        #     if self.rec_skeleton_pattern_matching(p, it._name):
        #         return False
        for p in self._skeleton_patterns:
            # print("# [debug] collected checking: p={}, it={}".format(p, it._name))
            if self.rec_skeleton_pattern_matching(p, it._name):
                return False
        return True

    # returns (fc, prog)
    # fc: is this the first candidate in the skeleton
    @profiler.ctimer("next")
    def next(self) -> Tuple[ bool, Optional[Node] ]:
        try:
            # note: this is required, since update with SkeletonAssertion may move the ptr forward
            #       which may implicitly make the ptr exceed
            if self._iter_ptr >= len(self._iters):
                # out of bound, no need to do skeleton level deduction, so make fc False
                return (False, None)

            # note: cold start skeleton checking
            #       this is required when the enumerator inherits an external KB
            if not self.check_iterator(self._iters[self._iter_ptr]):
                self._iter_ptr += 1
                return self.next()

            fc, ret_prog = next(self._iters[self._iter_ptr])
            while ret_prog is None:
                # this means the program fails the check
                # then keep trying until the enumerated program is not None
                fc, ret_prog = next(self._iters[self._iter_ptr])
            # attach skeleton information
            ret_prog._tag = {
                # "skeleton": self.pretty_print_ir1_cmd(self._cands[self._iter_ptr]),
                "skeleton": self.rec_list_to_tuple(self.pretty_print_ir1_cmd(self._cands[self._iter_ptr])),
                "nslot": self.get_ir2_nslot(self._cands[self._iter_ptr]),
                "iter_ptr": self._iter_ptr,
            }

            # print("# [debug] iter_ptr: {}".format(self._iter_ptr))
            # logger.debug("iter_ptr: {}".format(self._iter_ptr))
            return (fc, ret_prog)
        except StopIteration:
            if self._iter_ptr + 1 >= len(self._iters):
                # out of bound, no need to do skeleton level deduction, so make fc False
                return (False, None)
            else:
                # move to next skeleton
                self._iter_ptr += 1
                # test against skeleton level predicates until it finds one that fits
                while True:
                    if self.check_iterator(self._iters[self._iter_ptr]):
                        break
                    else:
                        self._iter_ptr += 1

                return self.next()

    @profiler.ctimer("update")
    def update(self, core: Any=None) -> None:
        '''
        Update the internal state of the enumerator. This can be useful when trying to prune the search space.
        By default, it does nothing.
        '''
        profiler.add1("update.{}".format(type(core).__name__))
        if isinstance(core, EnumAssertion):
            # print("# [DEBUG] core._context: {}".format(core._context))
            # parameter level conflict-driven learning
            self._iters[self._iter_ptr].update(core)
            pass
        elif isinstance(core, SkeletonAssertion):
            # fixme: an initial version, currently we only skip the current skeleton, may be extended to multiple skeletons
            self._skeleton_patterns.append(core._prog._tag["skeleton"])
            # by default, when throwing a SkeletonAssertion, it implies that you skip the current skeleton already
            # self._iter_ptr += 1
        elif isinstance(core, ComponentError):
            # similar to EnumAssertion
            self._iters[self._iter_ptr].update(core)
            pass
        elif isinstance(core, EqualityAssertion):
            self._iters[self._iter_ptr].update(core)
            pass
        else:
            raise NotImplementedError("Unsupported InterpreterError type for update, got: {}.".format(type(core)))
        # DEBUG: additional checking
        # tmp0 = self._iters[3].check_combination((25,22,98,9))
        # logger.debug("==== (25,22,98,9): {} ====".format(tmp0))
        
