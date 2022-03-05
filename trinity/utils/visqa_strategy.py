import itertools
import numpy as np
import pandas as pd

from collections import Counter
from .visqa import normalize_table, parse_value

def denote_aggr(arg_aggr_id):
    if arg_aggr_id==0:
        return lambda x: sorted(x, key=lambda y:str(y)) if isinstance(x,list) else [x]
    elif arg_aggr_id==1:
        return lambda x: [sum(x)]
    elif arg_aggr_id==2:
        return lambda x: [sum(x)/len(x)]
    elif arg_aggr_id==3:
        return lambda x: [len(x)]
    else:
        raise NotImplementedError("Unsupported aggr id, got: {}.".format(arg_aggr_id))

# NOTE!!!: this is problematic, need to re-do since not all the duplications need to be removed
#          e.g., sum of 1 1 1 2 2 2
def remove_value_duplication(arg_cpplist, arg_table):
    # this only keeps the unique value with highest probability if there's duplication
    # this method is value-based (as compared to iloc-based)
    ts = set()
    ret_cpplist = []
    for i in range(len(arg_cpplist)):
        p = arg_cpplist[i]
        if arg_table.iloc[p[0],p[1]] not in ts:
            ts.add(arg_table.iloc[p[0],p[1]])
            ret_cpplist.append(p)
        # else: do nothing
    return ret_cpplist

# ============================================ #
# ============================================ #
# ============================================ #
# ============================================ #
# dynamic strategy
# arg_key_values: values from the table for prioritization
# note: this returns a more preferred candidate set `high_values` and a less preferred one `low_values`
# fixme: values in column names are not indexed
MAX_FSC = 10 # maximum number of candidate returned
def fallback_strategy_comparison(arg_rendered_table, arg_key_values):
    # sub table that only contains values of numeric types
    table_numeric = arg_rendered_table.select_dtypes(np.number)

    # def parse_number(v):
    #     try:
    #         return int("{:.4f}".format(v))
    #     except:
    #         return float("{:.4f}".format(v))

    def row_wise_comparison(arg_row0, arg_row1):
        # extract all comparable values from both arg_row0 and arg_row1
        # and compute difference between them
        # note/fixme: values from within the same row are treated as different types
        #             and won't be compared
        # fixme: types can be more fine-grained than number when necessary, e.g., int vs float
        tmp_values0 = table_numeric.iloc[arg_row0,:].tolist()
        tmp_values1 = table_numeric.iloc[arg_row1,:].tolist()
        tmp_pairs = list(itertools.product(tmp_values0,tmp_values1))
        # for every pair, compare forward and backward
        tmp_results = set()
        for p in tmp_pairs:
            # fixme: currently only consider the `-` operator
            tmp_results.add(p[0]-p[1])
            tmp_results.add(p[1]-p[0])
            # n0 = parse_number(p[0])
            # n1 = parse_number(p[1])
            # tmp_results.add(parse_number(n0-n1))
            # tmp_results.add(parse_number(n1-n0))
            # print("# n0={}, n1={}, n1-n0={}".format(n0,n1,n1-n0))
        return tmp_results

    # first extract key row numbers based on number of key values matched
    where_key_vals = [
        # save the rows only
        np.where(arg_rendered_table==p)[0].tolist()
        for p in arg_key_values
    ]
    # unpack the row ids
    row_counter = Counter()
    for p in where_key_vals:
        row_counter.update(p)
    # add some "dumb" row ids, i.e., every row id +1 for convenience of comparison
    row_counter.update(list(range(arg_rendered_table.shape[0])))
    # start from most common ones
    # print("# [debug] row_counter: {}".format(row_counter.most_common()))
    high_rows = list(filter(lambda x:x[1]>1, row_counter.most_common())) # remove the dumb ones (but the counter has accumulated dumb ones)
    low_rows = row_counter.most_common() # including all the dumb ones, a high row is also a low row

    # compute row-wise difference (comparison) using hierarchical BFS style
    high_pairs = list(itertools.combinations(high_rows, r=2))
    high_values = []
    for p in high_pairs:
        # p is ((row_id0, count), (row_id1, count))
        set_values = row_wise_comparison(p[0][0],p[1][0])
        # all these values share the same combined count
        for q in list(set_values):
            high_values.append((q,p[0][1]*p[1][1]))
    # sort by multiple keys, counter first, then prefer positive values
    high_values = sorted(high_values, key=lambda x:(x[1],x[0]), reverse=True)

    low_pairs = list(itertools.combinations(low_rows, r=2))
    low_pairs = [p for p in low_pairs if p not in high_pairs] # don't include those already in high_pairs
    low_values = []
    for p in low_pairs:
        set_values = row_wise_comparison(p[0][0],p[1][0])
        for q in list(set_values):
            low_values.append((q,p[0][1]*p[1][1]))
    # sort by multiple keys, counter first, then prefer positive values
    low_values = sorted(low_values, key=lambda x:(x[1],x[0]), reverse=True)

    return high_values[:MAX_FSC], low_values[:MAX_FSC]

# ============================================ #
# ============================================ #
# ============================================ #
# ============================================ #
# static strategy
def strategy_TaPas_A(arg_parsed_output, arg_rendered_table):
    # strategy A is **kind of** mimicking the original TaPas hard cut-off strategy with prob threshold >0.5
    # strategy rules: (operator, operand), where operand is always a list (of one or more values)
    # - PREPROCESSING
    #     - first filter the operand to only keep values with prob>=0.5
    # - NONE
    #     - wrap the values into separate rows with 1 column as an answer table
    # - SUM/AVERAGE/COUNT
    #     - same as above but proceed after the computation
    # - EXCEPTION
    #     - if the operand is an empty list, then there won't be an answer table
    #     - if operand is ill-typed for SUM/AVERAGE/COUNT, then there won't be an answer table
    #     - this strategy does not deal with duplication of cell values (nor duplication or answers
    #       since it only proposes one answer)
    tmp_op = arg_parsed_output[0]
    tmp_list = arg_parsed_output[1]
    qlist = remove_value_duplication(
        [cpp for cpp in tmp_list if cpp[2]>0.1],
        arg_rendered_table
    )
    # directly construct the table
    lambda_op = denote_aggr(tmp_op)
    vlist = [arg_rendered_table.iloc[p[0],p[1]] for p in qlist] # values
    plist = [p[2] for p in qlist] # probabilities
    
    ret_repr_answers = []
    olist = None
    if len(vlist)>0:
        try:
            olist = lambda_op(vlist)
        except TypeError:
            olist = ["<type error>"]
    else:
        olist = ["<no answer>"]
        
    ret_repr_answers.append(olist)
    
    return ret_repr_answers

# ============================================ #
# ============================================ #
# ============================================ #
# ============================================ #
# static strategy
def strategy_TaPas_B(arg_parsed_output, arg_rendered_table):
    tmp_op = arg_parsed_output[0]
    tmp_list = arg_parsed_output[1]
    qlist = remove_value_duplication(
        [cpp for cpp in tmp_list if cpp[2]>0],
        arg_rendered_table
    )
    # directly construct the table
    lambda_op = denote_aggr(tmp_op)
    vlist = [arg_rendered_table.iloc[p[0],p[1]] for p in qlist] # values
    plist = [p[2] for p in qlist] # probabilities
    
    ret_repr_answers = []
    olist = None
    if len(vlist)>0:
        try:
            olist = lambda_op(vlist)
        except TypeError:
            olist = ["<type error>"]
    else:
        olist = ["<no answer>"]
        
    ret_repr_answers.append(olist)
    
    return ret_repr_answers

# ============================================ #
# ============================================ #
# ============================================ #
# ============================================ #
# static strategy
def strategy_TaPas_C(arg_parsed_output, arg_rendered_table):
    # strategy C is a beam extension to the original TaPas strategy
    # strategy rules: (operator, operand), where operand is always a list (of one or more values)
    # - PREPROCESSING
    #     - list_a: filter the operand to only keep values with prob>=0.5
    #     - list_b: filter the operand to only keep values with prob>0.0
    #     - list_c: filter the operand to only keep values with prob>=0.0
    #     - TOP_K: top-k answers to include (only when len(list_a)!=1)
    # - ASSUMPTION
    #     - operator is always "more correct" than operand
    #       so when an operation is ill-typed, change the operand rather than the operator
    # - PROCEDURE
    #     - A0. WELL-TYPED SINGULAR ANSWER
    #         - IF: type_checked and len(list_a)==1
    #         - e.g., (NONE, ["France"])
    #         - THEN: wrap every value in **list_b** into a separate answer table
    #     - (NOT IMPLEMENTED HERE) A1. ILL-TYPED SINGULAR ANSWER
    #         - IF: not type_checked and len(list_a)==1
    #         - e.g., (SUM, ["France"])
    #         - THEN: wrap every value (mapped to its well-typed column) in **list_b** into 
    #                 a separate answer table
    #         - NOTE: if there are multiple well-typed columns, include all of them separately
    #     - B0. WELL-TYPED PLURAL/UNDETERMINISTIC ANSWER
    #         - IF: type_checked and ( len(list_a)>1 or len(list_a)==0 )
    #         - e.g., (NONE, []), (SUM, [1,2,3])
    #         - THEN: wrap every value combination from **list_b** into a separate answer table
    #                 and only keep the top-k candidates according to averaged probs
    #         - NOTE: if len(list_b)==0, [TODO] return blank answer
    #     - (NOT IMPLEMENTED HERE) B1. ILL-TYPED PLURAL/UNDETERMINISTIC ANSWER
    #         - IF: not type_checked and ( len(list_a)>1 or len(list_a)==0 )
    #         - e.g., (NONE, []), (SUM, ["France", "China"])
    #         - THEN: retrieve corresponding well-typed column values, and proceed with B1
    #         - NOTE: if len(list_b)==0, [TODO] return blank answer
    # - DUPLICATION
    #     - remove duplication is op is NONE
    #     - don't remove duplication for other ops
    
    def evaluate_answer_table(arg_cpp_list):
        vlist = [arg_rendered_table.iloc[cpp[0],cpp[1]] for cpp in arg_cpp_list]
        plist = [cpp[2] for cpp in arg_cpp_list]
        olist = None
        if len(vlist)>0:
            try:
                olist = lambda_op(vlist)
            except TypeError:
                # ill-typed, no answer
                olist = ["<type error>"]
        else:
            olist = ["<no answer>"]

        # pp: prob
        pp = sum(plist)/len(plist)
        return pp, olist
    
    TOP_K = 30
    MAX_R = 10
    tmp_op = arg_parsed_output[0]
    tmp_list = arg_parsed_output[1]
    
    if tmp_op==0:
        list_a = remove_value_duplication(
            [cpp for cpp in tmp_list if cpp[2]>0.5],
            arg_rendered_table
        )
        list_b = remove_value_duplication(
            [cpp for cpp in tmp_list if cpp[2]>0.0],
            arg_rendered_table
        )
    else:
        list_a = [cpp for cpp in tmp_list if cpp[2]>0.5]
        list_b = [cpp for cpp in tmp_list if cpp[2]>0.0]
    
    # directly construct the table
    lambda_op = denote_aggr(tmp_op)
    ret_answer_tables = []
    ret_answer_probs = []
    if len(list_a)==1:
        # A branch
        if len(list_b)>0:
            for i in range(len(list_b)):
                tmp_cpplist = [list_b[i]]
                tmp_prob, tmp_at = evaluate_answer_table(tmp_cpplist)
                ret_answer_tables.append(tmp_at)
                ret_answer_probs.append(tmp_prob)
        # else: no answer
    else:
        # B branch
        if len(list_b)>0:
            r = 0
            while len(ret_answer_tables)<TOP_K and r<MAX_R:
                r += 1
                combs = itertools.combinations(list_b, r)
                # every comb is a potential answer
                for comb in combs:
                    tmp_prob, tmp_at = evaluate_answer_table(comb)
                    ret_answer_tables.append(tmp_at)
                    ret_answer_probs.append(tmp_prob)
        # else: no answer
        
    # remove duplicate answer tables
    tmp_set_table = set()
    _ret_answer_tables = []
    _ret_answer_probs = []
    for i in range(len(ret_answer_tables)):
        p = ret_answer_tables[i]
        q = ret_answer_probs[i]
        if str(p) not in tmp_set_table:
            tmp_set_table.add(str(p))
            _ret_answer_tables.append(p)
            _ret_answer_probs.append(q)
    # sort
    inds = list(range(len(_ret_answer_tables)))
    sinds = sorted(inds, key=lambda x:_ret_answer_probs[x], reverse=True)
    ret_answer_tables = [_ret_answer_tables[p] for p in sinds]
    ret_answer_probs = [_ret_answer_probs[p] for p in sinds]
    
    return ret_answer_probs, ret_answer_tables