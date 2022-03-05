import pandas as pd
import numpy as np
import argparse
import os
import re
import json
import time
import pickle
from io import StringIO

import trinity.spec as S
from trinity.enumerator import LineSkeletonEnumerator
from trinity.interpreter import Watson4VisInterpreter, Watson4VisCoarseAbstractInterpreter, Watson4VisPartialInterpreter, Watson4VisEvalInterpreter
from trinity.decider import Example, Watson4VisDecider, NaiveDecider
from trinity.synthesizer import Watson4VisSynthesizer, Watson4VisPauseSynthesizer
from trinity.reviewer import Watson4VisOptimalReviewer, NaiveReviewer
from trinity.logger import get_logger
from trinity.utils.visqa import get_question_tokens, wrap_as_table, parse_value, append_const_val_postfix, \
                                infer_aggr_func, infer_num_func, infer_bool_func, infer_ind_func, infer_const_val
from trinity.utils.visqa_strategy import fallback_strategy_comparison

logger = get_logger('trinity')

def one_synthesis(arg_config):
    # performs one synthesis, returns a list of solutions

    enumerator = LineSkeletonEnumerator( spec=arg_config["spec"], cands=arg_config["skeleton_list"] )
    interpreter = Watson4VisInterpreter( spec=arg_config["spec"] )
    # coarse: perform skeleton level abstract interpretation, throws SkeletonAssertion
    coarse_abstract_interpreter = Watson4VisCoarseAbstractInterpreter()
    partial_interpreter = Watson4VisPartialInterpreter(
        interpreter=interpreter,
        abstract_interpreter=coarse_abstract_interpreter,
    )
    # eval_interpreter = Watson4VisEvalInterpreter( spec=arg_config["spec"] )

    decider = None
    if arg_config["mode"] == "full" or arg_config["mode"] == "abstract-only":
        decider = Watson4VisDecider( 
            interpreter=interpreter, 
            coarse_abstract_interpreter=coarse_abstract_interpreter,
            partial_interpreter=partial_interpreter,
            examples=[arg_config["example"]], 
            equal_output=interpreter.equal_tb,
            recorder=None,
        )
    elif arg_config["mode"] == "optimal-only":
        decider = NaiveDecider(
            interpreter=interpreter,
            examples=[arg_config["example"]],
            equal_output=interpreter.equal_tb,
        )
    else:
        raise NotImplementedError("Unsupported mode, got: {}.".format(arg_config["mode"]))

    synthesizer = Watson4VisPauseSynthesizer(
        enumerator=enumerator,
        decider=decider
    )

    # tprog: program with trinity data structure
    # rprog: readable/reproducible program in list s-expression
    tprogs = synthesizer.synthesize(timeout=arg_config["remaining_time"])
    return tprogs

def one_review(arg_config):
    # performs one review

    # start to review the candidate collections
    optimal_reviewer = None
    if arg_config["mode"] == "full" or arg_config["mode"] ==  "optimal-only":
        optimal_reviewer = Watson4VisOptimalReviewer()
    elif arg_config["mode"] == "abstract-only":
        optimal_reviewer = NaiveReviewer()
    else:
        raise NotImplementedError("Unsupported mode, got: {}.".format(arg_config["mode"]))

    sorted_candidate_collections = []
    # note: still include stop words to account for different operators, e.g., "most" -> "max"
    question_tokens = get_question_tokens(arg_config["query"], remove_stop_words=False, lemmatized=False, keywords=arg_config["table_keywords"])
    for p in arg_config["candidate_collections"]:
        tmp_score = optimal_reviewer.score(
            arg_config["alignment_map"], question_tokens, p["tprog"], p["rprog"]
        )
        sorted_candidate_collections.append((tmp_score,p))
    sorted_candidate_collections = sorted(sorted_candidate_collections, key=lambda x:x[0], reverse=True)

    # then print
    print("\n# ========== review report ========== #")
    for i in range(min(1000,len(sorted_candidate_collections))):
        p = sorted_candidate_collections[i]
        # print("# top-{}, score: {:.2f}, answer: {}".format(i+1, p[0], p[1]["example"].output.iloc[0,0]))
        print("# top-{}, score: {:.2f}, answer: {}".format(i+1, p[0], p[1]["example"].output.values.flatten().tolist()))
        # print("  # rprog: {}".format(p[1]["rprog"]))
        print("  # tprog: {}".format(p[1]["tprog"]))


if __name__ == "__main__":
    # logger.setLevel('DEBUG')
    logger.setLevel('CRITICAL')

    ap = argparse.ArgumentParser()
    ap.add_argument("-b", "--benchmark", default="Q0f532", type=str, help="6-charactered benchmark id, default: Q0f532")
    ap.add_argument("-d", "--dsl", default="test_min", type=str, choices=["test_min", "meta_visqa"], help="DSL definition to use, default: test_min")
    ap.add_argument("-s", "--skeletons", default="test_min", choices=["test_min", "visqa_simple", "visqa_normal"], help="skeleton list to use, default: test_min")
    ap.add_argument("-g", "--strategy", default="TaPas_C", choices=["TaPas_C"], help="candidate generation strategy to apply, default: TaPas_C")
    ap.add_argument("-f", "--fallback", default="none", type=str, choices=["none", "auto"], help="specify fallback strategy, default: none")
    ap.add_argument("-t", "--timeout", default=0, type=int, help="timeout in seconds, default: 0 (no timeout)")
    ap.add_argument("-m", "--mode", default="full", type=str, choices=["full", "optimal-only", "abstract-only"], help="ablation mode to use, default: full")
    ap.add_argument("--expected-only", action="store_true", help="whether or not to only process the expected answer (for debugging), default: False")
    args = ap.parse_args()
    print("# parsed arguments: {}".format(args))
    if args.timeout<=0:
        args.timeout = None

    # load dataset
    with open("./benchmarks/VisQA/shared/tapas_on_visqa_dataset.pkl", "rb") as f:
        dt = pickle.load(f)
    # build reverse index
    short_id_to_id = {dt[i]["short_id"]:i for i in range(len(dt))}

    # load benchmark
    print("# loading benchmark...")
    benchmark_id = short_id_to_id[args.benchmark]
    benchmark = dt[benchmark_id]
    expected_example = Example( input=[benchmark["rendered_table"]], output=benchmark["expected_output"], query=benchmark["query"] )
    # note: we only use 1 input in this client
    candidate_examples = [
        Example( input=[benchmark["rendered_table"]], output=wrap_as_table(p), query=benchmark["query"] )
        for p in benchmark["candidate_outputs"][args.strategy]
    ]
    # debug: trim the candidate list for quick debugging
    candidate_examples = candidate_examples[:30]

    # get a keyword list for the table
    # note: this is a list of words, not values or phrases, so you need to split as much as possible
    # also-note: keywords should be made string
    table_values = [parse_value(str(p)) for p in benchmark["rendered_table"].values.flatten().tolist() + benchmark["rendered_table"].columns.values.flatten().tolist()]
    table_keywords = []
    for p in table_values:
        if isinstance(p, str):
            table_keywords += p.split(" ")
        else:
            table_keywords.append(str(p))
    table_keywords = set([p.lower() for p in table_keywords]) - {"the", "a", "an", "of"} # remove super keywords
    print("# table keywords: {}".format(table_keywords))

    print("# input type: {}".format(expected_example.input[0].dtypes.tolist()))
    print("# input is:\n{}".format(expected_example.input[0]))
    print("# query is: {}".format(expected_example.query))
    print("# expected output type:{}".format(expected_example.output.dtypes.tolist()))
    print("# expected output is:\n{}".format(expected_example.output))

    # load dsl spec
    spec = None
    if args.dsl.startswith("test"):
        spec = S.parse_file("./benchmarks/VisQA/tests/{}_{}.dsl".format(args.benchmark, args.dsl))
    elif args.dsl == "meta_visqa":

        if args.mode == "full" or args.mode == "optimal-only":
            # automatically infer DSL (for full mode and optimal-only mode)
            with open("./dsls/meta_visqa.tyrell", "r") as f:
                str_dsl = f.read()
            tmp_const_val, tmp_cmap = infer_const_val(expected_example.query, expected_example.input[0], arg_keywords=table_keywords)
            tmp_aggr_func, tmp_amap = infer_aggr_func(expected_example.query, arg_keywords=table_keywords)
            tmp_num_func, tmp_nmap = infer_num_func(expected_example.query, arg_keywords=table_keywords)
            tmp_bool_func, tmp_bmap = infer_bool_func(expected_example.query, tmp_const_val, arg_keywords=table_keywords)
            tmp_ind_func, tmp_imap = infer_ind_func(expected_example.query, arg_keywords=table_keywords)
            for x in [tmp_const_val, tmp_aggr_func, tmp_num_func, tmp_bool_func, tmp_ind_func]:
                x += ["<NULL>"]
            tmp_alignment_map = {
                "const_val": tmp_cmap, "aggr_func": tmp_amap, "num_func": tmp_nmap, 
                "bool_func": tmp_bmap, "ind_func": tmp_imap,
            }
            print("# inferred DSL terminals:")
            print("  # ConstVal: {}".format(tmp_const_val))
            print("     # cmap: {}".format(tmp_cmap))
            print("  # AggrFunc: {}".format(tmp_aggr_func))
            print("     # amap: {}".format(tmp_amap))
            print("  # NumFunc: {}".format(tmp_num_func))
            print("     # nmap: {}".format(tmp_nmap))
            print("  # BoolFunc: {}".format(tmp_bool_func))
            print("     # bmap: {}".format(tmp_bmap))
            print("  # IndFunc: {}".format(tmp_ind_func))
            print("     # imap: {}".format(tmp_imap))
            # fill in DSL
            str_dsl = str_dsl.replace("<ConstVal>", "enum ConstVal {{ {} }}".format(",".join(['"{}"'.format(p) for p in tmp_const_val])))
            str_dsl = str_dsl.replace("<AggrFunc>", "enum AggrFunc {{ {} }}".format(",".join(['"{}"'.format(p) for p in tmp_aggr_func])))
            str_dsl = str_dsl.replace("<NumFunc>", "enum NumFunc {{ {} }}".format(",".join(['"{}"'.format(p) for p in tmp_num_func])))
            str_dsl = str_dsl.replace("<BoolFunc>", "enum BoolFunc {{ {} }}".format(",".join(['"{}"'.format(p) for p in tmp_bool_func])))
            str_dsl = str_dsl.replace("<IndFunc>", "enum IndFunc {{ {} }}".format(",".join(['"{}"'.format(p) for p in tmp_ind_func])))
            # parse the DSL
            spec = S.parse(str_dsl)

        elif args.mode == "abstract-only":
            # load the full DSL and replace the ConstVal enum
            with open("./dsls/visqa_abstract_only.tyrell", "r") as f:
                str_dsl = f.read()
            # construct ConstVal using all table values
            unique_table_values = list(set(table_values))
            tmp_const_val = [append_const_val_postfix(p) for p in unique_table_values]
            tmp_cmap = [(None,p) for p in unique_table_values]
            tmp_aggr_func, tmp_amap = [],[]
            tmp_num_func, tmp_nmap = [],[]
            tmp_bool_func, tmp_bmap = [],[]
            tmp_ind_func, tmp_imap = [],[]
            # don't need NULL here
            # for x in [tmp_const_val, tmp_aggr_func, tmp_num_func, tmp_bool_func, tmp_ind_func]:
            #     x += ["<NULL>"]
            tmp_alignment_map = {
                "const_val": tmp_cmap, "aggr_func": tmp_amap, "num_func": tmp_nmap, 
                "bool_func": tmp_bmap, "ind_func": tmp_imap,
            }
            print("# inferred DSL terminals:")
            print("  # ConstVal: {}".format(tmp_const_val))
            print("     # cmap: {}".format(tmp_cmap))
            # fill in DSL
            str_dsl = str_dsl.replace("<ConstVal>", "enum ConstVal {{ {} }}".format(",".join(['"{}"'.format(p) for p in tmp_const_val])))
            # parse the DSL
            spec = S.parse(str_dsl)

        else:
            raise NotImplementedError("Unsupported mode, got: {}.".format(args.mode))
    else:
        raise Exception("You should not reach here.")

    # ====================================== #
    # ==== fallback strategy zone (pre) ==== #
    # ====================================== #
    if args.fallback == "auto":
        # auto trigger
        # ==== categorized fallback strategy triggering === #

        # for comparison
        tmp_all_answers = [p for p in benchmark["candidate_outputs"][args.strategy]]
        # print("# [debug] tmp_all_answers: {}".format(tmp_all_answers))

        # comparison fallback policy
        if "-" in tmp_num_func or "diff" in tmp_num_func:
            print("# ====> fallback strategy (comparison) is triggered")
            # should trigger comparison
            high_values, low_values = fallback_strategy_comparison(benchmark["rendered_table"], [p[1] for p in tmp_cmap])
            print("  # [debug] high_values: {}".format(high_values))
            # filter out answers that already exist
            high_values = [p for p in high_values if [p[0]] not in tmp_all_answers] # wrap as repr_answer to compare
            low_values = [p for p in low_values if p[1]>1 and p[0] not in tmp_all_answers] # wrap as repr_answer to compare
            print("  # [debug] high_values (no dup.): {}".format(high_values))
            print("  # [debug] filtered low_values (no dup.): {}".format(low_values))
            tmp_all_answers += [[p[0]] for p in high_values] + [p[0] for p in low_values]
            # fixme: remove duplication within all_fallback_answers
            if len(high_values)>0:
                # add the fallback answers to candidates
                for p in high_values:
                    candidate_examples.append(
                        Example( input=[benchmark["rendered_table"]], output=wrap_as_table([p[0]]), query=benchmark["query"] )
                    )
            if len(low_values)>0:
                for p in low_values:
                    candidate_examples.append(
                        Example( input=[benchmark["rendered_table"]], output=wrap_as_table([p[0]]), query=benchmark["query"] )
                    )
                print("  # [debug] fallback (comparison) answers are added".format(len(high_values)))
            # else: nothing is added
            # fixme: currently we don't consider low_values
        
        # selection fallback policy
        tmp_selection_query = benchmark["query"].lower().strip()
        tmp_selection_flag = bool(re.search(r".+? or .+?$", tmp_selection_query))
        if tmp_selection_flag:
            print("# ====> fallback strategy (selection) is triggered")
            # should trigger selection
            # simply add const values as extra answer
            tmp_selection_answers = list(set([p[1] for p in tmp_cmap]))
            print("  # [debug] fallback (selection) answers: {}".format(tmp_selection_answers))
            # filter out answers that already exist
            tmp_selection_answers = [p for p in tmp_selection_answers if [p] not in tmp_all_answers] # wrap as repr_answer to compare
            print("  # [debug] fallback (selection) answers (no dup.): {}".format(tmp_selection_answers))
            tmp_all_answers += [[p] for p in tmp_selection_answers]
            # fixme: remove duplication within all_fallback_answers
            for p in tmp_selection_answers:
                candidate_examples.append(
                    Example( input=[benchmark["rendered_table"]], output=wrap_as_table([p]), query=benchmark["query"] )
                )
            print("  # [debug] {} fallback (selection) answers are added".format(len(tmp_selection_answers)))

        # assertion fallback policy
        # TODO: you need an Assert component and just simply wrap it on top of comparison skeletons, will do later
    # else: do nothing
    
    # ====================================== #
    # ====================================== #

    # load skeleton
    print("# loading skeleton list...")
    skeleton_list = None
    if args.skeletons.startswith("test"):
        with open("./benchmarks/VisQA/tests/{}_{}_skeletons.json".format(args.benchmark, args.skeletons), "r") as f:
            skeleton_list = json.load(f)
    else:
        with open("./skeletons/{}_skeletons.json".format(args.skeletons), "r") as f:
            skeleton_list = json.load(f)

    # input("PRESS To START")
    start_time = time.time()
    if args.expected_only:
        # debugging mode, replace with expected example
        candidate_examples = [expected_example]

    candidate_collections = []
    # start processing all the candidate examples
    print("\n# ========== candidate program report ========== #")
    for i in range(len(candidate_examples)):
        curr_time = time.time() - start_time

        tmp_example = candidate_examples[i]
        print("# (t={:.2f}) i={}, candidate={}".format(curr_time, i, tmp_example.output.to_dict("records")))
        tmp_synth_config = { 
            "spec": spec, "skeleton_list": skeleton_list, "example": tmp_example, "mode": args.mode,
            "remaining_time": None if args.timeout is None else args.timeout - curr_time, # note: this gives the remaining time
        }

        # ==== synthesis call ==== #
        tprogs = one_synthesis(tmp_synth_config)
        eval_interpreter = Watson4VisEvalInterpreter( spec=spec )

        print("  # found {} program(s)".format(len(tprogs)))
        for p in tprogs:
            print("    # {}".format(p))
            eval_interpreter._colname_count = 0
            ret_tb, rprog = eval_interpreter.eval(p, expected_example.input)
            print("      --> {}".format(rprog))
            # add program into item
            candidate_collections.append({
                "example": tmp_example,
                "tprog": p,
                "rprog": rprog,
            })

        # note-important: this should be later than candidate append, otherwise no candidate will be added
        curr_time = time.time() - start_time
        if args.timeout is not None and curr_time>args.timeout:
            print("---------- timeout ----------")
            break

    # ======================================= #
    # ==== fallback strategy zone (post) ==== #
    # ======================================= #
    if args.fallback == "auto" and len(candidate_collections)==0:
        print("# ====> fallback strategy (retrieval) is triggered because no explanation is found so far")
        # order table_values first
        # then try to solve them

        # priritize numbers
        retrieval_values = sorted(list(set(table_values)), key=lambda x:str(x), reverse=False)
        print("# ====> retrieval_values: {}".format(retrieval_values))

        for i in range(len(retrieval_values)):
            # fixme: skip duplicated answers

            curr_time = time.time() - start_time

            tmp_example = Example(
                input=[benchmark["rendered_table"]],
                output=wrap_as_table([retrieval_values[i]]),
                query=benchmark["query"]
            )
            print("# (t={:.2f}) fallback i={}, candidate={}".format(curr_time, i, tmp_example.output.to_dict("records")))
            tmp_config = {
                "spec": spec, "skeleton_list": skeleton_list, "example": tmp_example, "mode": args.mode,
                "remaining_time": None if args.timeout is None else args.timeout - curr_time, # note: this gives the remaining time
            }

            # ==== synthesis call ==== #
            tprogs = one_synthesis(tmp_config)
            eval_interpreter = Watson4VisEvalInterpreter( spec=spec )

            print("  # found {} program(s)".format(len(tprogs)))
            for p in tprogs:
                print("    # {}".format(p))
                eval_interpreter._colname_count = 0
                ret_tb, rprog = eval_interpreter.eval(p, expected_example.input)
                print("      --> {}".format(rprog))
                # add program into item
                candidate_collections.append({
                    "example": tmp_example,
                    "tprog": p,
                    "rprog": rprog,
                })

            # note-important: this should be later than candidate append, otherwise no candidate will be added
            curr_time = time.time() - start_time
            if args.timeout is not None and curr_time>args.timeout:
                print("---------- timeout ----------")
                break

    # else: do nothing

    if args.fallback == "auto" and len(candidate_collections)==0:
        print("# ====> fallback strategy (original) is triggered")
        print("# ====> use the original TaPas output as answer, which is: {}".format(benchmark["candidate_outputs"]["TaPas_original"]))
    # ======================================= #
    # ======================================= #

    tmp_review_config = {
        "query": expected_example.query,
        "table_keywords": table_keywords,
        "candidate_collections": candidate_collections,
        "alignment_map": tmp_alignment_map,
        "mode": args.mode
    }
    one_review(tmp_review_config)
    



