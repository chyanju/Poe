from ..dsl import AtomNode, ApplyNode

class Watson4VisOptimalReviewer():

    def __init__(self):
        pass

    def score(self, arg_maps, arg_question_tokens, arg_tprog, arg_rprog):
        # arg_maps comes in dict form {"const_val": ??, "aggr_func": ??, ...}

        # first construct reversed mapping from language constructs / table tokens to query tokens
        tmp_rmap = {}
        for dkey in arg_maps.keys():
            for p in arg_maps[dkey]:
                if p[1] not in tmp_rmap.keys():
                    tmp_rmap[p[1]] = []
                tmp_rmap[p[1]].append(p[0])

        tmp_score = {
            "triangle_alignment": self.indicator_triangle_alignment(tmp_rmap, arg_question_tokens, arg_rprog),
            "occams_razor": self.indicator_occams_razor(arg_rprog),
            "const_val_exact_match": self.indicator_const_val_exact_match(arg_question_tokens, arg_tprog),
        }

        tmp_total_score = sum(tmp_score.values())

        return tmp_total_score

    # ref: https://stackoverflow.com/questions/10823877/what-is-the-fastest-way-to-flatten-arbitrarily-nested-lists-in-python
    def flatten_list(self, arg_list):
        for i in arg_list:
            if isinstance(i, (list,tuple)):
                for j in self.flatten_list(i):
                    yield j
            else:
                yield i

    # a helper light-up procedure that tries to +1 to a matching token in a list
    # it tries to +1 for 0 values first
    # if no matching 0 values exists, it accumulates all to the first appearance
    # in-place operation
    # returns whether the token is found or not
    def light_up(self, arg_question_tokens, arg_qflags, arg_token):

        # first phase: find 0
        is_found = False
        for i in range(len(arg_question_tokens)):
            if arg_token==arg_question_tokens[i]:
                is_found = True
                if arg_qflags[i]==0:
                    arg_qflags[i] += 1
                    return True
            # for phrases like "more or less", split it first before matching
            for zz in str(arg_token).split(" "):
                if zz==arg_question_tokens[i]:
                    is_found = True
                    if arg_qflags[i]==0:
                        arg_qflags[i] += 1
                        # don't return, finish all
            # and return here instead after processing all the zz tokens
            if is_found:
                return True

        if not is_found:
            raise NotImplementedError("light_up procedure can't find a matching token for {} in list {}.".format(
                arg_token, arg_question_tokens
            ))
            # return False

        # if you reach here, it means:
        # - all the matching tokens have value >0
        # then find the first position and +1
        for i in range(len(arg_question_tokens)):
            # if arg_token.lower()==arg_question_tokens[i].lower():
            if str(arg_token).lower()==str(arg_question_tokens[i]).lower():
                arg_qflags[i] += 1
                return True

        # hmm... you can't reach here
        raise NotImplementedError("You can't reach here. Check your implementation.")

    def indicator_occams_razor(self, arg_rprog):
        # prefers the shorter solutions
        flatten_rprog = list(self.flatten_list(arg_rprog))
        return 1.0/len(flatten_rprog)

    def indicator_const_val_exact_match(self, arg_question_tokens, arg_tprog):
        # fixme: consider applying this to DSL inference directly
        # if a ConstVal of STRING is used, it has bonus points if it EXACTLY (case insensitive) matches part of the query
        def extract_str_const_val(tp):
            if isinstance(tp, AtomNode):
                if "@Str" in tp.data:
                    return [tp.data[:-4]]
                else:
                    return []
            elif isinstance(tp, ApplyNode):
                ret_list = []
                for p in tp.children:
                    ret_list += extract_str_const_val(p)
                return ret_list
            else:
                return []

        scv_list = extract_str_const_val(arg_tprog)
        # tmp_question = " ".join(arg_question_tokens).lower()
        tmp_question = " ".join([p for p in arg_question_tokens if isinstance(p, str)]).lower()
        tmp_score = 0.0
        tmp_hit, tmp_miss = 0.0, 0.0
        for p in scv_list:
            if p.lower() in tmp_question:
                tmp_hit += 1.0
            else:
                tmp_miss += 1.0
        if tmp_hit>0:
            tmp_score = 1.0
        # need to penalize the missed ones: you can't propose arbitrary constant strings
        if tmp_miss==0:
            return tmp_score
        else:
            return tmp_score * 1.0/tmp_miss

    def indicator_triangle_alignment(self, arg_rmap, arg_question_tokens, arg_rprog):
        # triangle alignment was semi-enforced during DSL inference phase
        # so here we perform partial alignment of it with certain complexity and repetition decay

        # i.e.,
        # - for query - table alignment, only make sure all keywords are covered at least once
        #   because query keywords can be matched by words (not phrase), and multiple edges may be created
        # - for table - program alignment, this should be exact one-to-one, extra alignment/edges will be penalized
        #   because table keywords are already word/phrase based

        # several notes and fixme:
        # 1. What happens if there are multiple repeated keywords in the query? E.g., what is the highest bar of the country with highest value?
        #    Usually the program will contain something like 2 `max` constructs, then you only need to find matching for them separately in the query.
        #    This addresses the issue.
        # 2. What happens if multiple words in a phrase map to the same table token? 
        #    E.g., "maximum entropy model" where "maximum" -> "maximum entropy model", "entropy" -> the same, "model" -> the same
        #    Then if you find "maximum" in the program, all of "maximum entropy model" from the query should be lit up once
        #    And if you find "maximum" in the program again, you should search for different appearance of "maximum entropy model" again
        #    and if you can't find it, apply punishments.
        

        flatten_rprog = list(self.flatten_list(arg_rprog))

        qflags = [0 for _ in range(len(arg_question_tokens))]
        # print("# init flatten_rprog: {}".format(flatten_rprog))
        # print("# init questions_tokens: {}".format(arg_question_tokens))
        # print("# init qflags: {}".format(qflags))
        # print("# arg_rmap: {}".format(arg_rmap))

        # light-up procedure
        for dtoken in flatten_rprog:
            if dtoken in arg_rmap.keys():
                # search and light up
                for p in arg_rmap[dtoken]:
                    if p is None:
                        # skip the special token (this is used for "==" which doesn't require any indicator)
                        continue
                    # for every query token, light them up once
                    self.light_up(arg_question_tokens, qflags, p)

        # print("# final qflags: {}".format(qflags))
        # the score is 1/max_light_up_value
        m = max(qflags) # maximum flag value
        n = sum([1 if p>1 else 0 for p in qflags]) # number of flag values larger than (>) 1
        z = sum([1 if p==1 else 0 for p in qflags]) # number of flag values equal to (==) 1
        if m==0:
            return 0
        else:
            if n==0:
                # add bonus: z/len(qflags) --- the more hit the better if there's no over-matching
                return 1.0/m + z/len(qflags)
            else:
                # base: (1/m) decided by the maximum flag value
                # decay (1/n): the more over-matched flags, the more punishment
                return 1.0/m * 1.0/n

