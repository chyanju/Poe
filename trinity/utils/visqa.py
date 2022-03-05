import subprocess
import itertools
import json
import copy
import simplejson
import itertools
import numpy as np
import pandas as pd
import nltk
import spacy
nlp = spacy.load("en_core_web_sm")
from xml.dom import minidom
from PIL import ImageColor
from nltk.corpus import wordnet as wn

# ========================================= #
# ========== VisQA related utils ========== #
# ========================================= #

# this translates a given hex color to its closest rgb hue color
# picked from the given unambiguous color set
DEFAULT_HUE_PALETTE = {
    "red": (195, 33, 49), "orange": (232,95,67), "yellow": (247, 230, 59),
    "green": (23, 118, 62), "blue": (43, 88, 162), "purple": (66, 43, 130),
    "pink": (244, 198, 219), "brown": (113, 80, 46), "gray": (114, 119, 119),
    "black": (0, 0, 0), "white": (255, 255, 255),
}
DEFAULT_NORMALIZED_HUE_PALETTE = {
    k: (v[0]/256, v[1]/256, v[2]/256)
    for k,v in DEFAULT_HUE_PALETTE.items()
}
# this gets the color when the spec doesn't have the color key
def get_default_color_hue(arg_color):
    tmp_rgb = ImageColor.getrgb(arg_color)
    tmp_nrgb = (tmp_rgb[0]/256, tmp_rgb[1]/256, tmp_rgb[2]/256)
    # then compute color similarity by euclidean distance (normalized)
    tmp_dist = {
        k: ( (v[0]-tmp_nrgb[0])**2 + (v[1]-tmp_nrgb[1])**2 + (v[2]-tmp_nrgb[2])**2 )**0.5
        for k,v in DEFAULT_NORMALIZED_HUE_PALETTE.items()
    }
    # then find out the minimum
    tmp_hue = min(tmp_dist, key=tmp_dist.get)
    return tmp_hue

# reference for vega default color schemes: https://vega.github.io/vega-lite/docs/scale.html#scheme
VEGA_HUE_PALETTE = {
    "tableau10": {
        "#4c78a8": "blue", "#f58518": "orange", "#e45756": "red", "#72b7b2": "teal", "#54a24b": "green",
        "#eeca3b": "yellow", "#b279a2": "purple", "#ff9da6": "pink", "#9d755d": "brown", "#bab0ac": "gray",
    }
}
def get_color_hue(arg_color, arg_type):
    if arg_type == "nominal":
        return VEGA_HUE_PALETTE["tableau10"][arg_color]
    else:
        raise NotImplementedError("Unsupported type for color scheme, got: {}.".format(arg_type))

def parse_value(arg_value):
    # strange bug: − and - are different, which will cause the negative number parsing to fail
    p = arg_value.replace("−", "-")
    # try to parse as int first, then float, then string
    try:
        # consider thousand separator
        return int(p.replace(",",""))
    except ValueError as e:
        try:
            return float(p.replace(",",""))
        except ValueError as e:
            if isinstance(p, str):
                return p.strip()
            else:
                raise Exception("arg_value should be a int/float/string, got: {}".format(type(arg_value)))

def parse_property(arg_property):
    if isinstance(arg_property, str):
        return arg_property.strip()
    else:
        raise Exception("arg_property should be a string, got: {}".format(type(arg_property)))

def normalize_table(arg_table):
    # this infers the correct type for every column/cell
    # also replaces some corner tokens into its correct form, e.g., "-" into null in some cases
    ret_table = arg_table.copy(deep=True)
    nr, nc = ret_table.shape
    for i in range(nr):
        for j in range(nc):
            # fixme: this may be wrong in general, but correct in VisQA dataset
            if ret_table.iloc[i,j] == "-":
                ret_table.iloc[i,j] = None
                pass
            elif np.issubdtype(type(ret_table.iloc[i,j]), np.inexact):
                # skip known type
                pass
            elif np.issubdtype(type(ret_table.iloc[i,j]), np.integer):
                # skip known type
                pass
            elif isinstance(ret_table.iloc[i,j], str):
                ret_table.iloc[i,j] = parse_value(ret_table.iloc[i,j])
    ret_table = ret_table.infer_objects()
    return ret_table

def wrap_as_table(arg_repr_answer):
    # wrap a repr answer into a table and normalize
    return normalize_table(pd.DataFrame.from_records(
        np.asarray([arg_repr_answer]).T, columns=["ANSWER"],
    ))

# note: this requires the node command `vl2svg` ready, see README for configuration
# note: the spec provided should have the data embedded into the "data" -> "values" field
# returns a string of the returned svg
def vl_to_svg(arg_spec):
    p = subprocess.run(["npx", "vl2svg"], stdout=subprocess.PIPE, input=simplejson.dumps(arg_spec, ignore_nan=True), encoding="utf-8")
    assert p.returncode==0, \
        "Exception running command `vl2svg`, please check that you have the vega environment set correctly." + \
        "\nOutput is: {}\n".format(p.stdout) + \
        "\nError message is: {}\n".format(p.stderr)
    return p.stdout

# extended version 0: this separates the spec into smaller ones with different column properties and stack them back later
#                     in order to preserve the column information; example seen in "bar_grouped.json"
# note: this is an experimental feature
# this returns a list of ({field:value}, svg)
def vl_to_grouped_svg(arg_spec):
    rlist, clist = [], []
    if "row" in arg_spec["encoding"].keys():
        # should enable ex0 mode
        tmp_row_field = arg_spec["encoding"]["row"]["field"]
        rlist = list(set([v[tmp_row_field] for v in arg_spec["data"]["values"]]))
    if "column" in arg_spec["encoding"].keys():
        # should enable ex0 mode
        tmp_col_field = arg_spec["encoding"]["column"]["field"]
        clist = list(set([v[tmp_col_field] for v in arg_spec["data"]["values"]]))

    if len(rlist)==0 and len(clist)==0:
        # fall back to basic mode
        return [({}, vl_to_svg(arg_spec))]
    else:
        # extended mode
        if len(rlist)==0:
            rlist = [None]
        if len(clist)==0:
            clist = [None]
        tmp_rclist = itertools.product(rlist, clist)
        
        retlist = []
        for v in tmp_rclist:
            # for every column (group) value, get the svg separately
            tmp_spec = copy.deepcopy(arg_spec)

            tmp_spec["encoding"].pop("row", None)
            tmp_spec["encoding"].pop("column", None)
            if "transform" not in tmp_spec.keys():
                tmp_spec["transform"] = []
            # fixme: if other transformations change the name, this may result in errors / unwanted results
            # insert the filter operation at first
            tmp_dict = {}
            if v[0] is not None:
                if isinstance(v[0], str):
                    tmp_spec["transform"] = [{"filter":"datum[\"{}\"] == \"{}\"".format(tmp_row_field, v[0])}] + tmp_spec["transform"]
                else:
                    # fixme: potentially dangerous, e.g., None
                    tmp_spec["transform"] = [{"filter":"datum[\"{}\"] == {}".format(tmp_row_field, v[0])}] + tmp_spec["transform"]
                tmp_dict[tmp_row_field] = v[0]
            if v[1] is not None:
                if isinstance(v[1], str):
                    tmp_spec["transform"] = [{"filter":"datum[\"{}\"] == \"{}\"".format(tmp_col_field, v[1])}] + tmp_spec["transform"]
                else:
                    # fixme: potentially dangerous, e.g., None
                    tmp_spec["transform"] = [{"filter":"datum[\"{}\"] == {}".format(tmp_col_field, v[1])}] + tmp_spec["transform"]
                tmp_dict[tmp_col_field] = v[1]

            p = subprocess.run(["npx", "vl2svg"], stdout=subprocess.PIPE, input=simplejson.dumps(tmp_spec, ignore_nan=True), encoding="utf-8")
            assert p.returncode==0 and len(p.stdout.strip())>0, \
                "Exception running command `vl2svg`, please check that you have the vega environment set correctly." + \
                "\nReturnCode is: {}\n".format(p.returncode) + \
                "\nOutput is: {}\n".format(p.stdout) + \
                "\nError message is: {}\n".format(p.stderr)

            retlist.append(( tmp_dict , p.stdout ))
        return retlist   

# note: this requires the ARIA labels in svg, otherwise no information will be extracted
#       to attach the ARIA info automatically, you need vega-lite 5.1.1+ version, see README
def svg_to_table(arg_svg, arg_spec):
    doc = minidom.parseString(arg_svg)
    all_paths = doc.getElementsByTagName('path')
    # only keep those paths where role is "graphics-symbol"
    all_objs = [p for p in all_paths if p.getAttribute("role") == "graphics-symbol"]
    # then parse the attributes
    parsed_objs = []
    for p in all_objs:
        tmp_obj = {}
        # parse normal attribute
        tmp_str = p.getAttribute("aria-label")
        tmp_pvs = tmp_str.split(";")
        for q in tmp_pvs:
            tmp_pv = q.split(":")
            tmp_p = parse_property(tmp_pv[0])
            tmp_v = parse_value(tmp_pv[1])
            # print("# tmp_v is: {}, type is: {}".format(tmp_v, type(tmp_v)))
            tmp_obj[tmp_p] = tmp_v
        # parse special attribute: color
        tmp_color = p.getAttribute("fill").lower()
        if tmp_color=="":
            # if no color, try to get the "stroke" attribute (for some cases for line mark)
            tmp_color = p.getAttribute("stroke").lower()
        if "color" not in arg_spec["encoding"].keys():
            # no color scheme specified, use the general color hue
            tmp_hue = get_default_color_hue(tmp_color)
        elif "scale" in arg_spec["encoding"]["color"].keys():
            if "range" in arg_spec["encoding"]["color"]["scale"].keys():
                # it's using customized color, use the general color hue
                tmp_hue = get_default_color_hue(tmp_color)
            else:
                raise NotImplementedError("Unsupported color spec, got: {}.".format(arg_spec["encoding"]["color"]))
        else:
            # a color scheme is implied, use the vega color hue
            tmp_hue = get_color_hue(tmp_color, arg_spec["encoding"]["color"]["type"])
        tmp_obj["color"] = tmp_hue
        # add to object list
        parsed_objs.append(tmp_obj)
    # return pd.DataFrame.from_records(parsed_objs).convert_dtypes()
    ret_table =  pd.DataFrame.from_records(parsed_objs)
    ret_table = normalize_table(ret_table)
    return ret_table

# paired with `vl_to_grouped_svg`
def grouped_svg_to_table(arg_grouped_svg, arg_spec):
    tmp_tables = []
    for q in arg_grouped_svg:
        arg_svg = q[1]
        arg_group = q[0]

        doc = minidom.parseString(arg_svg)
        all_paths = doc.getElementsByTagName('path')
        # only keep those paths where role is "graphics-symbol"
        all_objs = [p for p in all_paths if p.getAttribute("role") == "graphics-symbol"]
        # then parse the attributes
        parsed_objs = []
        for p in all_objs:
            tmp_obj = {}
            # parse normal attribute
            tmp_str = p.getAttribute("aria-label")
            # extend the aria label to include the grouping info
            for k,v in arg_group.items():
                tmp_str += "; {}: {}".format(k, v)
            tmp_pvs = tmp_str.split(";")
            for q in tmp_pvs:
                tmp_pv = q.split(":")
                tmp_p = parse_property(tmp_pv[0])
                tmp_v = parse_value(tmp_pv[1])
                # print("# tmp_v is: {}, type is: {}".format(tmp_v, type(tmp_v)))
                tmp_obj[tmp_p] = tmp_v
            # parse special attribute: color
            tmp_color = p.getAttribute("fill").lower()
            if tmp_color=="":
                # if no color, try to get the "stroke" attribute (for some cases for line mark)
                tmp_color = p.getAttribute("stroke").lower()
            if "color" not in arg_spec["encoding"].keys():
                # no color scheme specified, use the general color hue
                tmp_hue = get_default_color_hue(tmp_color)
            elif "scale" in arg_spec["encoding"]["color"].keys():
                if "range" in arg_spec["encoding"]["color"]["scale"].keys():
                    # it's using customized color, use the general color hue
                    tmp_hue = get_default_color_hue(tmp_color)
                else:
                    raise NotImplementedError("Unsupported color spec, got: {}.".format(arg_spec["encoding"]["color"]))
            else:
                # a color scheme is implied, use the vega color hue
                tmp_hue = get_color_hue(tmp_color, arg_spec["encoding"]["color"]["type"])
            tmp_obj["color"] = tmp_hue
            # add to object list
            parsed_objs.append(tmp_obj)
        # return pd.DataFrame.from_records(parsed_objs).convert_dtypes()
        ret_table =  pd.DataFrame.from_records(parsed_objs)
        ret_table = normalize_table(ret_table)
        tmp_tables.append(ret_table)
    # stack all tables together
    final_table = pd.concat(tmp_tables, axis=0, join="outer", ignore_index=True)
    return final_table

# ========================================= #
# ============= DSL inference ============= #
# ========================================= #

def get_question_tokens(arg_question, keywords=set(), remove_stop_words=True, lemmatized=True):
    # keywords: some keywords that should be excluded from stopwords, i.e., these keywords should not be removed
    # nlp_question = nlp(arg_question.lower())
    # hint: .lemma_ is automatically lowered, so actually there's no need to call .lower(), but whatever
    nlp_question = nlp(arg_question.replace("("," ( ").replace(")"," ) "))
    nlp_question = [p for p in nlp_question if not p.is_punct] # don't consider punctuations
    if remove_stop_words:
        if lemmatized:
            # no stop words, with lemmatization
            # return [parse_value(str(p.lemma_)) for p in nlp_question if not p.is_stop]
            # this considers potential abbreviations, e.g., CA (which is a stop word in spacy)
            return [parse_value(str(p.lemma_).lower()) for p in nlp_question if p.is_upper or (not p.is_stop) or str(p).lower() in keywords]
        else:
            # no stop words, no lemmatization
            # return [parse_value(str(p)) for p in nlp_question if not p.is_stop]
            # this considers potential abbreviations, e.g., CA (which is a stop word in spacy)
            return [parse_value(str(p).lower()) for p in nlp_question if p.is_upper or (not p.is_stop) or str(p).lower() in keywords]
    else:
        if lemmatized:
            # with stop words, with lemmatization
            return [parse_value(str(p.lemma_).lower()) for p in nlp_question]
        else:
            # with stop words, no lemmatization
            return [parse_value(str(p).lower()) for p in nlp_question]

# (deprecated)
def relax_str(arg_str):
    stoks = "~!@#$%^&*()_+`-=[]{}|;:'\",<.>/?\\"
    tmp_str = arg_str.lower()
    for st in stoks:
        tmp_str = tmp_str.replace(st, " {} ".format(st))
    tmp_str = " ".join(list(filter(None, tmp_str.split(" "))))
    return tmp_str

# (deprecated)
def remove_stok(arg_str):
    stoks = "~!@#$%^&*()_+`-=[]{}|;:'\",<.>/?\\"
    tmp_str = arg_str
    for st in stoks:
        tmp_str = tmp_str.replace(st, "")
    tmp_str = " ".join(list(filter(None, tmp_str.split(" "))))
    return tmp_str

# fetch all related words (according to definition)
# see an example: http://wordnetweb.princeton.edu/perl/webwn?s=worse&sub=Search+WordNet&o2=&o0=1&o8=1&o1=1&o7=&o5=&o9=&o6=&o3=&o4=&h=
def fetch_wn_related_words(arg_word):
    ret_words = []
    if isinstance(arg_word, str):
        tmp_tokens = wn.synsets(arg_word)
        for dtoken in tmp_tokens:
            for ddtoken in dtoken.lemmas():
                ret_words.append(ddtoken.name())
    # else: nothing happens, return an empty list
    return list(set(ret_words))

# decides under what score can the current word be accepted as a related word
MAXIMUM_EDIT_PERCENTAGE = 0.5
# variant of edit distance with relative metric
def edit_percentage(arg_word0, arg_word1):
    tmp_score = nltk.edit_distance(arg_word0, arg_word1)
    n0 = len(arg_word0)
    n1 = len(arg_word1)
    return tmp_score/max(n0,n1)

def ll_matching(arg_sub, arg_main):
    # see if arg_sub is a sublist of arg_main
    for i in range(len(arg_main)):
        if i+len(arg_sub)>len(arg_main):
            # early stop: overflow, impossible to match anymore
            return False
        tmp_found = True
        for j in range(len(arg_sub)):
            if arg_sub[j]!=arg_main[i+j]:
                tmp_found = False
                break
        if tmp_found:
            return True
        else:
            continue
    # if you are here, then nothing is found
    return False

AGGR_FUNC_INDICATORS = {
    "min": ["min", "minimum", "least", "smallest", "fewest", "lowest", "slightest", "shortest", "best", "worst"],
    "max": ["max", "maximum", "most", "biggest", "largest", "highest", "top", "longest", "greatest", "best", "worst"],
    "sum": ["sum", "total", "all", "summed", "add", "overall"],
    "mean": ["mean", "average", "averaged"],
    "count": ["total", "count", "number", "how many"],
}
# template based inference
def infer_aggr_func(arg_question, arg_keywords=set()):
    ret_aggr_func = []
    ret_map = [] # keeps track of question token - aggr token mapping

    # note: this INCLUDES stop words
    question_tokens = get_question_tokens(arg_question, keywords=arg_keywords, remove_stop_words=False, lemmatized=False)
    for dop in AGGR_FUNC_INDICATORS.keys():
        for indw in AGGR_FUNC_INDICATORS[dop]:
            # get tokens because indicators may be a phrase, e.g., "how many" -> "count"
            indw_tokens = get_question_tokens(indw, keywords=arg_keywords, remove_stop_words=False, lemmatized=False)
            if ll_matching(indw_tokens, question_tokens):
                ret_aggr_func.append(dop)
                ret_map.append((indw,dop))
                break

    return list(set(ret_aggr_func)), list(set(ret_map))

NUM_FUNC_INDICATORS = {
    "+": [],
    "-": ["more", "less", "higher", "different", "minus", "subtract", "larger", "longer", "bigger", "shorter"],
    "*": [],
    "/": [],
    "diff": ["more or less", "difference", "different", "change"],
}
# template based inference
def infer_num_func(arg_question, arg_keywords=set()):
    ret_num_func = []
    ret_map = [] # keeps track of question token - num token mapping

    # note: this INCLUDES stop words
    question_tokens = get_question_tokens(arg_question, keywords=arg_keywords, remove_stop_words=False, lemmatized=False)
    for dop in NUM_FUNC_INDICATORS.keys():
        for indw in NUM_FUNC_INDICATORS[dop]:
            # correction for "more or less"
            if "more or less" in arg_question:
                if indw in ["more", "less"]:
                    continue
            # get tokens because indicators may be a phrase, e.g., "how many" -> "count"
            indw_tokens = get_question_tokens(indw, keywords=arg_keywords, remove_stop_words=False, lemmatized=False)
            if ll_matching(indw_tokens, question_tokens):
                ret_num_func.append(dop)
                ret_map.append((indw,dop))
                break

    return list(set(ret_num_func)), list(set(ret_map))

BOOL_FUNC_INDICATORS = {
    "<": ["less", "smaller", "fewer"],
    "<=": ["less", "smaller", "fewer"],
    "==": ["equal"],
    ">=": ["more", "larger", "greater"],
    ">": ["more", "larger", "greater"],
    "!=": ["unequal", "not equal", "not"],
    "setin": ["include", "included", "including"],
}
# template based inference
def infer_bool_func(arg_question, arg_const_val, arg_keywords=set()):
    # there will always be an "=="
    ret_bool_func = ["=="]
    ret_map = [(None,"==")] # keeps track of question token - bool token mapping

    # note: this INCLUDES stop words
    question_tokens = get_question_tokens(arg_question, keywords=arg_keywords, remove_stop_words=False, lemmatized=False)
    for dop in BOOL_FUNC_INDICATORS.keys():
        for indw in BOOL_FUNC_INDICATORS[dop]:
            # get tokens because indicators may be a phrase, e.g., "how many" -> "count"
            indw_tokens = get_question_tokens(indw, keywords=arg_keywords, remove_stop_words=False, lemmatized=False)
            if ll_matching(indw_tokens, question_tokens):
                ret_bool_func.append(dop)
                ret_map.append((indw,dop))
                break

    # also infer the "setin" operator based on arg_const_val
    for p in arg_const_val:
        if "@Set" in p:
            ret_bool_func.append("setin")
            ret_map.append((None,"setin"))
            break

    return list(set(ret_bool_func)), list(set(ret_map))

IND_FUNC_INDICATORS = {
    "eqmin": ["min", "minimum", "least", "smallest", "fewest", "lowest", "slightest", "shortest", "best", "worst"],
    "eqmax": ["max", "maximum", "most", "biggest", "largest", "highest", "top", "longest", "greatest", "best", "worst"],
}
# template based inference
def infer_ind_func(arg_question, arg_keywords=set()):
    ret_ind_func = []
    ret_map = [] # keeps track of question token - ind token mapping

    # note: this INCLUDES stop words
    question_tokens = get_question_tokens(arg_question, keywords=arg_keywords, remove_stop_words=False, lemmatized=False)
    for dop in IND_FUNC_INDICATORS.keys():
        for indw in IND_FUNC_INDICATORS[dop]:
            # get tokens because indicators may be a phrase, e.g., "how many" -> "count"
            indw_tokens = get_question_tokens(indw, keywords=arg_keywords, remove_stop_words=False, lemmatized=False)
            if ll_matching(indw_tokens, question_tokens):
                ret_ind_func.append(dop)
                ret_map.append((indw,dop))
                break

    return list(set(ret_ind_func)), list(set(ret_map))

# linguistics based inference
# note: arg_input is a single input, so just pass example.input[0] as argument since we are only using 1 input in this client
# note: we assume table `arg_input` is normalized
def tu_matching(arg_token, arg_unit):
    # helper function that does token-unit matching
    # as long as the token appears in one of the tokens from the unit, then returns true
    return arg_token in arg_unit

def append_const_val_postfix(v):
    if isinstance(v, int):
        return "{}@Int".format(v)
    elif isinstance(v, float):
        return "{}@Float".format(v)
    elif isinstance(v, str):
        return "{}@Str".format(v)
    elif isinstance(v, set):
        return "{{{}}}@Set".format(
            ",".join(["'{}'".format(p) if isinstance(p, str) else "{}".format(p) for p in v])
        )
    else:
        raise NotImplementedError("Unsupported type, got: {}.".format(type(v)))

def infer_const_val(arg_question, arg_input, arg_keywords=set()):
    question_tokens = get_question_tokens(arg_question, keywords=arg_keywords, remove_stop_words=True, lemmatized=False) # NLP tokens
    question_ltokens = get_question_tokens(arg_question, keywords=arg_keywords, remove_stop_words=True, lemmatized=True) # lemmatized tokens

    # related tokens generation
    # note: this is a 2-degress expansion
    question_rtokens = [fetch_wn_related_words(p) for p in question_tokens]
    question_rtokens = [
        # flatten inner list of lists
        list(set(itertools.chain(*[fetch_wn_related_words(q) for q in p])))
        for p in question_rtokens
    ]
    # filter out words that are too far away on edit distance (percentage)
    # note: even though `edit_percentage` only accepts strings, if a question token is int, it will have no related words
    #       which won't trigger type error in `edit_percentage` method
    question_rtokens = [
        [p for p in question_rtokens[i] if edit_percentage(p, question_tokens[i])<=MAXIMUM_EDIT_PERCENTAGE] 
        for i in range(len(question_rtokens))
    ]

    # a table unit can be a single word or a phrase
    table_values = arg_input.values.flatten().tolist() + arg_input.columns.values.flatten().tolist()
    table_values = list(set(table_values))
    # a unit should be a tuple of tokens, process to correct form
    table_units = [tuple(get_question_tokens(str(p), keywords=arg_keywords, remove_stop_words=True, lemmatized=False)) for p in table_values]
    table_lunits = [tuple(get_question_tokens(str(p), keywords=arg_keywords, remove_stop_words=True, lemmatized=True)) for p in table_values]

    ret_const_vals = [] # stores the matched table value (it's table value, not other forms)
    ret_map = [] # keep track of question token - table value matching (pay attention to the key-value)
    # start matching
    for i in range(len(table_values)):
        for j in range(len(question_tokens)):

            # 1. match table lunits with question ltoken
            if tu_matching(question_ltokens[j], table_lunits[i]):
                ret_const_vals.append(table_values[i])
                ret_map.append((question_tokens[j], table_values[i]))
                # don't break, we'll remove duplicates later

            # 2. match table lunits with question related rtokens
            for k in range(len(question_rtokens[j])):
                if tu_matching(question_rtokens[j][k], table_lunits[i]):
                    ret_const_vals.append(table_values[i])
                    ret_map.append((question_tokens[j], table_values[i]))
                    # don't break, we'll remove duplicates later

    # remove duplicates
    ret_const_vals = list(set(ret_const_vals))
    ret_map = list(set(ret_map))

    # # ==== set value inference ==== #
    # # also combine the currently inferred values into sets, grouped by column
    # # note: set values don't contain type postfix, so this should happen before postfix generation
    # where_const_vals = [
    #     # save the columns only
    #     np.where(arg_input==p)[1].tolist()
    #     for p in ret_const_vals
    # ]
    # # arrange the columns
    # column_groups = {}
    # for i in range(len(where_const_vals)):
    #     for c in where_const_vals[i]:
    #         if c not in column_groups.keys():
    #             column_groups[c] = set()
    #         column_groups[c].add(ret_const_vals[i])
    # # print("# [debug] column_groups: {}".format(column_groups))
    # # generate set vals
    # for c in column_groups.keys():
    #     ret_const_vals.append(column_groups[c])
    # # fixme: sets are not included in the alignment map, add them later
    # # fixme: values in column names are not indexed

    # append type as postfix
    for i in range(len(ret_const_vals)):
        ret_const_vals[i] = append_const_val_postfix(ret_const_vals[i])

    return ret_const_vals, list(set(ret_map))












