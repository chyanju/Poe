from collections import namedtuple

Interval = namedtuple("Interval", ["lb", "ub"])
IMIN = 0 # default minimum value of an interval
IMAX = 127 # default maximum value of an interval
# ########## interval operations ########## #
# translate constraint to operations
# e.g., a.height == b.height, is translated to, a.height <- (== b.height)
# e.g., a.height == b.height - 1, is translated to, a.height <- (== ( - b.height (1,1) )))
#       here 1 is converted to interval first

# note-important: validate MUST be done before calling rebound
#                 otherwise invalid intervals may be converted to valid ones
# note-important: before or after every interval operation (including intersection), 
#                 you need to call validate & rebound
# note-important: intervals use None to represent empty set, *NOT* (IMAX,IMIN).
def validate_interval(iv):
    if iv is None:
        return None
    elif iv.lb > iv.ub:
        return None
    else:
        return iv

def rebound_interval(iv):
    if iv is None:
        return None
    else:
        return Interval(
            max(IMIN, iv.lb),
            min(IMAX, iv.ub),
        )

def interval_binary_op(op, arg0, arg1):
    # note: arg0 is the main arg, that is,
    #       arg0 is out, arg1 is a/...
    #       since the binary op is NOT symmetric for ==/<=/>=/</> (comparators)
    res = None

    if arg0 is None or arg1 is None:
        return None
    else:
        if op == "+":
            res = Interval( arg0.lb+arg1.lb, arg0.ub+arg1.ub )
        elif op == "-":
            res = Interval( arg0.lb-arg1.ub, arg0.ub-arg1.lb )
        elif op == "*":
            # via validate & rebound every time, we know arg0 and arg1 won't have negative values
            # that can flip the lb&ub after multiplication, so it's safe to assume that they are all non-negative
            res = Interval( arg0.lb*arg1.lb, arg0.ub*arg1.ub )
        elif op == "==":
            # essentially this is doing intersection
            # fixme: for quick impl. use set intersection (which is not efficient)
            set0 = set(list(range(arg0.lb, arg0.ub+1)))
            set1 = set(list(range(arg1.lb, arg1.ub+1)))
            set2 = set0 & set1
            if len(set2)==0:
                return None
            else:
                res = Interval( min(set2), max(set2) )
        elif op == "<=":
            res = Interval( arg0.lb, min(arg0.ub, arg1.ub) )
        elif op == ">=":
            res = Interval( max(arg0.lb, arg1.lb), arg0.ub )
        elif op ==  "<":
            res = Interval( arg0.lb, min(arg0.ub, arg1.ub-1) )
        elif op == ">":
            res = Interval( max(arg0.lb, arg1.lb+1), arg0.ub )
        else:
            raise ValueError("Unsupported unary op {}.".format(op))

    res = validate_interval(res)
    res = rebound_interval(res)
    return res

def interval_is_intersected(iv0, iv1):
    r0 = validate_interval(iv0)
    r0 = rebound_interval(r0)
    r1 = validate_interval(iv1)
    r1 = rebound_interval(r1)
    if r0 is None or r1 is None:
        return False
    else:
        return ( (iv0.lb<=iv1.ub) and (iv1.lb<=iv0.ub) )

def interval_to_set(iv):
    return set([x for x in range(iv.lb, iv.ub+1)])