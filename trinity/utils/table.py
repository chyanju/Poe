import pandas as pd

from ..spec.interval import *

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Padding(metaclass=Singleton):
    def __str__(self):
        return "<PAD>"

    def __repr__(self):
        return "<PAD>"

# override the class with instance
PADDING = Padding()

class RCPartialTable():
    '''
    A partial table class that provides some aligned APIs with the real pandas table,
    so that some methods like `get_row` can be reused. This table follows the filling order of RC-series.
    '''
    def __init__(self, columns, data, expected_shape, pp_columns, pp_data):
        assert len(columns.shape)==1, "Multiple indices/columns are not supported, got: {}.".format(columns)
        assert len(columns)==data.shape[1], "Header length should match column length, got {} and {}.".format(len(columns), data.shape[1])
        self.columns = columns
        self.data = data

        assert expected_shape==data.shape, "Expected shape should match padded data shape, got: {} and {}.".format(expected_shape, data.shape)
        # note: usually you shouldn't have used/known this property in KS setting
        self.expected_shape = expected_shape

        assert pp_columns<len(columns), "Shape checking fails, pp_columns={}, len(columns)={}.".format(pp_columns, len(columns))
        assert pp_columns>=0, "pp_columns should be non-negative, got: {}.".format(pp_columns)
        assert pp_data<expected_shape[0]*expected_shape[1], "Shape checking fails, pp_data={}, expected_shape={}.".format(pp_data, expected_shape)
        assert pp_data>=0, "pp_data should be non-negative, got: {}.".format(pp_data)
        if pp_columns<len(columns)-1:
            # still need to fill in column names, then no data shall be filled in now
            assert pp_data==0, "Invalid partial data state: need to fill in column name first, but pp_columns={}, pp_data={}.".format(pp_columns, pp_data)
        # pp: progress pointer, all cells before pp flattened position are concrete
        #     can also be interpreted as number of concrete cells in the table
        self.pp_columns = pp_columns
        self.pp_data = pp_data

    def __str__(self):
        return pd.DataFrame(data=self.data, columns=self.columns).__str__()
        
def get_row(arg_tb):
    if isinstance(arg_tb, pd.DataFrame):
        # precise table
        v = arg_tb.shape[0]
        return Interval(v, v)
    elif isinstance(arg_tb, RCPartialTable):
        # RCPartialTable follows the RC Composer Assumption
        # partial table, imprecise
        # for row, no need to care about columns
        p = arg_tb.pp_data // arg_tb.expected_shape[1]
        q = arg_tb.pp_data % arg_tb.expected_shape[1]
        v = p+1 if q>0 else p
        if arg_tb.pp_data<arg_tb.expected_shape[0]*arg_tb.expected_shape[1]:
            # still filling in cell values
            return Interval(v, IMAX)
        else:
            # done filling in cell values, you can have precise version now
            return Interval(v, v)
    else:
        raise NotImplementedError("Unsupported type of table for get_row, got: {}.".format(type(arg_tb)))

def get_col(arg_tb):
    if isinstance(arg_tb, pd.DataFrame):
        # precise table
        v = arg_tb.shape[1]
        return Interval(v, v)
    elif isinstance(arg_tb, RCPartialTable):
        # RCPartialTable follows the RC Composer Assumption
        # partial table, imprecise
        if arg_tb.pp_columns<arg_tb.expected_shape[1]:
            # still filling in column names
            return Interval(arg_tb.pp_columns, IMAX)
        else:
            # done filling in column names, you can have precise number of columns already
            return Interval(arg_tb.expected_shape[1], arg_tb.expected_shape[1])
    else:
        raise NotImplementedError("Unsupported type of table for get_col, got: {}.".format(type(arg_tb)))

def get_head(arg_base, arg_tb):
    if isinstance(arg_base, pd.DataFrame):
        if isinstance(arg_tb, pd.DataFrame):
            # precise table
            head0 = set(arg_base.columns)
            content0 = set(arg_base.values.flatten().tolist())
            # note-important: relax head values for answer table
            head1 = set(arg_tb.columns)
            v = len(head1 - head0 - content0)
            if "ANSWER" in head1:
                # special answer table, ignore the "ANSWER" column
                return Interval(v-1, v)
            else:
                return Interval(v, v)
        elif isinstance(arg_tb, RCPartialTable):
            # RCPartialTable follows the RC Composer Assumption
            # partial table, imprecise
            head0 = set(arg_base.columns)
            content0 = set(arg_base.values.flatten().tolist())
            # note-important: relax head values for answer table
            head1 = set(arg_tb.columns) - set([PADDING])
            v = len(head1 - head0 - content0)
            if arg_tb.pp_columns<arg_tb.expected_shape[1]:
                # still filling in column names
                if "ANSWER" in head1:
                    return Interval(v-1, IMAX)
                else:
                    return Interval(v, IMAX)
            else:
                # done filling in column names, you can have precise version now
                if "ANSWER" in head1:
                    return Interval(v-1,v)
                else:
                    return Interval(v, v)
        else:
            raise NotImplementedError("Unsupported type of table for get_head, got: {}.".format(type(arg_tb)))
    else:
        raise NotImplementedError("Unsupported type of base table for get_head, got: {}.".format(type(arg_base)))

def get_content(arg_base, arg_tb):
    if isinstance(arg_base, pd.DataFrame):
        if isinstance(arg_tb, pd.DataFrame):
            # precise table
            content0 = set(arg_base.values.flatten().tolist())
            content1 = set(arg_tb.values.flatten().tolist())
            v = len(content1 - content0)
            return Interval(v, v)
        elif isinstance(arg_tb, RCPartialTable):
            # RCPartialTable follows the RC Composer Assumption
            # partial table, imprecise
            content0 = set(arg_base.values.flatten().tolist())
            content1 = set(arg_tb.data.flatten().tolist()) - set([PADDING])
            v = len(content1 - content0)
            if arg_tb.pp_data<arg_tb.expected_shape[0]*arg_tb.expected_shape[1]:
                # still filling in cell values
                return Interval(v, IMAX)
            else:
                # done filling in cell values, you can have precise version now
                return Interval(v, v)
        else:
            raise NotImplementedError("Unsupported type of table for get_head, got: {}.".format(type(arg_tb)))
    else:
        raise NotImplementedError("Unsupported type of base table for get_head, got: {}.".format(type(arg_base)))

def make_abs():
    return {
        "row": Interval(IMIN,IMAX),
        "col": Interval(IMIN,IMAX),
        "head": Interval(IMIN,IMAX),
        "content": Interval(IMIN,IMAX),
    }

def assemble_abstract_table(arg_tb0, arg_tb1):
    # arg_tb0 is the base, i.e., one of the exampe input(s)
    return {
        "row": get_row(arg_tb1),
        "col": get_col(arg_tb1),
        "head": get_head(arg_tb0, arg_tb1),
        "content": get_content(arg_tb0, arg_tb1),
    }

def abs_intersected(abs0, abs1):
    # within this framework, abs0 and abs1 have the same key set
    for p in abs0.keys():
        if not interval_is_intersected(abs0[p], abs1[p]):
            return False
    return True