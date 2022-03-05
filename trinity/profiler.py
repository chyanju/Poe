import time

class Profiler():
    def __init__(self, name):
        self.name = name
        self.counters = {}
        self.cumulative_timers = {} # every time new time is added
        self.sequential_timers = {} # every time new time is appended

        self.float_precision = 2

    def is_empty(self):
        return len(self.counters) == 0 and \
               len(self.cumulative_timers) == 0 and \
               len(self.sequential_timers) == 0

    def reset(self):
        self.counters = {}
        self.cumulative_timers = {}
        self.sequential_timers = {}

    # decorator
    def ctimer(self, item):
        def decorator(func):
            def wrapper(*args, **kwargs):
                if item not in self.cumulative_timers:
                    self.cumulative_timers[item] = 0
                tstart = time.time()
                try:
                    res = func(*args, **kwargs)
                except:
                    # with exception
                    tend = time.time()
                    self.cumulative_timers[item] += tend-tstart
                    raise
                # no exception
                tend = time.time()
                self.cumulative_timers[item] += tend-tstart
                return res
            return wrapper
        return decorator

    # decorator
    def stimer(self, item):
        def decorator(func):
            def wrapper(*args, **kwargs):
                if item not in self.sequential_timers:
                    self.sequential_timers[item] = []
                tstart = time.time()
                try:
                    res = func(*args, **kwargs)
                except:
                    # with exception
                    tend = time.time()
                    self.sequential_timers[item] += tend-tstart
                    raise
                # no exception
                tend = time.time()
                self.sequential_timers[item] += tend-tstart
                return res
            return wrapper
        return decorator

    # normal function
    def add1(self, item):
        if item not in self.counters:
            self.counters[item] = 0
        self.counters[item] += 1

    def __str__(self):
        return self.pretty_str()

    def pretty_str(self):
        output_str =  "Profiler: {}\n".format(self.name)

        if self.is_empty():
            output_str = "(empty) " + output_str
        else:
            output_str += "|- counters\n"
            for dkey in sorted(list(self.counters.keys())):
                output_str += "   |- {}: {}\n".format(dkey, self.counters[dkey])
            
            output_str += "|- timers (cumulative)\n"
            for dkey in sorted(list(self.cumulative_timers.keys())):
                output_str += "   |- {}: {}\n".format(dkey, round(self.cumulative_timers[dkey], self.float_precision))

            output_str += "|- timers (sequential)\n"
            for dkey in sorted(list(self.sequential_timers.keys())):
                output_str += "   |- {}: {}\n".format(dkey, [round(p, self.float_precision) for p in self.sequential_timers[dkey]])

        return output_str

# global profiler pool
_profiler_pool = {}
def get_profiler_pool():
    return _profiler_pool
def clear_profiler_pool():
    _profiler_pool = {}
def reset_profiler_pool():
    # reset all statistics
    # usually called after keystroke process to clear off interpreter
    for dkey in _profiler_pool.keys():
        _profiler_pool[dkey].reset()

def get_profiler(name, register=True):
    if register:
        if name in _profiler_pool.keys():
            return _profiler_pool[name]
        else:
            new_profiler = Profiler(name)
            _profiler_pool[name] = new_profiler
            return new_profiler
    else:
        new_profiler = Profiler(name)
        return new_profiler
