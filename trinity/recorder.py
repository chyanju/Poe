import time
import pickle

class Recorder():
    def __init__(self, name):
        self.name = name
        self.entries = []
        self.timestamp = time.time()

        self.helper_dump = pickle.dump
        self.helper_open = open

    def record(self, entry):
        self.entries.append((time.time(), entry))

    def serialize(self, sformat="pkl"):
        if sformat=="pkl":
            with self.helper_open("{}/{}.pkl".format(_recorder_dest[0], str(self)), "wb") as f:
                self.helper_dump(self.entries, f)
        else:
            raise NotImplementedError("Unsupported serialization format, got: {}.".format(sformat))

    def __str__(self):
        return "Recorder.{}.{}".format(self.timestamp, self.name)

# global recorder pool
_recorder_pool = {}
def get_recorder_pool():
    return _recorder_pool

def get_recorder(name):
    if name in _recorder_pool.keys():
        return _recorder_pool[name]
    else:
        new_recorder = Recorder(name)
        _recorder_pool[name] = new_recorder
        return new_recorder

# note: put inside a list to keep it mutable, so that 
# Recorder will store its reference, not value at initialization
_recorder_dest = [""]
def set_recorder_dest(path):
    # should use absolute path
    _recorder_dest[0] = path