from typing import Callable, NamedTuple, List, Any
from .decider import Decider

Example = NamedTuple('Example', [
    ('input', List[Any]),
    ('output', Any),
    ('query', str)
])
