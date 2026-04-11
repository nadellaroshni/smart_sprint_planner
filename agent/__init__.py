from .heuristic_agent import HeuristicAgent

__all__ = ["HeuristicAgent"]

try:
    from .features import encode, action_space, FEATURE_DIM
except ImportError:
    encode = None
    action_space = None
    FEATURE_DIM = None
else:
    __all__.extend(["encode", "action_space", "FEATURE_DIM"])

try:
    from .dqn_agent import DDQNAgent
except ImportError:
    DDQNAgent = None
else:
    __all__.append("DDQNAgent")
