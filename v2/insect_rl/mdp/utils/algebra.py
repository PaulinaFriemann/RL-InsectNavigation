import numpy as np


def angle(action, homing_vector):
    """Angle between an action and the homing vector, in radians [0,pi].
    For action (0,0), return pi
    """
    if not any(action): # (0,0) action
        return np.pi
    a = np.array(action, dtype=float)
    b = np.array(homing_vector, dtype=float)
    return np.arctan2(np.linalg.norm(np.cross(a,b)), np.dot(a,b))


def similarity(action, homing_vector):
    # inversely scale vector angle to [-1,1]
    return 2* (1 - angle(action, homing_vector) / np.pi) - 1
