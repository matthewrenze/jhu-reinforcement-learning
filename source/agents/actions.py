from enum import Enum

class Action(Enum):
    NONE = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4



# action_ids = {
#     "none": 0,
#     "up": 1,
#     "down": 2,
#     "left": 3,
#     "right": 4}
#
# action_names = {v: k for k, v in action_ids.items()}