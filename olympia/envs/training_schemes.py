
"""

This file could have different parameters for training schemes hardcoded

- one player
- one v one
- two v two
- three v three
- etc...

"""

scheme = {
    'one_player': [[(0.7, 0.5)]],
    'one_v_one': [[(0.7, 0.5)], [(0.3, 0.5)]],
    'two_v_two': [[(0.7, 0.25), (0.7,0.75)], [(0.3, 0.25), (0.3, 0.75)]]
}