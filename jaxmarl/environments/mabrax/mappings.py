from typing import Dict, List, Tuple, Union
import jax.numpy as jnp

# TODO: programatically generate these mappings from the kinematic trees
#       and add an observation distance parameter to the environment


_agent_action_mapping = {
    "ant_8x1": {
        "agent_0": jnp.array([0]),
        "agent_1": jnp.array([1]),
        "agent_2": jnp.array([2]),
        "agent_3": jnp.array([3]),
        "agent_4": jnp.array([4]),
        "agent_5": jnp.array([5]),
        "agent_6": jnp.array([6]),
        "agent_7": jnp.array([7]),
    },
    "ant_4x2": {
        "agent_0": jnp.array([0, 1]),
        "agent_1": jnp.array([2, 3]),
        "agent_2": jnp.array([4, 5]),
        "agent_3": jnp.array([6, 7]),
    },
    "halfcheetah_6x1": {
        "agent_0": jnp.array([0]),
        "agent_1": jnp.array([1]),
        "agent_2": jnp.array([2]),
        "agent_3": jnp.array([3]),
        "agent_4": jnp.array([4]),
        "agent_5": jnp.array([5]),
    },
    "halfcheetah_2x3": {
        "agent_0": jnp.array([0, 1, 2]),
        "agent_1": jnp.array([3, 4, 5]),
    },
    "hopper_3x1": {
        "agent_0": jnp.array([0]),
        "agent_1": jnp.array([1]),
        "agent_2": jnp.array([2]),
    },
    "humanoid_9|8": {
        "agent_0": jnp.array([0, 1, 2, 11, 12, 13, 14, 15, 16]),
        "agent_1": jnp.array([3, 4, 5, 6, 7, 8, 9, 10]),
    },
    "walker2d_2x3": {
        "agent_0": jnp.array([0, 1, 2]),
        "agent_1": jnp.array([3, 4, 5]),
    },
    "walker2d_6x1": {
        "agent_0": jnp.array([0]),
        "agent_1": jnp.array([1]),
        "agent_2": jnp.array([2]),
        "agent_3": jnp.array([3]),
        "agent_4": jnp.array([4]),
        "agent_5": jnp.array([5]),
    },
}

_agent_state_mapping = {
    "ant_8x1": {
        "agent_0": jnp.array(range(0,27)),
        "agent_1": jnp.array(range(0,27)),
        "agent_2": jnp.array(range(0,27)),
        "agent_3": jnp.array(range(0,27)),
        "agent_4": jnp.array(range(0,27)),
        "agent_5": jnp.array(range(0,27)),
        "agent_6": jnp.array(range(0,27)),
        "agent_7": jnp.array(range(0,27)),
    },
    "ant_4x2": {
        "agent_0": jnp.array(range(0,27)),
        "agent_1": jnp.array(range(0,27)),
        "agent_2": jnp.array(range(0,27)),
        "agent_3": jnp.array(range(0,27)),
    },
    "halfcheetah_6x1": {
        "agent_0": jnp.array(range(0,17)),
        "agent_1": jnp.array(range(0,17)),
        "agent_2": jnp.array(range(0,17)),
        "agent_3": jnp.array(range(0,17)),
        "agent_4": jnp.array(range(0,17)),
        "agent_5": jnp.array(range(0,17)),
    },
    "halfcheetah_2x3": {
        "agent_0": jnp.array(range(0,17)),
        "agent_1": jnp.array(range(0,17)),
    },
    "hopper_3x1": {
        "agent_0": jnp.array(range(0,11)),
        "agent_1": jnp.array(range(0,11)),
        "agent_2": jnp.array(range(0,11)),
    },
    "humanoid_9|8": {
        "agent_0": jnp.array(range(0,244)),
        "agent_1": jnp.array(range(0,244)),
    },
    "walker2d_6x1": {
        "agent_0": jnp.array(range(0,17)),
        "agent_1": jnp.array(range(0,17)),
        "agent_2": jnp.array(range(0,17)),
        "agent_3": jnp.array(range(0,17)),
        "agent_4": jnp.array(range(0,17)),
        "agent_5": jnp.array(range(0,17)),
    },
    "walker2d_2x3": {
        "agent_0": jnp.array(range(0,17)),
        "agent_1": jnp.array(range(0,17)),
    },
}


def listerize(ranges: List[Union[int, Tuple[int, int]]]) -> List[int]:
    return [
        i
        for r in ranges
        for i in (range(r[0], r[1] + 1) if isinstance(r, tuple) else [r])
    ]


ranges: Dict[str, Dict[str, List[Union[int, Tuple[int, int]]]]] = {
    "ant_8x1": {
        "agent_0": [(0, 7), 9, 11, (13, 19)],
        "agent_1": [(0, 6), (13, 18), 30],
        "agent_2": [(0, 5), (7, 9), 11, (13, 18), 21],
        "agent_3": [(0, 4), 7, 8, (13, 18), 22],
        "agent_4": [(0, 5), 7, 9, 10, 11, (13, 18), 23],
        "agent_5": [(0, 4), 9, 10, (13, 18), 24],
        "agent_6": [(0, 5), 7, 9, (11, 18), 25],
        "agent_7": [(0, 4), 11, 12, (13, 18), 26],
    },
    "ant_4x2": {
        "agent_0": [(0, 5), 6, 7, 9, 11, (13, 18), 19, 20],
        "agent_1": [(0, 5), 7, 8, 9, 11, (13, 18), 21, 22],
        "agent_2": [(0, 5), 7, 9, 10, 11, (13, 18), 23, 24],
        "agent_3": [(0, 5), 7, 9, 11, 12, (13, 18), 25, 26],
    },
    "halfcheetah_6x1": {
        "agent_0": [(0, 3), 5, (8, 11)],
        "agent_1": [(0, 4), (8, 10), 12],
        "agent_2": [(0, 1), 3, 4, (8, 10), 13],
        "agent_3": [(0, 2), 5, 6, (8, 10), 14],
        "agent_4": [(0, 1), 5, 6, 7, (8, 10), 15],
        "agent_5": [(0, 1), 6, 7, (8, 10), 16],
    },
    "halfcheetah_2x3": {
        "agent_0": [(0, 5), (8, 13)],
        "agent_1": [(0, 2), (5, 10), (14, 16)],
    },
    "hopper_3x1": {
        "agent_0": [(0, 1), 2, 3, (5, 7), 8],
        "agent_1": [(0, 1), 2, 3, 4, (5, 7), 9],
        "agent_2": [(0, 1), 3, 4, (5, 7), 10],
    },
    "humanoid_9|8": {
        "agent_0": [
            (0, 10),
            (12, 14),
            (16, 30),
            (39, 44),
            (45, 84),
            (95, 104),
            (115, 178),
            (185, 190),
            (197, 232),
            (234, 236),
            (238, 243),
        ],
        "agent_1": [
            (0, 15),
            (22, 38),
            (45, 114),
            (155, 196),
            (221, 237),
        ],
    },
    "walker2d_6x1": {
        "agent_0": [(0, 3), 5, (8, 11)],
        "agent_1": [(0, 4), (8, 10), 12],
        "agent_2": [0, 1, 3, 4, (8, 10), 13],
        "agent_3": [(0, 2), 5, 6, (8, 10), 14],
        "agent_4": [0, 1, (5, 10), 15],
        "agent_5": [0, 1, (6, 10), 16],
    },
    "walker2d_2x3": {
        "agent_0": [(0, 5), (8, 13)],
        "agent_1": [(0, 2), (5, 10), (14, 16)],
    },
}

_agent_observation_mapping = {
    k: {k_: jnp.array(listerize(v_)) for k_, v_ in v.items()} for k, v in ranges.items()
}