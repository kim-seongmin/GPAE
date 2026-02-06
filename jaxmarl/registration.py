from .environments import (
    SMAX,
    HeuristicEnemySMAX,
    LearnedPolicyEnemySMAX,
    Ant_4x2,
    Ant_8x1,
    Humanoid,
    Hopper,
    Walker2d_6x1,
    Walker2d_2x3,
    HalfCheetah_6x1,
    HalfCheetah_2x3,
)



def make(env_id: str, **env_kwargs):
    """A JAX-version of OpenAI's env.make(env_name), built off Gymnax"""
    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered jaxmarl environments.")

    # 3. SMAX
    if env_id == "SMAX":
        env = SMAX(**env_kwargs)
    elif env_id == "HeuristicEnemySMAX":
        env = HeuristicEnemySMAX(**env_kwargs)
    elif env_id == "LearnedPolicyEnemySMAX":
        env = LearnedPolicyEnemySMAX(**env_kwargs)

    # 4. MABrax
    if env_id == "ant_4x2":
        env = Ant_4x2(**env_kwargs)
    elif env_id == "ant_8x1":
        env = Ant_8x1(**env_kwargs)
    elif env_id == "halfcheetah_6x1":
        env = HalfCheetah_6x1(**env_kwargs)
    elif env_id == "halfcheetah_2x3":
        env = HalfCheetah_2x3(**env_kwargs)
    elif env_id == "hopper_3x1":
        env = Hopper(**env_kwargs)
    elif env_id == "humanoid_9|8":
        env = Humanoid(**env_kwargs)
    elif env_id == "walker2d_6x1":
        env = Walker2d_6x1(**env_kwargs)
    elif env_id == "walker2d_2x3":
        env = Walker2d_2x3(**env_kwargs)

    return env

registered_envs = [
    "SMAX",
    "HeuristicEnemySMAX",
    "LearnedPolicyEnemySMAX",
    "ant_8x1",
    "ant_4x2",
    "halfcheetah_6x1",
    "halfcheetah_2x3",
    "hopper_3x1",
    "humanoid_9|8",
    "walker2d_6x1",
    "walker2d_2x3",
]
