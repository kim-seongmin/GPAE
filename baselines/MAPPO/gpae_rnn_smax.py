"""
Based on PureJaxRL Implementation of IPPO, with changes to give a centralised critic.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Tuple, Union, Dict
import wandb
import functools
from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import DictConfig, OmegaConf
from functools import partial
from copy import deepcopy

from jaxmarl.wrappers.baselines import SMAXLogWrapper, JaxMARLWrapper
from jaxmarl.environments.smax import map_name_to_scenario, HeuristicEnemySMAX

class SMAXWorldStateWrapper(JaxMARLWrapper):
    """
    Provides a `"world_state"` observation for the centralised critic.
    world state observation of dimension: (num_agents, world_state_size)    
    """
    
    def __init__(self,
                 env: HeuristicEnemySMAX,
                 obs_with_agent_id=True,
                 state_with_agent_id=True,):
        super().__init__(env)
        self.obs_with_agent_id = obs_with_agent_id
        self.state_with_agent_id = state_with_agent_id
        
        if not self.state_with_agent_id:
            self._world_state_size = self._env.state_size
            self.world_state_fn = self.ws_just_env_state
        else:
            self._world_state_size = self._env.state_size + self._env.num_allies
            self.world_state_fn = self.ws_with_agent_id

        if not self.obs_with_agent_id:
            self._obs_size = self._env.obs_size
            self.obs_fn = self.obs_just_obs
        else:
            self._obs_size = self._env.obs_size + self._env.num_allies
            self.obs_fn = self.obs_with_id
            
    
    @partial(jax.jit, static_argnums=0)
    def reset(self,
              key):
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self.world_state_fn(obs, env_state)
        obs = self.obs_fn(obs)
        return obs, env_state
    
    @partial(jax.jit, static_argnums=0)
    def step(self,
             key,
             state,
             action):
        obs, env_state, reward, done, info = self._env.step(
            key, state, action
        )
        obs["world_state"] = self.world_state_fn(obs, state)
        obs = self.obs_fn(obs)
        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def ws_just_env_state(self, obs, state):
        #return all_obs
        world_state = obs["world_state"]
        world_state = world_state[None].repeat(self._env.num_allies, axis=0)
        return world_state
        
    @partial(jax.jit, static_argnums=0)
    def ws_with_agent_id(self, obs, state):
        #all_obs = jnp.array([obs[agent] for agent in self._env.agents])
        world_state = obs["world_state"]
        world_state = world_state[None].repeat(self._env.num_allies, axis=0)
        one_hot = jnp.eye(self._env.num_allies)
        return jnp.concatenate((world_state, one_hot), axis=1)

    @partial(jax.jit, static_argnums=0)
    def obs_just_obs(self, obs):
        #return all_obs
        return obs
        
    @partial(jax.jit, static_argnums=0)
    def obs_with_id(self, obs):
        one_hot = jnp.eye(self._env.num_allies)
        for i, agent in zip(one_hot, self._env.agents):
            obs[agent] = jnp.concatenate((obs[agent], i))
        return obs
        
    def world_state_size(self):
   
        return self._world_state_size 
    
    def obs_size(self):
   
        return self._obs_size 

class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        hidden_size = ins.shape[-1]
        rnn_state = jnp.where(
            jnp.expand_dims(resets,-1),
            self.initialize_carry(hidden_size, *ins.shape[:-1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell(features=hidden_size).initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )


class ActorRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones, avail_actions = x
        if self.config["LAYER_NORM"]:
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x
        
        if self.config["ACTIVATION"] == "relu":
            act = lambda x: nn.relu(x)
        elif self.config["ACTIVATION"] == "tanh":
            act = lambda x: nn.tanh(x)
        else:
            act = lambda x: nn.sigmoid(x)

        if self.config["NORM_INPUT"]:
            x = nn.LayerNorm()(obs)
        else:
            # dummy normalize input in any case for global compatibility
            x_dummy = nn.LayerNorm()(obs)
            x = obs

        for l in range(self.config["NUM_LAYERS"]):
            x = nn.Dense(self.config["HIDDEN_SIZE"])(x)
            x = normalize(x)
            x = act(x)

        rnn_in = (x, dones)
        hidden, x = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(self.action_dim)(x)
        unavail_actions = 1 - avail_actions
        action_logits = actor_mean - (unavail_actions * 1e10)

        pi = distrax.Categorical(logits=action_logits)

        return hidden, pi


class CriticRNN(nn.Module):
    config: Dict
    
    @nn.compact
    def __call__(self, hidden, x):
        world_state, action, dones = x

        if self.config["LAYER_NORM"]:
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x
        
        if self.config["ACTIVATION"] == "relu":
            act = lambda x: nn.relu(x)
        elif self.config["ACTIVATION"] == "tanh":
            act = lambda x: nn.tanh(x)
        else:
            act = lambda x: nn.sigmoid(x)

        if self.config["NORM_INPUT"]:
            world_state = nn.LayerNorm()(world_state)
            action = nn.LayerNorm()(action)
        else:
            # dummy normalize input in any case for global compatibility
            world_state_dummy = nn.LayerNorm()(world_state)
            action_dummy = nn.LayerNorm()(action)

        if self.config["ADVANTAGE"] == 'gpae' or self.config["ADVANTAGE"] == 'coma':
            state_embedding = nn.Dense(int(self.config["HIDDEN_SIZE"] / 2))(world_state)
            state_embedding = normalize(state_embedding)
            state_embedding = act(state_embedding)

            action_embedding = nn.Dense(int(self.config["HIDDEN_SIZE"] / 2))(action)
            action_embedding = normalize(action_embedding)
            action_embedding = act(action_embedding)

            embedding = jnp.concatenate([state_embedding, action_embedding], axis=-1)

        else:
            embedding = nn.Dense(self.config["HIDDEN_SIZE"])(world_state)
            embedding = normalize(embedding)
            embedding = act(embedding)
        
        for l in range(self.config["NUM_LAYERS"] - 1):
            embedding = nn.Dense(self.config["HIDDEN_SIZE"])(embedding)
            embedding = normalize(embedding)
            embedding = act(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        q_value = nn.Dense(1)(embedding)

        return hidden, jnp.squeeze(q_value, axis=-1)

class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    logits: jnp.ndarray
    log_prob: jnp.ndarray
    current_log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray


def batchify(x: dict, agent_list):
    return jnp.stack([x[a] for a in agent_list], axis=0)
    #print('batchify', x.shape)
    # return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs):
    x = x.reshape((len(agent_list), num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def marginal_actions(acts, logits, act_dim, num_agents):
    rollouts = acts.shape[2]
    steps = acts.shape[0]
    acts_flat = acts.reshape(-1, 1)
    input_action = jnp.zeros((acts_flat.shape[0], act_dim))
    input_action = input_action.at[jnp.arange(len(acts_flat)), acts_flat.squeeze()].set(1)
    input_action = input_action.reshape(steps, 1, rollouts, num_agents, act_dim)
    # repeated_action = jnp.zeros((steps, num_agents, rollouts, num_agents, act_dim))
    actprobs = jnp.exp(logits).transpose(1, 0, 2, 3)
    agent_indices = jnp.arange(num_agents)  # Shape: (num_agents,)
    repeated_action = input_action.repeat(num_agents, axis=1)
    repeated_action = repeated_action.at[:, agent_indices, :, agent_indices].set(actprobs[agent_indices])
    final_output = repeated_action.reshape(-1, num_agents, rollouts, num_agents * act_dim)

    return final_output

def tree_l2_norm_squared(tree):
    """Return the sum of squares of all entries in a PyTree."""
    leaves = jax.tree_util.tree_leaves(tree)
    return sum([jnp.sum(leaf ** 2) for leaf in leaves])

def make_train(config):
    scenario = map_name_to_scenario(config["MAP_NAME"])
    env = HeuristicEnemySMAX(scenario=scenario, **config["ENV_KWARGS"])
    config["NUM_ENVS"] = int(config["NUM_ENVS"] / config["BUFFER_SIZE"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["CLIP_EPS"] = (
        config["CLIP_EPS"] / env.num_agents
        if config["SCALE_CLIP_EPS"]
        else config["CLIP_EPS"]
    )
    config["ACT_DIM"] = env.action_space(env.agents[0]).n

    env = SMAXWorldStateWrapper(env, config["OBS_WITH_AGENT_ID"], config["STATE_WITH_AGENT_ID"])
    env = SMAXLogWrapper(env)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        actor_network = ActorRNN(config["ACT_DIM"], config=config)
        critic_network = CriticRNN(config=config)
        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
        ac_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], env.obs_size())),
            jnp.zeros((1, config["NUM_ENVS"])),
            jnp.zeros((1, config["NUM_ENVS"], config["ACT_DIM"])),
        )
        ac_init_hstate = ScannedRNN.initialize_carry(config["HIDDEN_SIZE"], config["NUM_ENVS"])
        actor_network_params = actor_network.init(_rng_actor, ac_init_hstate, ac_init_x)
        cr_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], env.world_state_size())),
            jnp.zeros((1, config["NUM_ENVS"], config["ACT_DIM"] * len(env.agents))),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        cr_init_hstate = ScannedRNN.initialize_carry(config["HIDDEN_SIZE"], config["NUM_ENVS"])
        critic_network_params = critic_network.init(_rng_critic, cr_init_hstate, cr_init_x)

        if config["ANNEAL_LR"]:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        actor_train_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_network_params,
            tx=actor_tx,
        )
        critic_train_state = TrainState.create(
            apply_fn=critic_network.apply,
            params=critic_network_params,
            tx=critic_tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        ac_init_hstate = ScannedRNN.initialize_carry(config["HIDDEN_SIZE"], len(env.agents), config["NUM_ENVS"])
        cr_init_hstate = ScannedRNN.initialize_carry(config["HIDDEN_SIZE"], len(env.agents), config["NUM_ENVS"])

        traj_info = {
            'returned_episode': jnp.zeros((config["NUM_STEPS"] * config["BUFFER_SIZE"], len(env.agents), config["NUM_ENVS"]), dtype=bool), 
            'returned_episode_lengths': jnp.zeros((config["NUM_STEPS"] * config["BUFFER_SIZE"], len(env.agents), config["NUM_ENVS"]), dtype=jnp.float32), 
            'returned_episode_returns': jnp.zeros((config["NUM_STEPS"] * config["BUFFER_SIZE"], len(env.agents), config["NUM_ENVS"]), dtype=jnp.float32), 
            'returned_won_episode': jnp.zeros((config["NUM_STEPS"] * config["BUFFER_SIZE"], len(env.agents), config["NUM_ENVS"]), dtype=jnp.float32)}
        buffer_state = Transition(
            jnp.zeros((config["NUM_STEPS"] * config["BUFFER_SIZE"], len(env.agents), config["NUM_ENVS"]), dtype=bool),
            jnp.zeros((config["NUM_STEPS"] * config["BUFFER_SIZE"], len(env.agents), config["NUM_ENVS"]), dtype=bool),
            jnp.zeros((config["NUM_STEPS"] * config["BUFFER_SIZE"], len(env.agents), config["NUM_ENVS"]), dtype=jnp.int32),
            jnp.zeros((config["NUM_STEPS"] * config["BUFFER_SIZE"], len(env.agents), config["NUM_ENVS"]), dtype=jnp.float32),
            jnp.zeros((config["NUM_STEPS"] * config["BUFFER_SIZE"], len(env.agents), config["NUM_ENVS"]), dtype=jnp.float32),
            jnp.zeros((config["NUM_STEPS"] * config["BUFFER_SIZE"], len(env.agents), config["NUM_ENVS"],env.action_space(env.agents[0]).n), dtype=jnp.float32),
            jnp.zeros((config["NUM_STEPS"] * config["BUFFER_SIZE"], len(env.agents), config["NUM_ENVS"]), dtype=jnp.float32),
            jnp.zeros((config["NUM_STEPS"] * config["BUFFER_SIZE"], len(env.agents), config["NUM_ENVS"]), dtype=jnp.float32),
            jnp.zeros((config["NUM_STEPS"] * config["BUFFER_SIZE"], len(env.agents), config["NUM_ENVS"],env.obs_size()), dtype=jnp.float32),
            jnp.zeros((config["NUM_STEPS"] * config["BUFFER_SIZE"], len(env.agents), config["NUM_ENVS"],env.world_state_size()), dtype=jnp.float32),
            traj_info,
            jnp.zeros((config["NUM_STEPS"] * config["BUFFER_SIZE"], len(env.agents), config["NUM_ENVS"],env.action_space(env.agents[0]).n), dtype=jnp.uint8),
        )

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state
            
            def _env_step(runner_state, unused):
                train_states, env_state, buffer_state, last_obs, last_done, hstates, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents)
                )
                obs_batch = batchify(last_obs, env.agents)
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                    avail_actions,
                )
                ac_hstate, pi = actor_network.apply(train_states[0].params, hstates[0], ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                logits = pi.logits
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"])

                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # VALUE
                # output of wrapper is (num_envs, num_agents, world_state_size)
                # swap axes to (num_agents, num_envs, world_state_size) before reshaping to (num_actors, world_state_size)
                world_state = last_obs["world_state"].swapaxes(0,1)  
                world_state = world_state.reshape((len(env.agents), config["NUM_ENVS"],-1))

                input_action = marginal_actions(action, logits, config["ACT_DIM"], len(env.agents))
                
                cr_in = (
                    world_state[None, :],  # State input
                    input_action,  # Action input
                    last_done[np.newaxis, :],  # Done flags
                )
                # print('env step cr in', cr_in)
                # print(train_states[1].params, hstates[1])
                cr_hstate, value = critic_network.apply(train_states[1].params, hstates[1], cr_in)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                info = jax.tree.map(lambda x: x.reshape((len(env.agents), config["NUM_ENVS"])), info)
                done_batch = batchify(done, env.agents).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents).reshape(env.num_agents,-1),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    config["REWARD_SCALE"] * batchify(reward, env.agents).squeeze(),
                    logits.squeeze(),
                    log_prob.squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    world_state,
                    info,
                    avail_actions,
                )

                runner_state = (train_states, env_state, buffer_state, obsv, done_batch, (ac_hstate, cr_hstate), rng)
                return runner_state, transition

            initial_hstates = runner_state[-2]
            runner_state, current_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            train_states, env_state, buffer_state, last_obs, last_done, hstates, rng = runner_state

            batch_size = config["NUM_STEPS"] * config["BUFFER_SIZE"]

            new_info = {k:jnp.concatenate([buffer_state.info[k], current_batch.info[k]])[-batch_size:] for k in buffer_state.info.keys()}

            buffer_state = Transition(
                jnp.concatenate([buffer_state.global_done, current_batch.global_done])[-batch_size:],
                jnp.concatenate([buffer_state.done, current_batch.done])[-batch_size:],
                jnp.concatenate([buffer_state.action, current_batch.action])[-batch_size:],
                jnp.concatenate([buffer_state.value, current_batch.value])[-batch_size:],
                jnp.concatenate([buffer_state.reward, current_batch.reward])[-batch_size:],
                jnp.concatenate([buffer_state.logits, current_batch.logits])[-batch_size:],
                jnp.concatenate([buffer_state.log_prob, current_batch.log_prob])[-batch_size:],
                jnp.concatenate([buffer_state.current_log_prob, current_batch.current_log_prob])[-batch_size:],
                jnp.concatenate([buffer_state.obs, current_batch.obs])[-batch_size:],
                jnp.concatenate([buffer_state.world_state, current_batch.world_state])[-batch_size:],
                new_info,
                jnp.concatenate([buffer_state.avail_actions, current_batch.avail_actions])[-batch_size:]
            )
            
            _, pi = actor_network.apply(
                train_states[0].params,
                initial_hstates[0].squeeze(),
                (buffer_state.obs, buffer_state.done, buffer_state.avail_actions),
            )

            input_action = marginal_actions(buffer_state.action, buffer_state.logits, config["ACT_DIM"], len(env.agents))

            _, value = critic_network.apply(train_states[1].params, initial_hstates[1], (buffer_state.world_state, input_action, buffer_state.done)) 

            traj_batch = Transition(
                buffer_state.global_done,
                buffer_state.done,
                buffer_state.action,
                value,
                buffer_state.reward,
                pi.logits,
                buffer_state.log_prob,
                pi.log_prob(buffer_state.action),
                buffer_state.obs,
                buffer_state.world_state,
                buffer_state.info,
                buffer_state.avail_actions
            )

            rng, _rng = jax.random.split(rng)
            avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
            avail_actions = jax.lax.stop_gradient(
                batchify(avail_actions, env.agents)
            )
            obs_batch = batchify(last_obs, env.agents)
            ac_in = (
                obs_batch[np.newaxis, :],
                last_done[np.newaxis, :],
                avail_actions,
            )
            _, pi = actor_network.apply(train_states[0].params, hstates[0], ac_in)
            action = pi.sample(seed=_rng)
            logits = pi.logits

            world_state = last_obs["world_state"].swapaxes(0,1)  
            input_action = marginal_actions(action, logits, config["ACT_DIM"], len(env.agents))
            
            cr_in = (
                world_state[None, :],  # State input
                input_action,  # Action input
                last_done[np.newaxis, :],  # Done flags
            )
            _, last_val = critic_network.apply(train_states[1].params, hstates[1], cr_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward, pi, mu = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                        transition.current_log_prob,
                        transition.log_prob
                    )
                    logratio = pi - mu
                    ratio = jnp.exp(logratio)
                    true_is = jnp.prod(ratio, axis=0, keepdims=True).repeat(len(env.agents),axis=0)
                    if config["CORRECTION"] == "st":
                        c = jnp.exp(jnp.sum(logratio, axis=0, keepdims=True).repeat(len(env.agents),axis=0))
                        c = jnp.clip(c, 0, 1.0)
                    elif config["CORRECTION"] == "dt":
                        c = jnp.exp(jnp.sum(logratio, axis=0, keepdims=True).repeat(len(env.agents),axis=0) - logratio)
                        c = jnp.clip(jnp.clip(c, 0, config["DT_ETA"]) * ratio, 0, 1.0)
                    elif config["CORRECTION"] == "it":
                        c = jnp.clip(ratio, 0, 1.0)
                    elif config["CORRECTION"] == "true":
                        c = ratio
                    else:
                        c = jnp.ones_like(ratio)
                    
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    if config["ADVANTAGE"] == 'coma':
                        return (gae * c, value), (delta, gae * jnp.clip(ratio, 0, 1.0), c.mean(), jnp.abs(c-true_is).mean(), jnp.abs(c-ratio).mean())
                    return (gae * c, value), (gae, gae * jnp.clip(ratio, 0, 1.0), c.mean(), jnp.abs(c-true_is).mean(), jnp.abs(c-ratio).mean())

                _, (advantages, adv_for_value, c, c_diff, c_diff2) = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, adv_for_value + traj_batch.value, c.mean(), c_diff.mean(), c_diff2.mean()

            advantages, targets, c, c_diff, c_diff2 = _calculate_gae(traj_batch, last_val)
            # print('advantages',advantages)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minibatch(train_states, batch_info):
                    actor_train_state, critic_train_state = train_states
                    ac_init_hstate, cr_init_hstate, traj_batch, advantages, targets = batch_info

                    def _actor_loss_fn(actor_params, init_hstate, traj_batch, gae):
                        # RERUN NETWORK
                        _, pi = actor_network.apply(
                            actor_params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done, traj_batch.avail_actions),
                        )
                        log_prob = pi.log_prob(traj_batch.action)
                        
                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(jnp.clip(logratio,-10,10))
                        old_ratio = jnp.exp(jnp.clip(traj_batch.current_log_prob - traj_batch.log_prob,-10,10))
                        adv = gae.std(axis=1)
                        adv_std = adv.mean()
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)

                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                old_ratio * (1.0 - config["CLIP_EPS"]),
                                old_ratio * (1.0 + config["CLIP_EPS"]),
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()

                        entropy = pi.entropy().mean()
                        approx_kl = ((ratio - 1) - logratio).mean()

                        clip_frac = jnp.mean(jnp.abs(ratio/old_ratio - 1) > config["CLIP_EPS"])
                        abs_ratio = jnp.mean(jnp.abs(ratio - 1))
                        abs_old_ratio = jnp.mean(jnp.abs(old_ratio - 1))
                        
                        actor_loss = loss_actor - config["ENT_COEF"] * entropy
                        
                        return actor_loss, (loss_actor, entropy, ratio, approx_kl, clip_frac, abs_ratio, abs_old_ratio, adv_std)
                    
                    def _critic_loss_fn(critic_params, init_hstate, traj_batch, targets):
                        # RERUN NETWORK
                        input_action = marginal_actions(traj_batch.action, traj_batch.logits, config["ACT_DIM"], len(env.agents))
                        _, value = critic_network.apply(
                            critic_params, 
                            init_hstate.squeeze(), 
                            (traj_batch.world_state, input_action, traj_batch.done), 
                            ) 
                        
                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        critic_loss = config["VF_COEF"] * value_loss

                        return critic_loss, (value_loss, traj_batch.value)

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss, actor_grads = actor_grad_fn(
                        actor_train_state.params, ac_init_hstate, traj_batch, advantages
                    )
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(
                        critic_train_state.params, cr_init_hstate, traj_batch, targets
                    )
                    
                    actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)
                    critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)
                    
                    actor_grad_l2 = jnp.sqrt(tree_l2_norm_squared(actor_grads))
                    critic_grad_l2 = jnp.sqrt(tree_l2_norm_squared(critic_grads))
                                        
                    total_loss = actor_loss[0] + critic_loss[0]
                    loss_info = {
                        "total_loss": total_loss,
                        "actor_loss": actor_loss[0],
                        "value_loss": critic_loss[0],
                        "value": critic_loss[1][1],
                        "entropy": actor_loss[1][1],
                        "ratio": actor_loss[1][2],
                        "approx_kl": actor_loss[1][3],
                        "clip_frac": actor_loss[1][4],
                        "abs_ratio": actor_loss[1][5],
                        "abs_old_ratio": actor_loss[1][6],
                        "adv_std": actor_loss[1][7],
                        "actor_grad_norm": actor_grad_l2,
                        "critic_grad_norm": critic_grad_l2,
                    }
                    
                    return (actor_train_state, critic_train_state), loss_info

                (
                    train_states,
                    init_hstates,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                init_hstates = jax.tree.map(lambda x: jnp.reshape(
                    x, (1, len(env.agents), config["NUM_ENVS"], -1)
                ), init_hstates)
                
                batch = (
                    init_hstates[0],
                    init_hstates[1],
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])

                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=2), batch
                )
                
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.swapaxes(
                            jnp.reshape(
                                x,
                                [*x.shape[:2], config["NUM_MINIBATCHES"], -1]
                                + list(x.shape[3:]),
                            ),
                            2,
                            0,
                        ),
                        1,
                        2,
                        ),
                    shuffled_batch,
                )

                train_states, loss_info = jax.lax.scan(
                    _update_minibatch, train_states, minibatches
                )
                update_state = (
                    train_states,
                    jax.tree.map(lambda x: x.squeeze(), init_hstates),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, loss_info

            update_state = (
                train_states,
                initial_hstates,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            loss_info["ratio_0"] = loss_info["ratio"].at[0,0].get()
            loss_info["c"] = c
            loss_info["c_diff"] = c_diff
            loss_info["c_diff2"] = c_diff2
            loss_info["c_diff_diff"] = c_diff - c_diff2
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            
            train_states = update_state[0]
            metric = traj_batch.info
            metric = jax.tree.map(
                lambda x: x.reshape(
                    (-1, config["NUM_ENVS"], env.num_agents)
                ),
                traj_batch.info,
            )
            metric["loss"] = loss_info
            rng = update_state[-1]

            def callback(metric):
                wandb.log(
                    {
                        # the metrics have an agent dimension, but this is identical
                        # for all agents so index into the 0th item of that dimension.
                        "returns": metric["returned_episode_returns"][:, :, 0][
                            metric["returned_episode"][:, :, 0]
                        ].mean(),
                        "win_rate": metric["returned_won_episode"][:, :, 0][
                            metric["returned_episode"][:, :, 0]
                        ].mean(),
                        "env_step": metric["update_steps"]
                        * config["NUM_ENVS"]
                        * config["NUM_STEPS"],
                        **metric["loss"],
                    }
                )
            
            metric["update_steps"] = update_steps
            jax.experimental.io_callback(callback, None, metric)
            update_steps = update_steps + 1
            runner_state = (train_states, env_state, buffer_state, last_obs, last_done, hstates, rng)
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            (actor_train_state, critic_train_state),
            env_state,
            buffer_state, 
            obsv,
            jnp.zeros((env.num_agents, config["NUM_ENVS"]), dtype=bool),
            (ac_init_hstate, cr_init_hstate),
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}

    return train

def tune(default_config):
    """Hyperparameter sweep with wandb."""

    # default_config = {**default_config, **default_config["alg"]}  # merge the alg config with the main config
    env_name = default_config["ENV_NAME"]
    map_name = default_config["MAP_NAME"]
    alg_name = default_config.get("ALG_NAME", "ACEPO_RNN")

    def wrapped_make_train():
        wandb.init(
            entity=default_config["ENTITY"],
            project=default_config["PROJECT"],
            tags=["ACEPO", "RNN", default_config["MAP_NAME"]],
            config=default_config,
            mode=default_config["WANDB_MODE"],
        )

        # update the default params
        config = deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config[k] = v

        print("running experiment with params:", config)

        rng = jax.random.PRNGKey(config["SEED"])
        with jax.disable_jit(False):
            train_jit = jax.jit(make_train(config)) 
            out = train_jit(rng)

    sweep_config = {
        "name": f"{env_name}_{map_name}",
        "method": "bayes",
        "metric": {
            "name": "win_rate",
            "goal": "maximize",
        },
        "parameters": {
            "LR": {
                "values": [
                    0.00005,0.0001,0.0003,0.0005,0.001
                ]
            },
            "NUM_STEPS":{"values": [128,256,400]},
            "NUM_ENVS":{"values":[32,64,128,256,512]},
            "NUM_MINIBATCHES":{"values":[1,2,4]},
            
            #"MAP_NAME": {"values": ['5m_vs_6m', '10m_vs_11m', '3s5z_vs_3s6z','6h_vs_8z']},
        },
    }

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=300)
    

@hydra.main(version_base=None, config_path="config", config_name="rnn_smax")
def main(config):

    config = OmegaConf.to_container(config, resolve=True)
    print("Config:\n", OmegaConf.to_yaml(config))

    if config["HYP_TUNE"]:
        tune(config)
    else:
        wandb.init(
            entity=config["ENTITY"],
            project=config["PROJECT"],
            tags=[config["ADVANTAGE"], "RNN", config["MAP_NAME"]],
            config=config,
            mode=config["WANDB_MODE"],
        )
        rng = jax.random.PRNGKey(config["SEED"])
        with jax.disable_jit(False):
            train_jit = jax.jit(make_train(config)) 
            out = train_jit(rng)

if __name__=="__main__":
    main()