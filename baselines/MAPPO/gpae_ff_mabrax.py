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
import chex

from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import DictConfig, OmegaConf
from functools import partial
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper, JaxMARLWrapper
from jaxmarl.environments.multi_agent_env import MultiAgentEnv, State


import wandb
import functools
import matplotlib.pyplot as plt

    
class MABraxWorldStateWrapper(JaxMARLWrapper):
    
    @partial(jax.jit, static_argnums=0)
    def reset(self,
              key):
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self.world_state(obs, env_state)
        
        return obs, env_state
    
    @partial(jax.jit, static_argnums=0)
    def step(self,
             key,
             state,
             action):
        obs, env_state, reward, done, info = self._env.step(
            key, state, action
        )
        obs["world_state"] = self.world_state(obs, env_state)
        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def world_state(self, obs, state):
        """ 
        For each agent: [agent obs, own hand]
        """
        one_hot = jnp.eye(self._env.num_agents)
        world_state = jnp.array([state.obs for agent in self._env.agents])
        print(world_state)
        #return jnp.concatenate((world_state, one_hot), axis=1)
        return world_state
        # hands = state.player_hands.reshape((self._env.num_agents, -1))
        # return jnp.concatenate((all_obs, hands), axis=1)
        
    
    def world_state_size(self):
        return self._env.state_spaces[self._env.agents[0]].shape[0] #+ self._env.num_agents


class ActorFF(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, x):
        obs = x
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

        actor_mean = nn.Dense(self.action_dim)(x)
        actor_logtstd = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
        #print(x.shape, actor_mean.shape, actor_logtstd.shape)
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        return pi

class CriticFF(nn.Module):
    config: Dict
    
    @nn.compact
    def __call__(self, x):
        world_state, action = x

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

        q_value = nn.Dense(1)(embedding)

        return jnp.squeeze(q_value, axis=-1)

class Transition(NamedTuple):
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


def batchify(x: dict, agent_list):
    max_dim = max([x[a].shape[-1] for a in agent_list])
    def pad(z):
        return jnp.concatenate([z, jnp.zeros(z.shape[:-1] + (max_dim - z.shape[-1],))], -1)

    x = jnp.stack([x[a] if x[a].shape[-1] == max_dim else pad(x[a]) for a in agent_list])
    return x


def unbatchify(x: jnp.ndarray, agent_list, num_envs):
    x = x.reshape((len(agent_list), num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def marginal_actions(acts, logits, act_dim, num_agents):
    rollouts = acts.shape[2]
    steps = acts.shape[0]
    input_action = acts.reshape(steps, 1, rollouts, num_agents, act_dim)
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
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["NUM_ENVS"] = int(config["NUM_ENVS"] / config["BUFFER_SIZE"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = MABraxWorldStateWrapper(env)
    env = LogWrapper(env, replace_info=True)
    
    config["ACT_DIM"] = env.action_space(env.agents[0]).shape[0]
    config["OBS_DIM"] = env.observation_space(env.agents[0]).shape[0]
    config["STATE_DIM"] = env.world_state_size()
    print(config["ACT_DIM"],config["OBS_DIM"],config["STATE_DIM"])
    config["CLIP_EPS"] = config["CLIP_EPS"] / env.num_agents if config["SCALE_CLIP_EPS"] else config["CLIP_EPS"]

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        actor_network = ActorFF(config["ACT_DIM"], config=config)
        max_dim = jnp.argmax(jnp.array([env.observation_space(a).shape[-1] for a in env.agents]))
        critic_network = CriticFF(config)
        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)

        ac_init_x = jnp.zeros((1, config["NUM_ENVS"], config["OBS_DIM"]))
        actor_network_params = actor_network.init(_rng_actor, ac_init_x)
        
        cr_init_x = (jnp.zeros((1, config["NUM_ENVS"], config["STATE_DIM"])),
            jnp.zeros((1, config["NUM_ENVS"], config["ACT_DIM"] * len(env.agents))),
        )
        
        critic_network_params = critic_network.init(_rng_critic, cr_init_x)
        
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
        obsv, env_state = jax.vmap(env.reset)(reset_rng)

        traj_info = {"returned_episode": jnp.zeros((config["NUM_STEPS"] * config["BUFFER_SIZE"], len(env.agents), config["NUM_ENVS"]), dtype=bool), 
            "returned_episode_lengths": jnp.zeros((config["NUM_STEPS"] * config["BUFFER_SIZE"], len(env.agents), config["NUM_ENVS"]), dtype=jnp.float32),
            "returned_episode_returns":jnp.zeros((config["NUM_STEPS"] * config["BUFFER_SIZE"], len(env.agents), config["NUM_ENVS"]), dtype=jnp.float32)
            }
        buffer_state = Transition(
            jnp.zeros((config["NUM_STEPS"] * config["BUFFER_SIZE"], len(env.agents), config["NUM_ENVS"]), dtype=bool),
            jnp.zeros((config["NUM_STEPS"] * config["BUFFER_SIZE"], len(env.agents), config["NUM_ENVS"],config["ACT_DIM"]), dtype=jnp.float32),
            jnp.zeros((config["NUM_STEPS"] * config["BUFFER_SIZE"], len(env.agents), config["NUM_ENVS"]), dtype=jnp.float32),
            jnp.zeros((config["NUM_STEPS"] * config["BUFFER_SIZE"], len(env.agents), config["NUM_ENVS"]), dtype=jnp.float32),
            jnp.zeros((config["NUM_STEPS"] * config["BUFFER_SIZE"], len(env.agents), config["NUM_ENVS"],config["ACT_DIM"]), dtype=jnp.float32),
            jnp.zeros((config["NUM_STEPS"] * config["BUFFER_SIZE"], len(env.agents), config["NUM_ENVS"]), dtype=jnp.float32),
            jnp.zeros((config["NUM_STEPS"] * config["BUFFER_SIZE"], len(env.agents), config["NUM_ENVS"]), dtype=jnp.float32),
            jnp.zeros((config["NUM_STEPS"] * config["BUFFER_SIZE"], len(env.agents), config["NUM_ENVS"],config["OBS_DIM"]), dtype=jnp.float32),
            jnp.zeros((config["NUM_STEPS"] * config["BUFFER_SIZE"], len(env.agents), config["NUM_ENVS"],config["STATE_DIM"]), dtype=jnp.float32),
            traj_info,
        )

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state
            
            def _env_step(runner_state, unused):
                train_states, env_state, buffer_state, last_obs, update_count, rng = runner_state

                obs_batch = batchify(last_obs, env.agents)
                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                
                pi = actor_network.apply(train_states[0].params, obs_batch[np.newaxis, :])
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                logits = pi.mode()
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"])
                
                # VALUE
                world_state = last_obs["world_state"].swapaxes(0,1)
                
                # world_state = world_state.reshape((config["NUM_ACTORS"],-1))
                input_action = marginal_actions(action, logits, config["ACT_DIM"], len(env.agents))
                value = critic_network.apply(train_states[1].params, (world_state[None, :], input_action))

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act,
                )
                info = jax.tree.map(lambda x: x.reshape((len(env.agents), config["NUM_ENVS"])), info)
                transition = Transition(
                    batchify(done, env.agents).squeeze(),
                    action.squeeze(0),
                    value.squeeze(),
                    batchify(reward, env.agents).squeeze(),
                    logits.squeeze(0),
                    log_prob.squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    world_state,
                    info,
                )
                runner_state = (train_states, env_state, buffer_state, obsv, update_count, rng)
                return runner_state, transition

            runner_state, current_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            train_states, env_state, buffer_state, last_obs, update_count, rng = runner_state

            batch_size = config["NUM_STEPS"] * config["BUFFER_SIZE"]

            new_info = {k:jnp.concatenate([buffer_state.info[k], current_batch.info[k]])[-batch_size:] for k in buffer_state.info.keys()}

            buffer_state = Transition(
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
            )
            
            pi = actor_network.apply(
                train_states[0].params,
                buffer_state.obs
            )

            input_action = marginal_actions(buffer_state.action, buffer_state.logits, config["ACT_DIM"], len(env.agents))

            value = critic_network.apply(train_states[1].params, (buffer_state.world_state, input_action)) 

            traj_batch = Transition(
                buffer_state.done,
                buffer_state.action,
                value,
                buffer_state.reward,
                pi.mode(),
                buffer_state.log_prob,
                pi.log_prob(buffer_state.action),
                buffer_state.obs,
                buffer_state.world_state,
                buffer_state.info,
            )

            obs_batch = batchify(last_obs, env.agents)
            pi = actor_network.apply(train_states[0].params, obs_batch[np.newaxis, :])
            action = pi.sample(seed=_rng)
            logits = pi.mode()
      
            last_world_state = last_obs["world_state"].swapaxes(0,1)
            input_action = marginal_actions(action, logits, config["ACT_DIM"], len(env.agents))
            # last_world_state = last_world_state.reshape((config["NUM_ACTORS"],-1))
            last_val = critic_network.apply(train_states[1].params, (last_world_state[None,:], input_action))
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward, pi, mu = (
                        transition.done,
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

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_states, batch_info):
                    actor_train_state, critic_train_state = train_states
                    traj_batch, advantages, targets = batch_info

                    def _actor_loss_fn(actor_params, traj_batch, gae):
                        # RERUN NETWORK
                        pi = actor_network.apply(
                            actor_params,
                            traj_batch.obs,
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(jnp.clip(logratio,-50,50))
                        old_ratio = jnp.exp(jnp.clip(traj_batch.current_log_prob - traj_batch.log_prob,-50,50))
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
                    
                    def _critic_loss_fn(critic_params, traj_batch, targets):
                        
                        # RERUN NETWORK
                        input_action = marginal_actions(traj_batch.action, traj_batch.logits, config["ACT_DIM"], len(env.agents))
                        value = critic_network.apply(
                            critic_params, 
                            (traj_batch.world_state, input_action), 
                            ) 
                        
                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = optax.huber_loss(value, targets, delta=10)
                        value_losses_clipped = optax.huber_loss(value_pred_clipped, targets, delta=10)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        critic_loss = config["VF_COEF"] * value_loss

                        return critic_loss, (value_loss, traj_batch.value)

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss, actor_grads = actor_grad_fn(
                        actor_train_state.params, traj_batch, advantages
                    )
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(
                        critic_train_state.params, traj_batch, targets
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

                train_states, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                batch = (traj_batch, advantages.squeeze(), targets.squeeze())
                # batch = jax.tree.map(
                #     lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                # )
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
                    _update_minbatch, train_states, minibatches
                )
                update_state = (train_states, traj_batch, advantages, targets, rng)
                return update_state, loss_info

            def callback(metric):
                if not metric["update_step"] % config["LOG_TERM"]:
                    wandb.log(
                        metric,
                        step=metric["env_step"],
                    )
                
            update_state = (train_states, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )            
            train_states = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            update_count = update_count + 1
            loss_info["ratio_0"] = loss_info["ratio"].at[0,0].get()
            loss_info["c"] = c
            loss_info["c_diff"] = c_diff
            loss_info["c_diff2"] = c_diff2
            loss_info["c_diff_diff"] = c_diff - c_diff2
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            metric = jax.tree.map(lambda x: x.mean(), metric)
            metric["update_step"] = update_count
            metric["env_step"] = update_count * config["NUM_STEPS"] * config["NUM_ENVS"]
            metric = {**metric, **loss_info}
            jax.experimental.io_callback(callback, None, metric)
            runner_state = (train_states, env_state, buffer_state, last_obs, update_count, rng)
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = ((actor_train_state, critic_train_state), env_state, buffer_state, obsv, 0, _rng)
        #runner_state = ((actor_train_state, critic_train_state), env_state, obsv, jnp.zeros((config["NUM_ACTORS"]), dtype=bool), _rng,)
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train

@hydra.main(version_base=None, config_path="config", config_name="ff_mabrax")
def main(config):

    config = OmegaConf.to_container(config)
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["ACEPO", "FF", config["ENV_NAME"]],
        config=config,
        mode=config["WANDB_MODE"],
    )
    rng = jax.random.PRNGKey(config["SEED"])
    with jax.disable_jit(False):
        train_jit = jax.jit(make_train(config)) 
        out = train_jit(rng)

    
if __name__=="__main__":
    main()
