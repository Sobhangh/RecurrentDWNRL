# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
import warnings
from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from tqdm import tqdm
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from RNNDWN import RNNDWN
from thermometer import ThermometerGaussian
import popgym
from NavEnv import GridNavEnv



@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ReccurentDLGRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "GridNav"
    """the id of the environment"""
    board_dim: int = 16
    """the dimension of the board for the GridNav environment"""
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 64
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    WNN_learning_rate: float = 5e-3
    """the learning rate of the optimizer"""

    # WNN
    wnn_agent: bool = True
    """" if toggled, the agent will be a WNN agent"""
    hidden_size: int = 1500
    """the size of the hidden layers of the RNNDWN"""
    bits: int = 63
    """the number of bits per input dimension for the thermometer"""
    n: int = 4
    """number of LUT inputs"""
    nb_layers: int = 2
    """number of hidden layers"""


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        # if capture_video and idx == 0:
        #     env = gym.make(env_id, render_mode="rgb_array")
        #     env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # else:
        #    env = gym.make(env_id)
        if capture_video and idx == 0:
            env = GridNavEnv(dimension=args.board_dim, render_mode="rgb_array")
            # Gymnasium may import moviepy for video recording, which can emit
            # a Python 3.12 SyntaxWarning from moviepy's legacy config file.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*invalid escape sequence '\\P'.*",
                    category=SyntaxWarning,
                    module=r"moviepy\.config_defaults",
                )
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = GridNavEnv(dimension=args.board_dim, render_mode="rgb_array")
            #print(f"env_id={env_id}, idx={idx}, capture_video={capture_video}, run_name={run_name}")
        #env = popgym.envs.position_only_cartpole.PositionOnlyCartPoleEasy()
        #env = gym.wrappers.FlattenObservation(env) 
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

def evaluate(
    agent,
    train_enviroment,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str = "eval",
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
    writer=None,
    global_step= 0,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, capture_video, run_name)])
    agent.eval()

    obs, _ = envs.reset()
    if args.wnn_agent:
        eval_hidden = agent.actor.init_hidden(batch_size=1, device=device, dtype=torch.float32)
        eval_rnn_hidden = torch.zeros(agent.critic_rnn.num_layers, 1, agent.critic_rnn.hidden_size, device=device, dtype=torch.float32)
    else:
        eval_hidden = torch.zeros(agent.memory.num_layers, 1, agent.memory.hidden_size, device=device, dtype=torch.float32)
    eval_done = torch.zeros(1, device=device, dtype=torch.float32)
    episodic_returns = []
    episodic_lengths = []
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            if args.wnn_agent:
                actions, _, _, _, eval_hidden, eval_rnn_hidden = agent.get_action_and_value(
                    torch.Tensor(obs).to(device),
                    eval_hidden,
                    eval_rnn_hidden,
                    eval_done,
                )
            else:
                actions, _, _, _, eval_hidden = agent.get_action_and_value(
                    torch.Tensor(obs).to(device),
                    eval_hidden,
                    eval_done,
                )
        next_obs, _, terminations, truncations, infos = envs.step(actions.cpu().numpy())
        eval_done = torch.as_tensor(np.logical_or(terminations, truncations), device=device, dtype=torch.float32)
        if "episode" in infos:
            #print("final info")
            ep_mask = infos.get("_episode", np.logical_or(terminations, truncations))
            ep = infos["episode"]
            for i, ended in enumerate(ep_mask):
                if ended:
                    r = float(ep["r"][i])
                    l = int(ep["l"][i])
                    episodic_returns += [r]
                    episodic_lengths += [l]
            print(f"eval_episode={len(episodic_returns)} out of {eval_episodes}")
        elif "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
                episodic_lengths += [info["episode"]["l"]]
        obs = next_obs
    ret_mean = float(np.mean(episodic_returns))
    ret_std = float(np.std(episodic_returns))
    len_mean = float(np.mean(episodic_lengths))
    if writer is not None:
        writer.add_scalar("eval/episodic_return_mean", ret_mean, global_step or 0)
        writer.add_scalar("eval/episodic_return_std", ret_std, global_step or 0)
        writer.add_scalar("eval/episodic_length_mean", len_mean, global_step or 0)
    agent.train()
    return episodic_returns

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()

        obs_dim = np.array(envs.single_observation_space.shape).prod()
        self.memory = nn.RNN(
            input_size=int(obs_dim),
            hidden_size=int(64),
            num_layers=3,
            nonlinearity="tanh",
            #batch_first=True,
        )
        self.critic = nn.Sequential(
            # layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            # nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            # layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            # nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_states(self, x, hidden_state, done):
        # RNN logic
        batch_size = hidden_state.shape[1]
        x = x.reshape((-1, batch_size, self.memory.input_size))
        done = done.reshape((-1, batch_size))
        new_x = []
        for h, d in zip(x, done):
            h, hidden_state = self.memory(
                h.unsqueeze(0),
                (1.0 - d).view(1, -1, 1) * hidden_state,
            )
            new_x += [h]
        new_x = torch.flatten(torch.cat(new_x), 0, 1)
        return new_x, hidden_state
    
    def get_value(self, x, hidden_state, done):
        x, _ = self.get_states(x, hidden_state, done)
        return self.critic(x)

    def get_action_and_value(self, x, hidden_state, done, action=None):
        x, hidden_state = self.get_states(x, hidden_state, done)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x), hidden_state


class WNNActor(nn.Module):
    def __init__(self, envs, args):
        super().__init__()

        obs_dim = np.array(envs.single_observation_space.shape).prod()
        print(f"Observation dimension: {obs_dim}")
        act_dim = envs.single_action_space.n
        print(f"Action dimension: {act_dim}")
        thermo_device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"

        #thermo = ThermometerGaussian(n_bits=args.bits, device=thermo_device)
        #min_values = envs.single_observation_space.low
        #max_values = envs.single_observation_space.high
        #thermo.fit(torch.zeros((1, obs_dim), device=thermo_device), min_value=min_values, max_value=max_values)
        #print(f"Thermometer threshold values: {thermo.thresholds}")

        init_log_alpha = args.init_log_alpha if hasattr(args, "init_log_alpha") else -0.6931
        self.actor = RNNDWN(
            input_dim=obs_dim,
            hidden_size=args.hidden_size,
            output_dim=act_dim,
            num_layers=args.nb_layers,
            thresholds=None,
            bits=args.bits,
            n=args.n,
            init_log_alpha=init_log_alpha,
        )

        # self.critic = nn.Sequential(
        #     layer_init(nn.Linear(obs_dim, 64)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, 64)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, 1), std=1.0),
        # )
        class EnsureSeqDim(nn.Module):
            def forward(self, x):
                return x.unsqueeze(1) if x.dim() == 2 else x

        class RNNOutputLastStep(nn.Module):
            def forward(self, rnn_out):
                output, _ = rnn_out
                return output[:, -1, :]

        self.critic_rnn = nn.RNN(
                input_size=int(obs_dim),
                hidden_size=int(64),
                num_layers=3,
                nonlinearity="tanh",
                #batch_first=True,
        )
        self.critic = nn.Sequential(
            nn.Linear(int(64), 1),
        )

    def get_states(self, network, x, hidden_state, done):
            # RNN logic
            batch_size = hidden_state.shape[1]
            #print(x.shape, hidden_state.shape, done.shape)
            if isinstance(network, RNNDWN):
                x = x.reshape((-1, batch_size, network.input_dim))
            else:
                x = x.reshape((-1, batch_size, network.input_size))
            #print(f"x reshaped to: {x.shape}")
            done = done.reshape((-1, batch_size))
            new_x = []
            for h, d in zip(x, done):
                if isinstance(network, RNNDWN):
                    h, hidden_state = network(
                        h, #.unsqueeze(0),
                        (1.0 - d).view(1, -1, 1) * hidden_state,
                    )
                    new_x += [h.unsqueeze(0)]
                else:
                    h, hidden_state = network(
                        h.unsqueeze(0),
                        (1.0 - d).view(1, -1, 1) * hidden_state,
                    )
                    new_x += [h]
            new_x = torch.flatten(torch.cat(new_x), 0, 1)
            return new_x, hidden_state
    
    def get_value(self, x, hidden_state, done):
        x, _ = self.get_states(self.critic_rnn, x, hidden_state, done)
        return self.critic(x)

    def get_action_and_value(self, x, hidden_state, rnn_hidden_state, done, action=None):
        # if x.dim() != 2:
        #     raise ValueError("x must have shape (batch_size, features)")

        # if hidden_state is None:
        #     hidden_state = self.actor.init_hidden(batch_size=x.shape[0], device=x.device, dtype=x.dtype)

        # batch_size = hidden_state.shape[1]
        # is_sequence_batch = x.shape[0] != batch_size

        # if done is None:
        #     done = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        # else:
        #     done = done.to(device=x.device, dtype=x.dtype).view(-1)

        # if is_sequence_batch:
        #     if x.shape[0] % batch_size != 0:
        #         raise ValueError("Sequence batch does not align with hidden-state batch size")
        #     seq_len = x.shape[0] // batch_size
        #     x_seq = x.view(seq_len, batch_size, -1)
        #     done_seq = done.view(seq_len, batch_size)
        #     if action is not None:
        #         action_seq = action.view(seq_len, batch_size)
        #     else:
        #         action_seq = None
        # else:
        #     seq_len = 1
        #     x_seq = x.view(1, batch_size, -1)
        #     done_seq = done.view(1, batch_size)
        #     if action is not None:
        #         action_seq = action.view(1, batch_size)
        #     else:
        #         action_seq = None

        # logits_steps = []
        # next_hidden = hidden_state
        # for t in range(seq_len):
        #     next_hidden = next_hidden * (1.0 - done_seq[t]).view(1, batch_size, 1)
        #     logits_t, next_hidden = self.actor(x_seq[t], next_hidden)
        #     logits_steps.append(logits_t)

        # logits = torch.stack(logits_steps, dim=0)
        # logits_flat = logits.reshape(-1, logits.shape[-1])
        # probs = Categorical(logits=logits_flat)

        # if action_seq is None:
        #     sampled = probs.sample()
        #     action_out = sampled if is_sequence_batch else sampled.view(batch_size)
        #     logprob = probs.log_prob(sampled)
        # else:
        #     action_flat = action_seq.reshape(-1).long()
        #     action_out = action_flat if is_sequence_batch else action_flat.view(batch_size)
        #     logprob = probs.log_prob(action_flat)

        # entropy = probs.entropy()
        critic_input, rnn_hidden_state = self.get_states(self.critic_rnn, x, rnn_hidden_state, done)
        logits, hidden_state = self.get_states(self.actor, x, hidden_state, done)
        
        #logits = self.actor(x)
        probs = Categorical(logits=logits)
        value = self.critic(critic_input)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value, hidden_state, rnn_hidden_state


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    print(f"Run name: {run_name}")
    print(f"Batch size: {args.batch_size}, Minibatch size: {args.minibatch_size}, Num iterations: {args.num_iterations}")
    if args.track:
        import wandb
        from cred import cred
        wandb.login(key=cred)
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    print("environment setup done")
    if args.wnn_agent:
        print("Using WNN agent")
        agent = WNNActor(envs, args).to(device)
    else:
        print("Using Normal agent")
        agent = Agent(envs).to(device)
    if args.wnn_agent:
        optimizer = optim.Adam(
            [
                {
                    "params": (
                        list(agent.actor.parameters())
                    ),
                    "lr": args.WNN_learning_rate,
                },
                {
                    "params": list(agent.critic_rnn.parameters()) + list(agent.critic.parameters()),
                    "lr": args.learning_rate,
                },
            ],
            eps=1e-5,
        )
        base_lrs = [args.WNN_learning_rate, args.learning_rate]
    else:   
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
        base_lrs = [args.learning_rate]
    episode_rewards_running_mean = 0
    
    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    if args.wnn_agent:
        next_actor_hidden = agent.actor.init_hidden(batch_size=args.num_envs, device=device, dtype=next_obs.dtype)
        next_critic_hidden = torch.zeros(agent.critic_rnn.num_layers, args.num_envs, agent.critic_rnn.hidden_size).to(device)
    else:
        next_actor_hidden = torch.zeros(agent.memory.num_layers, args.num_envs, agent.memory.hidden_size).to(device)
    
    print("Starting training...")
    for iteration in tqdm(range(1, args.num_iterations + 1)):
        initial_actor_state = next_actor_hidden.clone()
        if args.wnn_agent:
            initial_critic_state = next_critic_hidden.clone()

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
                param_group["lr"] = frac * base_lr
            # lrnow = frac * args.learning_rate if not args.wnn_agent else frac * args.WNN_learning_rate
            # optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            #print(f"Iteration {iteration}, Step {step}, Global Step {global_step}")
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                if args.wnn_agent:
                    action, logprob, _, value, next_actor_hidden, next_critic_hidden = agent.get_action_and_value(
                        next_obs,
                        next_actor_hidden,
                        next_critic_hidden,
                        next_done,
                    )
                else:
                    action, logprob, _, value, next_actor_hidden = agent.get_action_and_value(
                        next_obs,
                        next_actor_hidden,
                        next_done,
                    )
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            #print(f"infos: {infos}")
            if "episode" in infos:
                #print("final info")
                ep_mask = infos.get("_episode", np.logical_or(terminations, truncations))
                ep = infos["episode"]
                for i, ended in enumerate(ep_mask):
                    if ended:
                        r = float(ep["r"][i])
                        l = int(ep["l"][i])
                        episode_rewards_running_mean = 0.95 * episode_rewards_running_mean + 0.05 * r

                        #print(f"global_step={global_step}, episodic_return={r}")
                        writer.add_scalar("charts/episodic_return", r, global_step)
                        writer.add_scalar("charts/episodic_return_running_mean", episode_rewards_running_mean, global_step)
                        writer.add_scalar("charts/episodic_length", l, global_step)
            elif "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        #print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            if args.wnn_agent:
                next_value = agent.get_value(next_obs, next_critic_hidden, next_done).reshape(1, -1)
            else:
                next_value = agent.get_value(next_obs, next_actor_hidden, next_done).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        clipfracs = []
        for epoch in range(args.update_epochs):
            #print(f"        Epoch {epoch}")
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index
                mb_hidden = initial_actor_state[:, mbenvinds].contiguous()
                if args.wnn_agent:
                    mb_critic_hidden = initial_critic_state[:, mbenvinds].contiguous()
                    _, newlogprob, entropy, newvalue, _, _ = agent.get_action_and_value(
                        b_obs[mb_inds],
                        mb_hidden,
                        mb_critic_hidden,
                        b_dones[mb_inds],
                        b_actions.long()[mb_inds],
                    )
                else:
                    _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                        b_obs[mb_inds],
                        mb_hidden,
                        b_dones[mb_inds],
                        b_actions.long()[mb_inds],
                    )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        #print(f"Iteration {iteration} from {args.num_iterations}, SPS={int(global_step / (time.time() - start_time))}, value_loss={v_loss.item()}, policy_loss={pg_loss.item()}, entropy={entropy_loss.item()}, old_approx_kl={old_approx_kl.item()}, approx_kl={approx_kl.item()}, clipfrac={np.mean(clipfracs)}, explained_variance={explained_var}")
        # if we are at //20 of iterations, evaluate
        eval_every = max(args.num_iterations // 1, 1)
        # print(iteration, eval_every)
        if (((iteration) % eval_every == 0)):
                episodic_returns = evaluate(
                    agent,
                    envs,
                    make_env,
                    args.env_id,
                    eval_episodes = 10,
                    device = device,
                    capture_video= False,
                    writer=writer,
                    global_step=global_step,
                )
                ret_mean = float(np.mean(episodic_returns))
                ret_std = float(np.std(episodic_returns))
                print(f"Iteration {iteration} from {args.num_iterations}, eval_episodic_return_mean={ret_mean}, eval_episodic_return_std={ret_std}")


    envs.close()
    writer.close()
