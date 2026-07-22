import popgym
from popgym.wrappers import PreviousAction, Antialias, Markovian, Flatten, DiscreteAction
from popgym.core.observability import Observability, STATE
import gymnasium as gym



#env_classes = popgym.envs.ALL.keys()
#print(env_classes)
env = popgym.envs.position_only_cartpole.PositionOnlyCartPoleEasy()
print(f"observation space : {env.observation_space.shape}")
env = PreviousAction(env)
env = Antialias(env)
print(f"observation space after prev act : {env.observation_space}")
env = Flatten(env, flatten_action=False)  # gym.wrappers.FlattenObservation(env)
print(f"observation space after flatten : {env.observation_space.low}")
#env = DiscreteAction(env)
env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print("observation :")
print(obs)
print("infor :")
print(info)
print("action space :")
print(env.action_space.sample())


env = gym.make("CartPole-v1")
env = popgym.envs.position_only_cartpole.PositionOnlyCartPoleEasy()
print(env.observation_space.low)