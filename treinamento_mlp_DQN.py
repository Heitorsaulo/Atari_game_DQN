import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.dqn import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.dqn import CnnPolicy
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.common.callbacks import BaseCallback
from Ram.dict_ram_values import faixas


def verify_car_lane(pos_chicken):
    index = 0
    for pos_car in faixas.values():
        if pos_chicken in pos_car:
            return index
        index += 1


def check_collision(cars, lane_pos):
    if cars[lane_pos] in list(range(39, 54)):
        return True
    else:
        return False


def check_collision_with_action(action, score, past_score):
    return action != 2 and score == past_score


class FreewayRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_score = 0
        self.last_player_pos = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        ram = self.env.unwrapped.ale.getRAM()
        self.last_score = ram[103]  # Inicializa a pontuação
        self.last_player_pos = ram[14]  # Inicializa a posição do frango
        self.current_episode_rewards = []  # Limpa as recompensas do episódio atual
        return obs, info

    def reward(self, ram, action):
        player_pos_y = ram[14]  # Posição do frango
        score = ram[103]  # Pontuação do jogo
        cars_x = ram[108:118]
        # Recompensa se marcar ponto
        reward = 0.0
        # linha do jogador
        lane_pos = verify_car_lane(player_pos_y)
        # Recompensa adicional por se mover para cima
        if player_pos_y > self.last_player_pos:
            reward += 0.075
        if (check_collision(cars_x, lane_pos) or check_collision_with_action(action, score, self.last_score)) and player_pos_y < self.last_player_pos:
            reward -= 0.80
        if player_pos_y == self.last_player_pos:
            reward -= 0.035
        if score > self.last_score:
            reward = (score - self.last_score) * 2.57  # Amplificar impacto da pontuação

        # Atualizar estados anteriores
        self.last_score = score
        self.last_player_pos = player_pos_y

        return reward

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        ram = self.env.unwrapped.ale.getRAM()
        reward = self.reward(ram, action)

        return obs, reward, done, truncated, info


class RewardCallback(BaseCallback):
    """
    Callback personalizado para armazenar a recompensa média por episódio.
    """

    def __init__(self, check_freq=1000):
        super().__init__()
        self.check_freq = check_freq
        self.episode_rewards = []
        self.mean_rewards = []
        self.episode_times = []
        self.mean_times = []

    def _on_step(self) -> bool:
        """Chamado a cada passo do treinamento."""
        if "episode" in self.locals["infos"][0]:
            ep_reward = self.locals["infos"][0]["episode"]["r"]  # Recompensa do episódio
            print(ep_reward)
            self.episode_rewards.append(ep_reward)
            self.mean_rewards.append(np.mean(self.episode_rewards))
            ep_time = self.locals["infos"][0]["episode"]["t"]  # tempo do episodio
            self.episode_times.append(ep_time)
            self.mean_times.append(np.mean(self.episode_times))
            print(ep_time)
        return True

    def plot_rewards(self):
        """Gera o gráfico da evolução da recompensa média."""
        plt.plot(self.mean_rewards)
        plt.xlabel("Episódios")
        plt.ylabel("Média Acumulada das Recompensas")
        plt.title("Evolução da Recompensa Média Durante o Treinamento")
        plt.show()

    def plot_times(self):
        """Gera o gráfico da evolução da recompensa média."""
        plt.plot(self.mean_times)
        plt.xlabel("Episódios")
        plt.ylabel("Média Acumulada do tempo(s)")
        plt.title("Tempo médio de treinamento dos episódios")
        plt.show()


env = gym.make("ALE/Freeway-v5", render_mode="rgb_array")
env = FreewayRewardWrapper(env)
env = make_vec_env(lambda: env, n_envs=1)
reward_callback = RewardCallback()

model = DQN(MlpPolicy,
            env,
            learning_rate=1e-3,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=64,
            gamma=0.84,
            target_update_interval=1000,
            train_freq=4,
            exploration_fraction=0.12,
            exploration_final_eps=0.04,
            verbose=1,
            device="auto")

model.learn(total_timesteps=500000, callback=reward_callback)
model.save("D:/PycharmProjects/AtariDQN/modelos/freeway_mlp_0.075_500000_g0.84.zip")

reward_callback.plot_rewards()
reward_callback.plot_times()

# Testar o wrapper
env = gym.make("ALE/Freeway-v5", render_mode="human")
env = FreewayRewardWrapper(env)  # Aplica o wrapper de recompensa
env = make_vec_env(lambda: env, n_envs=1)
model = DQN.load("D:/PycharmProjects/AtariDQN/modelos/freeway_mlp_0.075_500000_g0.84.zip"", env=env)
obs = env.reset()
total_reward = []
acc_reward = 0
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated = env.step(action)

    # Extrair valores corretos do ambiente vetorizado
    print(f"Ação: {action}, Recompensa: {reward}")
    acc_reward += reward
    total_reward.append(acc_reward)
    env.render()
env.close()