"""
DESCRIÇÃO DO JOGO FREEWAY:

A galinha atravessa a rua para chegar ao outro lado, mas porque não ganhar pontos enquanto faz isso? Em Freeway, seu objetivo é cruzar uma gigantesca estrada, o maior número de vezes, com suas galinhas.
Caso seja atingido por um veículo, sua galinha não irá morrer, mas jogada para traz, tendo que fazer o caminho de volta.


Este código foi testado em uma máquina com Apple Silicon (M1). Porém, também é possível executar em outras GPUs.

"""

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.atari_wrappers import (
    NoopResetEnv,
    MaxAndSkipEnv,
    EpisodicLifeEnv,
    WarpFrame
)
from stable_baselines3.dqn import CnnPolicy, MlpPolicy
from stable_baselines3.common.callbacks import BaseCallback
import time
import os
import ale_py
import numpy as np
from gymnasium import Wrapper
import torch
import psutil


# Checa se o MPS (Apple Silicon GPU) está disponível
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA GPU (CUDA)")
else:
    device = torch.device("cpu")
    print("Using CPU")


SEED = 42
TOTAL_TIMESTEPS = 500_000
FRAME_SKIP = 1
LEARNING_RATE = 0.001
NUM_ENVS = 4  # Número de ambientes paralelos

class CrossingRewardWrapper(Wrapper):
    def __init__(self, env, crossing_reward=15.0, reward=5.0, penalty=-1.0):
        super().__init__(env)
        self.crossing_reward = crossing_reward
        self.reward = reward
        self.penalty = penalty
        self.last_y = None
        self.crossed = False
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_y = self._get_chicken_y(obs)
        self.crossed = False
        return obs, info
    
    def _get_chicken_y(self, obs):
        white_pixels = np.where(obs > 200)
        if len(white_pixels[0]) > 0:
            return np.mean(white_pixels[0])
        return None
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_y = self._get_chicken_y(obs)
        
        if self.last_y is not None and current_y is not None:
            if current_y < self.last_y and action == 1:  # Colisão durante movimento para frente
                reward = self.penalty * 2
            elif action == 2:  # Penalidade severa por movimento para trás desnecessário
                reward = self.penalty * 3
            elif current_y > self.last_y and action == 1:  # Recompensa por movimento para frente
                reward = self.reward * 2
            elif current_y < self.last_y and not self.crossed:  # Travessia completa
                reward = self.crossing_reward
                self.crossed = True
        
        self.last_y = current_y
        return obs, reward, terminated, truncated, info

class SmartScoreCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_rewards = None
        self.steps_in_episode = None
        self.collisions = None
        self.num_envs = None

    def _on_training_start(self):
        self.num_envs = self.training_env.num_envs
        self.current_rewards = [0] * self.num_envs
        self.steps_in_episode = [0] * self.num_envs
        self.collisions = [0] * self.num_envs

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]

        for env_idx in range(self.num_envs):
            self.current_rewards[env_idx] += rewards[env_idx]
            self.steps_in_episode[env_idx] += 1

            if rewards[env_idx] <= -1.0:
                self.collisions[env_idx] += 1

            if dones[env_idx]:
                efficiency = self.current_rewards[env_idx] / self.steps_in_episode[env_idx] if self.steps_in_episode[env_idx] > 0 else 0
                print(f"\nEpisódio finalizado no ambiente {env_idx}:")
                print(f"Pontuação: {self.current_rewards[env_idx]}")
                print(f"Passos: {self.steps_in_episode[env_idx]}")
                print(f"Colisões: {self.collisions[env_idx]}")
                print(f"Eficiência: {efficiency:.3f}")

                self.episode_rewards.append(self.current_rewards[env_idx])
                self.current_rewards[env_idx] = 0
                self.steps_in_episode[env_idx] = 0
                self.collisions[env_idx] = 0

        return True

class CollisionAwareWrapper(Wrapper):
    def __init__(self, env, lookahead_steps=3, collision_threshold=0.2):
        super().__init__(env)
        self.lookahead_steps = lookahead_steps
        self.collision_threshold = collision_threshold
        self.vehicle_color = np.array([84, 92, 214])  # Cor dos veículos em RGB
        self.last_action = None
        
    def _detect_vehicles(self, obs):
        # Cria máscara para pixels de veículos
        vehicle_mask = np.all(obs == self.vehicle_color, axis=-1)
        return vehicle_mask
    
    def _get_chicken_y(self, obs):
        white_pixels = np.where(obs > 200)
        if len(white_pixels[0]) > 0:
            return np.mean(white_pixels[0])
        return None

    def _predict_collision(self, current_y, obs):
        # Analisa a área à frente da galinha
        height, width = obs.shape[:2]
        safety_margin = int(height * 0.1)
        
        # Define área de perigo (faixa à frente)
        danger_zone = obs[
            max(0, int(current_y) - safety_margin) : int(current_y),
            int(width*0.3) : int(width*0.7)
        ]
        
        # Verifica presença de veículos na área de perigo
        vehicle_pixels = np.sum(self._detect_vehicles(danger_zone))
        total_pixels = danger_zone.shape[0] * danger_zone.shape[1]
        
        return (vehicle_pixels / total_pixels) > self.collision_threshold
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_y = self._get_chicken_y(obs)
        
        if current_y is not None:
            # Detecção preventiva de colisão
            collision_risk = self._predict_collision(current_y, obs)
            
            if collision_risk:
                if action == 2:  # Recompensa por recuar quando há risco
                    reward += 1.0
                elif action == 1:  # Penalidade por avançar com risco
                    reward -= 2.0
            else:
                if action == 1:  # Recompensa por avançar quando seguro
                    reward += 2.0
                elif action == 2:  # Penalidade alta por recuar sem necessidade
                    reward -= 5.0
                elif action == 0:  # Pequena penalidade por ficar parado sem necessidade
                    reward -= 0.5
        
        self.last_action = action
        return obs, reward, terminated, truncated, info
    
from collections import deque

class TemporalMemoryWrapper(Wrapper):
    def __init__(self, env, stack_size=4):
        super().__init__(env)
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.stack_size):
            self.frames.append(obs)
        return self._get_stacked_obs(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_stacked_obs(), reward, terminated, truncated, info
    
    def _get_stacked_obs(self):
        return np.array(self.frames).transpose(1, 2, 0, 3).squeeze()

class ResourceMonitor(BaseCallback):
    def __init__(self, verbose=0, print_freq=1000):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.process = psutil.Process()
    
    def _on_step(self) -> bool:
        if self.n_calls % self.print_freq == 0:
            # CPU Usage
            cpu_percent = psutil.cpu_percent(interval=None)
            process_cpu = self.process.cpu_percent()
            
            # RAM Usage
            ram_usage = psutil.virtual_memory().percent
            process_ram = self.process.memory_percent()
            
            print("\n=== Resource Usage ===")
            print(f"CPU Total: {cpu_percent:.1f}%")
            print(f"CPU Process: {process_cpu:.1f}%")
            print(f"RAM Total: {ram_usage:.1f}%")
            print(f"RAM Process: {process_ram:.1f}%")
            
            # GPU Usage (if available)
            if torch.cuda.is_available():
                gpu = torch.cuda.get_device_properties(0)
                gpu_util = torch.cuda.utilization()
                gpu_mem = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
                print(f"GPU {gpu.name}")
                print(f"GPU Utilization: {gpu_util}%")
                print(f"GPU Memory: {gpu_mem:.1f}MB")
            elif torch.backends.mps.is_available():
                print("MPS (Apple Silicon GPU) in use")
                # Note: Apple Silicon GPU metrics not directly available
            
            print("==================\n")
        return True

def make_env():
    env = gym.make(
        "ALE/Freeway-v5",
        render_mode="rgb_array",
        obs_type="rgb",
        frameskip=1,
        full_action_space=False,
        repeat_action_probability=0.2
    )
    
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=FRAME_SKIP)
    env = WarpFrame(env, width=84, height=84)       # 1. Redimensionamento
    env = CollisionAwareWrapper(env)        # 3. Detecção de colisão
    env = CrossingRewardWrapper(env)        # 4. Recompensas
    env = EpisodicLifeEnv(env)      # 5. Controle de episódios
    
    return env

# Criação do ambiente vetorizado
env = DummyVecEnv([lambda: make_env() for _ in range(NUM_ENVS)])
env = VecFrameStack(env, n_stack=3, channels_order='last')  # Stacking adicional

new_policy_kwargs = {
    'net_arch': [512, 256],
    'normalize_images': True,

}

# Configurações de treino atualizadas
model = DQN(
    policy=CnnPolicy,
    env=env,
    learning_rate=0.0003,
    buffer_size=75_000,
    learning_starts=30_000,
    batch_size=128,
    gamma=0.99,  # Gamma maior para dar mais importância às recompensas futuras
    target_update_interval=2500,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.01,
    exploration_fraction=0.3,
    train_freq=4,
    gradient_steps=1,
    policy_kwargs=new_policy_kwargs,
    verbose=1,
    seed=SEED,
    device=device
)

# modelo_path = "model_path"

# if os.path.exists(modelo_path):
#     model.load(modelo_path)

# Treinamento com monitoramento
start_time = time.time()
score_callback = SmartScoreCallback()
resource_callback = ResourceMonitor()

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        log_interval=50,
        progress_bar=True,
        callback=[score_callback, resource_callback]
    )
except KeyboardInterrupt:
    print("Treinamento interrompido")

print(f"\nTreinamento concluído em {time.time() - start_time:.1f}s")
if score_callback.episode_rewards:
    print(f"Média de pontuação: {sum(score_callback.episode_rewards) / len(score_callback.episode_rewards):.2f}")
    print(f"Melhor pontuação: {max(score_callback.episode_rewards):.2f}")

model.save("freeway_dqn_crx_vec.zip")