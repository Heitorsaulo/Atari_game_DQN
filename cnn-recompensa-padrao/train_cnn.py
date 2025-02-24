import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.dqn import CnnPolicy
import time
import os
import ale_py

env = gym.make("ALE/Freeway-v5", render_mode="human")

modelo_path = "cnn-recompensa-padrao/freeway_dqn_cnn_model.zip"

if os.path.exists(modelo_path):
    modelo = DQN.load("cnn-recompensa-padrao/freeway_dqn_cnn_model.zip", env=env)
else:
    print("Criando novo modelo e arquivo...")
    modelo = DQN(
        policy=CnnPolicy,
        env=env,
        learning_rate=0.0005,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        target_update_interval=1000,
        train_freq=4,
        verbose=1
    )
    modelo.save("cnn-recompensa-padrao/freeway_dqn_cnn_model.zip")

inicio = time.time()
modelo.learn(total_timesteps=600000, log_interval=10)
fim = time.time()

tempo_decorrido = int(fim - inicio)
horas = tempo_decorrido // 3600
minutos = (tempo_decorrido % 3600) // 60
segundos = tempo_decorrido % 60