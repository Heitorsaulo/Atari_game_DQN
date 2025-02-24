# __Atari Game DQN__

Este projeto tem como objetivo treinar um agente de aprendizado por reforço para jogar o jogo de Atari "Freeway" utilizando a técnica Deep Q-Network (DQN). Nesse projeto foram testados diversos abordagens, sendo elas: CNN com recompensão padrão, CNN com recompensa personalizada, MLP com recompensa padrão e MLP com recompensa personalizada.

##  __Integrantes do Grupo__

- **Heitor Saulo Dantas Santos**
- **Itor Carlos Souza Queiroz**
- **Lanna Luara Novaes Silva**
- **Lavínia Louise Rosa Santos**
- **Rômulo Menezes De Santana**



##  __Política CNN - Recompensa Padrão__

O treinamento do agente no jogo Freeway utilizando a política CNN com recompensa padrão está disponível no arquivo `train_cnn.py` na pasta `cnn-recompensa-padrao`.

### Como funciona?

- Na primeira execução, um modelo treinado será salvo no arquivo `freeway_dqn_cnn_model.zip` na mesma pasta de execução do arquivo.
- Nas execuções seguintes, o treinamento será continuado a partir do modelo salvo na etapa anterior.

### Instalação e Configuração

1. Clone o repositório e instale as dependências necessárias:

    ```sh
    pip install gymnasium stable-baselines3 ale-py
    pip install gymnasium[atari]
    ```

2. Execute o treinamento do modelo:

    ```sh
    python cnn-recompensa-padrao/train_cnn.py
    ```



##  __Política CNN - Recompensa Personalizada__

O treinamento do agente no jogo Freeway utilizando a política CNN com recompensa personalizada está disponível no arquivo `train_cnnR.py` na pasta `cnn-recompensa-personalizada`.

### Como funciona?

- Na primeira execução, um modelo treinado será salvo no arquivo `freeway_dqn_cnn_personalizada_model.zip` na mesma pasta de execução do arquivo.
- Nas execuções seguintes, o treinamento será continuado a partir do modelo salvo na etapa anterior.

### Instalação e Configuração

1. Clone o repositório e instale as dependências necessárias:

    ```sh
    pip install gymnasium[atari] stable-baselines3 ale-py autorom
    pip install numpy matplotlib torch
    ```

2. Execute o treinamento do modelo:

    ```sh
    python cnn-recompensa-personalizada/train_cnnR.py
    ```


##  __Política MLP - Recompensa Padrão__

O treinamento do agente no jogo Freeway utilizando a política MLP com recompensa padrão está disponível no arquivo `train_mlp.py` na pasta `mlp-recompensa-padrao`.

### Como funciona?

- Na primeira execução, um modelo treinado será salvo no arquivo `freeway_dqn_mlp_model.zip` na mesma pasta de execução do arquivo.
- Nas execuções seguintes, o treinamento será continuado a partir do modelo salvo na etapa anterior.

### Instalação e Configuração

1. Clone o repositório e instale as dependências necessárias:

    ```sh
    pip install gymnasium stable-baselines3 ale-py
    pip install gymnasium[atari]
    ```

2. Execute o treinamento do modelo:

    ```sh
    python mlp-recompensa-padrao/train_mlp.py
    ```


##  __Política MLP - Recompensa Personalizada__

O treinamento do agente no jogo Freeway utilizando a política MLP com recompensa personalizada está disponível no arquivo `train_mlpR.py` na pasta `mlp-recompensa-personalizada`.

### Como funciona?

- Na primeira execução, um modelo treinado será salvo no arquivo `freeway_dqn_mlp_personalizada_model.zip` na mesma pasta de execução do arquivo.
- Nas execuções seguintes, o treinamento será continuado a partir do modelo salvo na etapa anterior.

### Instalação e Configuração

1. Clone o repositório e instale as dependências necessárias:

    ```sh
    pip install gymnasium ale-py matplotlib numpy stable-baselines3 torch stable-baselines3[extra]
    ```

2. Execute o treinamento do modelo:

    ```sh
    python mlp-recompensa-personalizada/train_mlpR.py
    ```

