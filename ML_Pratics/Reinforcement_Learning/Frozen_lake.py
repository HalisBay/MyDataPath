import gym
import numpy as np
from tqdm import tqdm
import time


def create_env(render_mode=None):
    return gym.make(
        "FrozenLake-v1", map_name="8x8", is_slippery=False, render_mode=render_mode
    )


def train_q_learning_with_visualization(env, episodes=1000, alpha=0.5, gamma=0.9):

    nb_states = env.observation_space.n # Ortamda kaç farklı state var n x n

    """
    0: Move left , 1: Move down
    2: Move right, 3: Move up
    """
    nb_actions = env.action_space.n # aksiyonları alır = 0,1,2,3
    qtable = np.zeros((nb_states, nb_actions))
    for ep in range(episodes):
        state, _ = env.reset() 
        done = False
        print(f"Episode {ep+1}/{episodes}")
        while not done:
            env.render()
            if np.max(qtable[state]) > 0:
                action = np.argmax(qtable[state])
            else:
                action = env.action_space.sample()
            new_state, reward, done, info, _ = env.step(action)
            # Q(s, a) ← Q(s, a) + α * (r + γ * max(Q(s’, a’)) − Q(s, a))
            #Yeni Q-değeri = Eski Q-değeri + öğrenme oranı × (anlık ödül + discount Rate × (gelecekteki en iyi Q) - Eski Q-değeri)
            qtable[state, action] += alpha * (
                reward + gamma * np.max(qtable[new_state]) - qtable[state, action]
            )
            state = new_state
        # Episode bitince ekranı temizle
        env.render()
    return qtable


def train_q_learning(env, episodes=1000, alpha=0.5, gamma=0.9):
    nb_states = env.observation_space.n
    nb_actions = env.action_space.n
    qtable = np.zeros((nb_states, nb_actions))
    for _ in tqdm(range(episodes)):
        state, _ = env.reset()
        done = False
        while not done:
            if np.max(qtable[state]) > 0:
                action = np.argmax(qtable[state])
            else:
                action = env.action_space.sample()
            new_state, reward, done, info, _ = env.step(action)
            qtable[state, action] += alpha * (
                reward + gamma * np.max(qtable[new_state]) - qtable[state, action]
            )
            state = new_state
    return qtable


def test_agent(env, qtable, test_episodes=3):
    for ep in range(test_episodes):
        state, _ = env.reset()
        done = False
        print(f"\nTest Episode {ep+1}")
        while not done:
            env.render()
            action = np.argmax(qtable[state])
            new_state, reward, done, info, _ = env.step(action)
            state = new_state
        print("Success!" if reward else "Failure!")


def main():
    visualize = False  # True ile eğitim görsel, False ile eğitim tqdm ile izlenir

    if visualize:
        env = create_env(render_mode="human")
        qtable = train_q_learning_with_visualization(env)
        env.close()
    else:
        env = create_env()
        qtable = train_q_learning(env)
        env.close()

    print("Qtable After Training:")
    print(qtable)

    # Test (görsel)
    test_env = create_env(render_mode="human")
    test_agent(test_env, qtable)
    test_env.close()


if __name__ == "__main__":
    main()
