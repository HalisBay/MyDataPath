import gym
import numpy as np
from tqdm import tqdm
import time


def create_env(render_mode=None):
    return gym.make("Taxi-v3", render_mode=render_mode)


def train_q_learning(
    env,
    episodes=100,
    alpha=0.7,
    gamma=0.95,
    epsilon=1.0,
    epsilon_min=0.1,
    epsilon_decay=0.995,
):
    nb_states = env.observation_space.n
    nb_actions = env.action_space.n
    qtable = np.zeros((nb_states, nb_actions))

    for ep in tqdm(range(episodes)):
        state, _ = env.reset()
        done = False
        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(qtable[state])

            new_state, reward, done, truncated, _ = env.step(action)
            qtable[state, action] += alpha * (
                reward + gamma * np.max(qtable[new_state]) - qtable[state, action]
            )
            state = new_state

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

    return qtable


def test_agent_with_visualize(env, qtable, test_episodes=3):
    for ep in range(test_episodes):
        state, _ = env.reset()
        done = False
        print(f"\nTest Episode {ep+1}")
        total_reward = 0
        while not done:
            env.render()
            action = np.argmax(qtable[state])
            new_state, reward, done, truncated, _ = env.step(action)
            state = new_state
            total_reward += reward
        print(f"Episode Reward: {total_reward}")


def test_agent_with_tqdm(env, qtable, episodes=100):
    total_epoch, total_penalties = 0, 0

    for i in tqdm(range(episodes)):
        state, _ = env.reset()
        epochs, penalties, reward = 0, 0, 0
        done = False
        while not done:
            action = np.argmax(qtable[state])
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state
            if reward == -10:
                penalties += 1
            epochs += 1
        total_epoch += epochs
        total_penalties += penalties

    print("Result after {} episodes".format(episodes))
    print("Average timesteps per episode: ", total_epoch / episodes)
    print("Average penalties per episode: ", total_penalties / episodes)


def main(visualize=False):
    env = create_env()
    qtable = train_q_learning(env, episodes=1000)
    env.close()

    print("Q-table After Training:")
    print(qtable)

    if visualize:
        test_env = create_env(render_mode="human")
        test_agent_with_visualize(test_env, qtable)
        test_env.close()
    else:
        test_env = create_env()
        test_agent_with_tqdm(test_env, qtable)
        test_env.close()


if __name__ == "__main__":
    main(visualize=False)
