import argparse

import gym
from ma_gym.wrappers import Monitor
import matplotlib.pyplot as plt

from Agent import Agent
from RandomAgent import RandomAgent
import pickle

"""
Based on:
https://github.com/koulanurag/ma-gym/blob/master/examples/random_agent.py

This script executes the Pong simulator with two agents, both of which make
use of the ``Agent`` class. Nothing in this file needs to be changed, but
you can make changes for debugging purposes.
"""

def pong_agent_trainer(my_agent,env,episodes,render_env=True):

    agents = [
        my_agent,
        RandomAgent(1)
    ]

    print(f'Action space: {env.action_space}')
    print(f'Observation (state) space: {env.observation_space}')

    wins = []
    losses = []
    win_loss_history = []
    # Run for a number of episodes
    for ep_i in range(episodes):
        are_done = [False for _ in range(env.n_agents)]
        ep_rewards = [0, 0]

        env.seed(ep_i)
        prev_observations = env.reset()
        if render_env:env.render()

        while not all(are_done):
            # Observe:
            prev_observations = env.get_agent_obs()
            actions = []
            # For each agent, act:
            for (index, observation) in enumerate(prev_observations):
                action = agents[index].act(prev_observations)
                actions.append(action)
                # Use the command below to print the exact actions that the
                # agents are executing:
                # print([action_meanings[index][action] for action in actions])
            # Trigger the actual execution
            observations, rewards, are_done, infos = env.step(actions)
            # For each agent, update observations and rewards
            for agent in agents:
                agent.update_history(
                    prev_observations,
                    actions[agent.id],
                    rewards[agent.id],
                    observations
                )
            # Debug: print obs, rewards, info
            # print(observations, rewards, infos)
            # Rewards are either 0 or [1, -1], or [-1, 1], try out by out-commenting the line below
            # Note that your agent is agent 0
            # if not (rewards[0] == 0 and rewards[1] == 0): print(rewards)
            for (index, reward) in enumerate(rewards):
                ep_rewards[index] += reward
            if render_env: env.render()
        # Aggregate wins and losses
        bottom_line = ep_rewards[0]
        if bottom_line < 0:
            wins.append(0)
            losses.append(abs(bottom_line))
        else:
            wins.append(bottom_line)
            losses.append(0)
        win_loss_history.append(sum(wins) - sum(losses))
        print('Episode #{} Rewards: {}'.format(ep_i, ep_rewards))
        print(f'Wins - losses: {sum(wins) - sum(losses)}')
        print(f'Epsilon: {my_agent.epsilon}')
        print(f'Q table size: {len(my_agent.q)}')
        if len(wins) > 10:
            print(f'Last 10 games: {sum(wins[-10:]) - sum(losses[-10:])}')
    return win_loss_history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pong simulator for ma-gym')
    parser.add_argument('--env', default='PongDuel-v0',
                        help='Name of the environment (default: %(default)s)')
    parser.add_argument('--episodes', type=int, default=1,
                        help='episodes (default: %(default)s)')
    parser.add_argument('--qTable', type=str, default=None,
                        help='qtable (default: %(default)s)')
    parser.add_argument('-r', '--render', action='store_true', help='render the environment')

    # Parse the arguments
    args = parser.parse_args()

    # Set up environment
    env = gym.make(args.env)
    video_callable=lambda ep: ep==(args.episodes-1)
    env = Monitor(env, directory='recordings/' + args.env,video_callable=video_callable,force=True)
    action_meanings = env.get_action_meanings()
    # Initialize agents
    my_agent = Agent(0, env.action_space[0].n, env.observation_space[0].shape)
    qTable=my_agent.q
    if args.qTable:
     try:
      with open(args.qTable+'.pickle', 'rb') as f:
          qTable = pickle.load(f)
     except FileNotFoundError:
         print("qTable file doesn't exit, therefore it's going to be created")
    my_agent.q=qTable
    win_loss_history=pong_agent_trainer(my_agent=my_agent,env=env,episodes=args.episodes,render_env=args.render)

    if args.episodes>1:
     with open(args.qTable+'.pickle', 'wb') as f:
         pickle.dump(dict(my_agent.q), f)
     plt.clf()
     plt.cla()
     plt.close()
     plt.plot(win_loss_history)
     plt.savefig('win_loss_history_plot.png')
     plt.pause(0.1)
     env.close()
