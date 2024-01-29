import random
from collections import defaultdict
import numpy as np

def reshape_obs(observation):
    """
    Reshapes and 'discretizes' an observation for Q-table read/write
    Make sure the state space is not too large!

    :param observation: The to-be-reshaped/discretized observation. Contains the position of the
    'players', as well as the position and movement.
    direction of the ball.
    :return: The reshaped/discretized observation
    """
    # TODO: Discretize/simplify
    # transformation
    # return f'{np.asarray(observation).reshape(-1, 10)}'
    pass
    temp=f'{np.rint(np.asarray(observation).reshape(-1, 10))}'
    pass
    return temp


class Agent:


    def __init__(
            self, id, actions_n, obs_space_shape,
            gamma=1, # pick reasonable values for all of these!
            epsilon=0.7,
            min_epsilon=0.01,
            epsilon_decay=0.9,
            alpha=0.1
    ):
        """
        Initiates the agent
        :param id: The agent's id in the game environment
        :param actions_n: The id of actions in the agent's action space
        :param obs_space_shape: The shape of the agents observation space
        :param gamma: Depreciation factor for expected future rewards
        :param epsilon: The initial/current exploration rate
        :param min_epsilon: The minimal/final exploration rate
        :param epsilon_decay: The rate of epsilon/exploration decay
        :param alpha: The learning rate
        """
        self.id = id
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.actions_n = actions_n
        self.obs_space_shape = obs_space_shape
        self.alpha = alpha
        self.q = defaultdict(lambda: np.zeros(self.actions_n))

    def determine_action_probabilities(self, observation):
        """
        A function that takes the state as an input and returns the probabilities for each
        action in the form of a np array of length of the action space.
        :param observation: The agent's current observation
        :return: The probabilities for each action in the form of a np
        array of length of the action space.
        """
        action_probabilities=[]
        best_action = np.argmax(self.q[reshape_obs(observation)])
        for action in range(self.actions_n):
            if action==best_action:
                action_probabilities.append(1-self.epsilon)
            else:
                action_probabilities.append(self.epsilon/(self.actions_n-1))

        return np.asarray(action_probabilities)

    def act(self, observation):
        """
        Determines and action, given the current observation.
        :param observation: the agent's current observation of the state of
        the world
        :return: the agent's action
        """
        action_probabilities=self.determine_action_probabilities(observation)
        selected_action=random.choices(population=[i for i in range(self.actions_n)],weights=action_probabilities,k=1)[0]
        temp=self.epsilon_decay*self.epsilon
        if temp>self.min_epsilon:
            self.epsilon=temp
        return selected_action

    def update_history(
            self, observation, action, reward, new_observation
    ):
        """
        Updates the agent's Q-table

        :param observation: The observation *before* the action
        :param action: The action that has been executed
        :param reward: The reward the action has yielded
        :param new_observation: The observation *after* the action
        :return:
        """
        # counterfactual next action, to later backpropagate reward to current action
        next_action = np.argmax(self.q[reshape_obs(new_observation)])
        td_target = reward + self.gamma * self.q[reshape_obs(new_observation)][next_action]
        td_delta = td_target - self.q[reshape_obs(observation)][action]
        self.q[reshape_obs(observation)][action] += self.alpha * td_delta

