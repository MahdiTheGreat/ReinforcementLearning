# epsilon-greedy example implementation of a multi-armed bandit
import random

class Bandit:

    def __init__(self, arms, epsilon=0.7):
        """
        Initiates the bandits

        :param arms: List of arms
        :param epsilon: Epsilon value for random exploration
        """
        self.arms = arms
        self.epsilon = epsilon
        self.frequencies = [0] * len(arms)
        self.sums = [0] * len(arms)
        self.expected_values = [0] * len(arms)
        self.epsilon_decay=0.9


    def run(self):
        """
        Asks the bandit to recommend the next arm

        :return: Returns the arm the bandit recommends pulling
        """
        if min(self.frequencies) == 0:
            return self.arms[self.frequencies.index(min(self.frequencies))]
        if random.random() < self.epsilon:
            self.epsilon *= self.epsilon_decay
            return self.arms[random.randint(0, len(self.arms) - 1)]
        return self.arms[self.expected_values.index(max(self.expected_values))]

    def give_feedback(self, arm, reward):
        """
        Sets the bandit's reward for the most recent arm pull

        :param arm: The arm that was pulled to generate the reward
        :param reward: The reward that was generated
        """
        arm_index = self.arms.index(arm)
        sum = self.sums[arm_index] + reward
        self.sums[arm_index] = sum
		# frequency is the number of arm pulls
        frequency = self.frequencies[arm_index] + 1
        self.frequencies[arm_index] = frequency
        # using frequency to normalize expected value 
        expected_value = sum / frequency
        self.expected_values[arm_index] = expected_value