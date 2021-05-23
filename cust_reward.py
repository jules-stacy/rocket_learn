from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_GOAL_CENTER, BLUE_GOAL_CENTER
from rlgym.utils import math
import numpy as np



class SpeedReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        linear_velocity = player.car_data.linear_velocity
        reward = math.vec_mag(linear_velocity)
        
        return reward
        
    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0

class TimePunishment(RewardFunction):
    def __init__(self, timesteps):
        super().__init__()
        self.timesteps = timesteps

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        punishment = (-0.01 * self.timesteps)
        return punishment

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0


class GoalReward(RewardFunction):
    def __init__(self, per_goal: float = 1.):
        super().__init__()
        self.blue_score = 0
        self.orange_score = 0
        

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray, optional_data=None):
        if state.blue_score != self.blue_score or state.orange_score != self.orange_score:
            print("blue score: ", self.blue_score, "|", state.blue_score, "orange score: ", self.orange_score, "|", state.orange_score)
            self.blue_score = state.blue_score
            self.orange_score = state.orange_score
            return 100
        else:
            return 0

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray, optional_data=None):
        return 0