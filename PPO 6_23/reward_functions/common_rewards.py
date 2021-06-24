from abc import abstractmethod

import numpy as np
import csv
from rlgym.utils import math
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_GOAL_CENTER, BLUE_GOAL_CENTER, ORANGE_TEAM
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.reward_functions import RewardFunction
from stable_baselines3.common import logger

class EventReward(RewardFunction):
    def __init__(self, goal=0., team_goal=0., concede=-0., touch=0., shot=0., save=0., demo=0.):
        """
        :param goal: reward for goal scored by player.
        :param team_goal: reward for goal scored by player's team.
        :param concede: reward for goal scored by opponents. Should be negative if used as punishment.
        :param touch: reward for touching the ball.
        :param shot: reward for shooting the ball (as detected by Rocket League).
        :param save: reward for saving the ball (as detected by Rocket League).
        :param demo: reward for demolishing a player.
        """
        super().__init__()
        self.weights = np.array([goal, team_goal, concede, touch, shot, save, demo])

        # Need to keep track of last registered value to detect changes
        self.last_registered_values = {}

    @staticmethod
    def _extract_values(player: PlayerData, state: GameState):
        if player.team_num == BLUE_TEAM:
            team, opponent = state.blue_score, state.orange_score
        else:
            team, opponent = state.orange_score, state.blue_score

        return np.array([player.match_goals, team, opponent, player.ball_touched, player.match_shots,
                         player.match_saves, player.match_demolishes])

    def reset(self, initial_state: GameState, optional_data=None):
        # Update every reset since rocket league may crash and be restarted with clean values
        self.last_registered_values = {}
        for player in initial_state.players:
            self.last_registered_values[player.car_id] = self._extract_values(player, initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray, optional_data=None):
        old_values = self.last_registered_values[player.car_id]
        new_values = self._extract_values(player, state)

        diff_values = new_values - old_values
        diff_values[diff_values < 0] = 0  # We only care about increasing values

        reward = np.dot(self.weights, diff_values)

        self.last_registered_values[player.car_id] = new_values
        return reward
    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0
class DistanceBallToGoalReward(RewardFunction):
    def __init__(self, own_goal=False):
        super().__init__()
        self.own_goal = own_goal

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM and not self.own_goal \
                or player.team_num == ORANGE_TEAM and self.own_goal:
            objective = np.array(ORANGE_GOAL_CENTER)
        else:
            objective = np.array(BLUE_GOAL_CENTER)
        objective[1] *= 6000 / 5120  # Use back of net

        dist = np.linalg.norm(state.ball.position - objective) - 786  # Compensate for moving objective to back of net
        return np.exp(-0.5 * dist / 100)  # From https://arxiv.org/abs/2105.12196
    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0

class DistancePlayerToBallReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        dist = np.linalg.norm(player.car_data.position - state.ball.position) - 94  # Compensate for ball radius
        return np.exp(-0.5 * dist / 100)  # From https://arxiv.org/abs/2105.12196

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0
class VelocityPlayerToBallReward(RewardFunction):
    def __init__(self, use_scalar_projection=True):
        super().__init__()
        self.use_scalar_projection = use_scalar_projection

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        vel = player.car_data.linear_velocity
        pos_diff = state.ball.position - player.car_data.position
        if self.use_scalar_projection:
            # Vector version of v=d/t <=> t=d/v <=> 1/t=v/d
            # Max value should be max_speed / ball_radius = 2300 / 94 = 24.5
            # Used to guide the agent towards the ball
            inv_t = math.scalar_projection(vel, pos_diff)
            return inv_t
        else:
            # Regular component velocity
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            vel /= 100  # uu = cm -> m
            return float(np.dot(norm_pos_diff, vel))
    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0

class VelocityBallToGoalReward(RewardFunction):
    def __init__(self, own_goal=False, use_scalar_projection=True):
        super().__init__()
        self.own_goal = own_goal
        self.use_scalar_projection = use_scalar_projection

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM and not self.own_goal \
                or player.team_num == ORANGE_TEAM and self.own_goal:
            objective = np.array(ORANGE_GOAL_CENTER)
        else:
            objective = np.array(BLUE_GOAL_CENTER)
        objective[1] *= 6000 / 5120  # Use back of net instead to prevent exploding reward
        #multiplier for balls height during shot over the height of the goal)
        ball_height = float((1 + ((state.ball.position[2] - 92.75) / 550.025)))
        vel = state.ball.linear_velocity
        pos_diff = objective - state.ball.position
        if self.use_scalar_projection:
            # Vector version of v=d/t <=> t=d/v <=> 1/t=v/d
            # Max value should be max_speed / ball_radius = 2300 / 94 = 24.5
            # Used to guide the agent towards the ball
            inv_t = math.scalar_projection(vel, pos_diff)
            return inv_t * ball_height
        else:
            # Regular component velocity
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            vel /= 100  # uu/s = cm/s -> m/s
            return float(np.dot(norm_pos_diff, vel)*ball_height)
    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0
class GoalLineAgentBall(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:

        if player.team_num == BLUE_TEAM:
            objective = np.array(BLUE_GOAL_CENTER)
            bal_pos = state.ball.position
            car_pos = player.car_data.position
            car_vel = np.linalg.norm(player.car_data.linear_velocity)
        else:
            objective = np.array(ORANGE_GOAL_CENTER)
            bal_pos = state.inverted_ball.position
            car_pos = player.inverted_car_data.position
            car_vel = np.linalg.norm(player.inverted_car_data.linear_velocity)

        # Use back of net BUT not all the way back because if the ball is back there it's TOO LATE
        objective[1] *= 5560 / 5120
        car_dist_from_goal = np.linalg.norm(car_pos - objective)
        ball_dist_from_goal = np.linalg.norm(bal_pos - objective)

        if car_dist_from_goal > ball_dist_from_goal and car_vel < 1415:
            return -5
        elif car_dist_from_goal > ball_dist_from_goal and car_vel >= 1415 and car_vel < 2200:
            return -2
        elif car_dist_from_goal > ball_dist_from_goal and car_vel >= 2200:
            return -0.5
        else:
            return 1

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0


class VelocityReward(RewardFunction):
    # Simple reward function to ensure the model is training.
    def __init__(self, negative=False):
        super().__init__()
        self.negative = negative

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:

        return np.linalg.norm(player.car_data.linear_velocity) / 100 * (1 - 2 * self.negative)

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0

class SaveBoostReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # 1 reward for each frame with 100 boost, sqrt because 0->20 makes bigger difference than 80->100
        return np.sqrt(player.boost_amount)
    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0

class ConstantReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 1
    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0

class BallYCoordinateReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM:
            return (state.ball.position[1] / (5120 + 94)) ** 3
        else:
            return (state.inverted_ball.position[1] / (5120 + 94)) ** 3
    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0

class FaceBallReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        pos_diff = state.ball.position - player.car_data.position
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)

        return float(np.dot(player.car_data.forward(), norm_pos_diff))

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0

class ConditionalRewardFunction(RewardFunction):
    def __init__(self, reward_func: RewardFunction):
        super().__init__()
        self.reward_func = reward_func

    @abstractmethod
    def condition(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> bool:
        raise NotImplementedError

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if self.condition(player, state, previous_action):
            return self.reward_func.get_reward(player, state, previous_action)
        return 0

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if self.condition(player, state, previous_action):
            return self.reward_func.get_final_reward(player, state, previous_action)
        return 0


class RewardIfClosestToBall(ConditionalRewardFunction):
    def __init__(self, reward_func: RewardFunction, team_only=True):
        super().__init__(reward_func)
        self.team_only = team_only

    def condition(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> bool:
        dist = np.linalg.norm(player.car_data.position - state.ball.position)
        for player2 in state.players:
            if not self.team_only or player2.team_num == player.team_num:
                dist2 = np.linalg.norm(player2.car_data.position - state.ball.position)
                if dist2 < dist:
                    return False
        return True


class RewardIfTouchedLast(ConditionalRewardFunction):
    def condition(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> bool:
        return state.last_touch == player.car_id

class TouchBallReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if state.last_touch == player.car_id:
            por=10
        else:
            por=-1
        get_rewards = [player.car_id, "touch_ball_reward", por]
        with open('C:\\Users\\Daanesh\\PycharmProjects\\RocketLeagueRL\\Reward_Details\\Rewards.csv', 'a', newline='') as csvfile:
            my_writer = csv.writer(csvfile, delimiter=',')
            my_writer.writerow(get_rewards)
        return 10 if player.ball_touched else -1

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0

class RewardIfBehindBall(ConditionalRewardFunction):
    def condition(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> bool:
        return player.team_num == BLUE_TEAM and player.car_data.position[1] < state.ball.position[1] \
               or player.team_num == ORANGE_TEAM and player.inverted_car_data.position[1] < state.inverted_ball.position[1]


class MoveTowardsBallReward(RewardFunction):
    def reset(self, initial_state: GameState, optional_data=None):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray, optional_data=None):
        inv_t = math.scalar_projection(player.car_data.linear_velocity, state.ball.position - player.car_data.position)
        get_rewards = [player.car_id, "move_towards_ball_reward", (inv_t * .01)]
        with open('C:\\Users\\Daanesh\\PycharmProjects\\RocketLeagueRL\\Reward_Details\\Rewards.csv', 'a', newline='') as csvfile:
                my_writer = csv.writer(csvfile, delimiter=',')
                my_writer.writerow(get_rewards)
        return inv_t
    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0

class GoalReward(RewardFunction):
    def __init__(self, per_goal: float = 1., team_score_coeff: float = 0., concede_coeff: float = 0.):
        super().__init__()
        self.per_goal = per_goal
        self.team_score_coeff = team_score_coeff
        self.concede_coeff = concede_coeff

        # Need to keep track of last registered value to detect changes
        self.goals_scored = {}
        self.blue_goals = 0
        self.orange_goals = 0

    def reset(self, initial_state: GameState, optional_data=None):
        # Update every reset since rocket league may crash and be restarted with clean values
        self.blue_goals = initial_state.blue_score
        self.orange_goals = initial_state.orange_score
        for player in initial_state.players:
            self.goals_scored[player.car_id] = player.match_goals

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray, optional_data=None):
        self.goals_scored[player.car_id], d_player = player.match_goals, player.match_goals - self.goals_scored[player.car_id]
        self.blue_goals, d_blue = state.blue_score, state.blue_score - self.blue_goals
        self.orange_goals, d_orange = state.blue_score, state.blue_score - self.orange_goals

        if player.team_num == BLUE_TEAM:
            get_rewards = [player.car_id, "goal_reward",
                          (self.per_goal * d_player + self.team_score_coeff * d_blue - self.concede_coeff * d_orange) * 100]
            with open('C:\\Users\\Daanesh\\PycharmProjects\\RocketLeagueRL\\Reward_Details\\Rewards.csv', 'a',
                      newline='') as csvfile:
                my_writer = csv.writer(csvfile, delimiter=',')
                my_writer.writerow(get_rewards)
            return self.per_goal * d_player + self.team_score_coeff * d_blue - self.concede_coeff * d_orange
        else:
            get_rewards = [player.car_id, "goal_reward",
                          (self.per_goal * d_player + self.team_score_coeff * d_orange - self.concede_coeff * d_blue) * 100]
            with open('C:\\Users\\Daanesh\\PycharmProjects\\RocketLeagueRL\\Reward_Details\\Rewards.csv', 'a',
                      newline='') as csvfile:
                my_writer = csv.writer(csvfile, delimiter=',')
                my_writer.writerow(get_rewards)
            return self.per_goal * d_player + self.team_score_coeff * d_orange - self.concede_coeff * d_blue

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray, optional_data=None):
        return 0