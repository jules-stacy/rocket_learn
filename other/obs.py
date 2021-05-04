from typing import Any, List
import math
import numpy as np

from rlgym.utils import ObsBuilder, common_values
from rlgym.utils.gamestates import PlayerData, GameState, PhysicsObject


class CustomObs(ObsBuilder):
    POS_STD = 2300
    ANG_STD = math.pi

    def __init__(self, cars):
        super().__init__()
        self.obs_size = 9 + 8 + 25 + 31 * (cars - 1)

    def reset(self, initial_state, optional_data=None):
        pass

    def build_obs(self, state: GameState, optional_data=None) -> np.ndarray:
        pass

    def build_obs_for_player(self, player: PlayerData, state: GameState, previous_action: np.ndarray,
                             optional_data=None) -> Any:

        if player.team_num == common_values.ORANGE_TEAM:
            inverted = True
            ball = state.inv_ball
        else:
            inverted = False
            ball = state.ball

        obs = [ball.position / CustomObs.POS_STD,
               ball.linear_velocity / CustomObs.POS_STD,
               ball.angular_velocity / CustomObs.ANG_STD,
               previous_action]

        player_car = self._add_player_to_obs(obs, player, ball, inverted)

        allies = []
        enemies = []

        for other in state.players:
            if other.car_id == player.car_id:
                continue

            if other.team_num == player.team_num:
                team_obs = allies
            else:
                team_obs = enemies

            other_car = self._add_player_to_obs(team_obs, other, ball, inverted)

            # Extra info
            team_obs.extend([
                (other_car.position - player_car.position) / CustomObs.POS_STD,
                (other_car.linear_velocity - player_car.linear_velocity) / CustomObs.POS_STD
            ])

        obs.extend(allies)
        obs.extend(enemies)
        return np.concatenate(obs)

    def _add_player_to_obs(self, obs: List, player: PlayerData, ball: PhysicsObject, inverted: bool):
        if inverted:
            player_car = player.inverted_car_data
        else:
            player_car = player.car_data

        rel_pos = ball.position - player_car.position
        rel_vel = ball.linear_velocity - player_car.linear_velocity

        obs.extend([
            rel_pos / CustomObs.POS_STD,
            rel_vel / CustomObs.POS_STD,
            player_car.position / CustomObs.POS_STD,
            player_car.forward(),
            player_car.up(),
            player_car.linear_velocity / CustomObs.POS_STD,
            player_car.angular_velocity / CustomObs.ANG_STD,
            [int(player.has_flip),
             int(player.boost_amount),
             int(player.on_ground),
             int(player.is_alive)]])

        return player_car
