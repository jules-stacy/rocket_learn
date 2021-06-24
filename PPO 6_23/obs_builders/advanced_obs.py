import math
import numpy as np
from typing import Any, List
from rlgym.utils import common_values
from rlgym.utils.gamestates import PlayerData, GameState, PhysicsObject
from rlgym.utils.obs_builders import ObsBuilder
from rlutilities.linear_algebra import vec3
from rlutilities.simulation import Game, Ball
import sys
import os

Game.set_mode("soccar")

b = Ball()

def get_predict_locs(position, velocity, angular_velocity):


    # initialize it to a position in the middle of the field, some distance above the ground
    b.position = vec3(*position)
    b.velocity = vec3(*velocity)
    b.angular_velocity = vec3(*angular_velocity)

    predict_locs = []
    for i in range(720):

        frame_skip = 120

        # calling step() advances the Ball forward in time
        b.step(1.0 / 120.0)

        # and we can see where it goes
        if (i + 1) % frame_skip == 0:
            predict_locs.append(np.array((b.position[0], b.position[1], b.position[2])))
    return predict_locs

class AdvancedObs(ObsBuilder):
    POS_STD = 2300
    ANG_STD = math.pi

    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:

        if player.team_num == common_values.ORANGE_TEAM:
            inverted = True
            ball = state.inverted_ball
            pads = state.inverted_boost_pads
        else:
            inverted = False
            ball = state.ball
            pads = state.boost_pads

        predicted_paths = get_predict_locs(position=ball.position, velocity = ball.linear_velocity,angular_velocity= ball.angular_velocity)


       # print("Ball Position = ", ball.position / self.POS_STD)
       # print("predict Position 1 = ", predicted_paths[0])
       # print("predict Position 2 = ", predicted_paths[1])
       # print("predict Position 3 = ", predicted_paths[2])
       # print("predict Position 4 = ", predicted_paths[3])
       # print("predict Position 5 = ", predicted_paths[4])
       # print("predict Position 6 = ", predicted_paths[5])
        predicted_paths = [p / self.POS_STD for p in predicted_paths]
        obs = [ball.position / self.POS_STD,
               *predicted_paths,
               ball.linear_velocity / self.POS_STD,
               ball.angular_velocity / self.ANG_STD,
               previous_action,
               pads]

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
                (other_car.position - player_car.position) / self.POS_STD,
                (other_car.linear_velocity - player_car.linear_velocity) / self.POS_STD
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
            rel_pos / self.POS_STD,
            rel_vel / self.POS_STD,
            player_car.position / self.POS_STD,
            player_car.forward(),
            player_car.up(),
            player_car.linear_velocity / self.POS_STD,
            player_car.angular_velocity / self.ANG_STD,
            [player.boost_amount,
             int(player.on_ground),
             int(player.has_flip),
             int(player.is_demoed)]])

        return player_car
