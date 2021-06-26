from rlutilities.linear_algebra import vec3
from rlutilities.simulation import Game, Ball
import numpy as np


def get_predict_locs(self, position, velocity, angular_velocity):
    Game.set_mode("soccar")

    # make an instance of a Ball object
    b = Ball()

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
            predict_locs.append(np.array(b.position))
    return np.vstack((predict_locs))