from rlutilities.linear_algebra import vec3
from rlutilities.simulation import Game, Ball
import sys
import os

# Disable
def blockPrinting(func):
    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        sys.stdout = open(os.devnull, 'w')
        # call the method in question
        value = func(*args, **kwargs)
        # enable all printing to the console
        sys.stdout = sys.__stdout__
        # pass the return value of the method back
        return value

    return func_wrapper
# configure the Game to use the soccar collision geometry.

@blockPrinting
def setgamemode():
    Game.set_mode("soccar")

# make an instance of a Ball object
b = Ball()

# initialize it to a position in the middle of the field, some distance above the ground
b.position = vec3(0, 0, 0)
b.velocity = vec3(0, 0, 0)
b.angular_velocity = vec3(0, 0, 0)

for i in range(120):

    # calling step() advances the Ball forward in time
    b.step(1.0 / 120.0)

    # and we can see where it goes
    print(b.position)