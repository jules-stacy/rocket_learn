from typing import *
from typing import Iterable as iterable
from typing import Iterator as iterator
from numpy import float64

import rlutilities

_Shape = Tuple[int, ...]
from rlutilities.linear_algebra import *
__all__  = [
"Ball",
"BoostPad",
"BoostPadState",
"BoostPadType",
"Car",
"ControlPoint",
"Curve",
"Field",
"Game",
"GameState",
"Goal",
"Input",
"Navigator",
"obb",
"ray",
"sphere",
"tri",
"intersect"
]
class Ball():
    collision_radius = 93.1500015258789
    drag = -0.030500000342726707
    friction = 2.0
    mass = 30.0
    max_omega = 6.0
    max_speed = 4000.0
    moment_of_inertia = 99918.75
    radius = 91.25
    restitution = 0.6000000238418579

    @overload
    def __init__(self) -> None: 
        pass
    @overload
    def __init__(self, arg0: Ball) -> None: ...
    def hitbox(self) -> sphere: ...
    @overload
    def step(self, arg0: float, arg1: Car) -> None: 
        pass
    @overload
    def step(self, arg0: float) -> None: ...

    angular_velocity: vec3
    position: vec3
    time: float
    velocity: vec3
    pass
class BoostPad():

    def __init__(self) -> None: ...

    position: vec3
    state: BoostPadState
    timer: float
    type: BoostPadType
    pass
class BoostPadState():
    Available: rlutilities.simulation.BoostPadState = None # type: rlutilities.simulation.BoostPadState
    Unavailable: rlutilities.simulation.BoostPadState = None # type: rlutilities.simulation.BoostPadState
    __entries: dict = None # type: dict
    __members__: dict = None # type: dict

    def __eq__(self, arg0: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __init__(self, arg0: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, arg0: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, arg0: int) -> None: ...

    name: None
    pass
class BoostPadType():
    Full: rlutilities.simulation.BoostPadType = None # type: rlutilities.simulation.BoostPadType
    Partial: rlutilities.simulation.BoostPadType = None # type: rlutilities.simulation.BoostPadType
    __entries: dict = None # type: dict
    __members__: dict = None # type: dict

    def __eq__(self, arg0: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __init__(self, arg0: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, arg0: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, arg0: int) -> None: ...

    name: None
    pass
class Car():

    @overload
    def __init__(self, arg0: Car) -> None: 
        pass
    @overload
    def __init__(self) -> None: ...
    def extrapolate(self, arg0: float) -> None: ...
    def forward(self) -> vec3: ...
    def hitbox(self) -> obb: ...
    def left(self) -> vec3: ...
    def step(self, arg0: Input, arg1: float) -> None: ...
    def up(self) -> vec3: ...

    angular_velocity: vec3
    boost: int
    controls: Input
    demolished: bool
    dodge_timer: float
    double_jumped: bool
    hitbox_offset: vec3
    hitbox_widths: vec3
    id: int
    jump_timer: float
    jumped: bool
    on_ground: bool
    orientation: mat3
    position: vec3
    supersonic: bool
    team: int
    time: float
    velocity: vec3
    pass
class ControlPoint():

    @overload
    def __init__(self, arg0: vec3, arg1: vec3, arg2: vec3) -> None: 
        pass
    @overload
    def __init__(self) -> None: ...

    n: vec3
    p: vec3
    t: vec3
    pass
class Curve():

    @overload
    def __init__(self, arg0: List[vec3]) -> None: 
        pass
    @overload
    def __init__(self, arg0: List[ControlPoint]) -> None: ...
    def calculate_distances(self) -> None: ...
    def calculate_max_speeds(self, arg0: float, arg1: float) -> float: ...
    def calculate_tangents(self) -> None: ...
    def curvature_at(self, arg0: float) -> float: ...
    def find_nearest(self, arg0: vec3) -> float: ...
    def max_speed_at(self, arg0: float) -> float: ...
    def point_at(self, arg0: float) -> vec3: ...
    def pop_front(self) -> None: ...
    def tangent_at(self, arg0: float) -> vec3: ...
    def write_to_file(self, arg0: str) -> None: ...

    length: float
    points: List[vec3]
    pass
class Field():
    mode = 'Uninitialized'
    triangles = []

    @staticmethod
    @overload
    def collide(arg0: sphere) -> ray: 
        pass
    @staticmethod
    @overload
    def collide(arg0: obb) -> ray: ...
    @staticmethod
    def raycast_any(arg0: ray) -> ray: ...
    @staticmethod
    def raycast_nearest(arg0: ray) -> ray: ...

    pass
class Game():
    gravity: rlutilities.linear_algebra.vec3 = None # type: rlutilities.linear_algebra.vec3
    map = 'map_not_set'

    def __init__(self) -> None: ...
    def read_field_info(self, arg0: object) -> None: ...
    def read_packet(self, arg0: object) -> None: ...
    @staticmethod
    def set_mode(arg0: str) -> None: ...

    ball: Ball
    cars: List[Car]
    frame: int
    goals: List[Goal]
    pads: List[BoostPad]
    state: GameState
    time: float
    time_remaining: float
    pass
class GameState():
    Active: rlutilities.simulation.GameState = None # type: rlutilities.simulation.GameState
    Countdown: rlutilities.simulation.GameState = None # type: rlutilities.simulation.GameState
    Ended: rlutilities.simulation.GameState = None # type: rlutilities.simulation.GameState
    GoalScored: rlutilities.simulation.GameState = None # type: rlutilities.simulation.GameState
    Inactive: rlutilities.simulation.GameState = None # type: rlutilities.simulation.GameState
    Kickoff: rlutilities.simulation.GameState = None # type: rlutilities.simulation.GameState
    Paused: rlutilities.simulation.GameState = None # type: rlutilities.simulation.GameState
    Replay: rlutilities.simulation.GameState = None # type: rlutilities.simulation.GameState
    __entries: dict = None # type: dict
    __members__: dict = None # type: dict

    def __eq__(self, arg0: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __init__(self, arg0: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, arg0: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, arg0: int) -> None: ...

    name: None
    pass
class Goal():

    def __init__(self) -> None: ...

    direction: vec3
    height: float
    position: vec3
    team: int
    width: float
    pass
class Input():

    def __init__(self) -> None: ...

    boost: bool
    handbrake: bool
    jump: bool
    pitch: float
    roll: float
    steer: float
    throttle: float
    use_item: bool
    yaw: float
    pass
class Navigator():
    nodes: list = None # type: list

    def __init__(self, arg0: Car) -> None: ...
    def analyze_surroundings(self, arg0: float) -> None: ...
    def path_to(self, arg0: vec3, arg1: vec3, arg2: float) -> Curve: ...

    pass
class obb():

    def __init__(self) -> None: ...

    center: vec3
    half_width: vec3
    orientation: mat3
    pass
class ray():

    @overload
    def __init__(self, arg0: vec3, arg1: vec3) -> None: 
        pass
    @overload
    def __init__(self) -> None: ...

    direction: vec3
    start: vec3
    pass
class sphere():

    @overload
    def __init__(self) -> None: 
        pass
    @overload
    def __init__(self, arg0: vec3, arg1: float) -> None: ...

    center: vec3
    radius: float
    pass
class tri():

    def __getitem__(self, arg0: int) -> vec3: ...
    def __init__(self) -> None: ...
    def __setitem__(self, arg0: int, arg1: vec3) -> None: ...

    pass
@overload
def intersect(arg0: sphere, arg1: obb) -> bool:
    pass
@overload
def intersect(arg0: obb, arg1: sphere) -> bool:
    pass