from rlgym.utils.gamestates import GameState
from rlgym.utils.terminal_conditions import TerminalCondition


class CustomTerminalCondition(TerminalCondition):
  def reset(self, initial_state: GameState):
    pass

  def is_terminal(self, current_state: GameState) -> bool:
    blue_score: int
    orange_score: int
    # initialize blue and orange scores

    # make a check to see if a goal was scored
    if blue_score < current_state.blue_score or orange_score < current_state.orange_score:
      # update scores and return True
      blue_score = current_state.blue_score
      orange_score = current_state.orange_score
      return True
    else:
      pass




