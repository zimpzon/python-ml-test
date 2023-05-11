import Arena
from MCTS import MCTS
from TixyGame import TixyGame


import numpy as np
from TixyPlayers import TixyRandomPlayer
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""
g = TixyGame(5, 5)

# LOOKING GOOD! Random players win at the expected ~50% ratio
rp1 = TixyRandomPlayer(g).play
rp2 = TixyRandomPlayer(g).play

arena = Arena.Arena(rp1, rp2, g, display=TixyGame.display)

print(arena.playGames(100, verbose=True))
