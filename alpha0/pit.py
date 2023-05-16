import Arena
from MCTS import MCTS
from TixyGame import TixyGame

import numpy as np
from TixyPlayers import TixyGreedyPlayer, TixyRandomPlayer
from utils import *

g = TixyGame(5, 5)

rp1 = TixyRandomPlayer(g, 1).play
rp2 = TixyRandomPlayer(g, -1).play

# in round two wins for -1 counts towards player1 wins! This makes sure wins are counted per player type, not per player id
rp3 = TixyGreedyPlayer(g, 1).play
rp4 = TixyRandomPlayer(g, -1).play

arena = Arena.Arena(rp1, rp2, rp3, rp4, g, display=TixyGame.display)

print(arena.playGames(500, verbose=False))
