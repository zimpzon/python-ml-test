import Arena
from MCTS import MCTS
from TixyGame import TixyGame


import numpy as np
from TixyPlayers import TixyGreedyPlayer, TixyRandomPlayer
from utils import *

g = TixyGame(5, 5)

rp1 = TixyRandomPlayer(g).play
rp2 = TixyRandomPlayer(g).play

arena = Arena.Arena(rp1, rp2, g, display=TixyGame.display)

print(arena.playGames(2, verbose=True))
