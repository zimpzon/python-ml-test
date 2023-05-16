import Arena
from MCTS import MCTS
from TixyNNetWrapper import NNetWrapper as nn

import numpy as np
from TixyGame import TixyGame
from TixyNNetWrapper import NNetWrapper
from TixyPlayers import TixyRandomPlayer
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""


g = TixyGame(5, 5)

nnet = nn(g)

net = nnet.__class__(g) 
net.load_checkpoint(folder='./temp/', filename='temp.pth.tar')

pmcts = MCTS(g, nnet, self.args)


p1 = TixyRandomPlayer(g).play
p2 = TixyRandomPlayer(g).play

arena = Arena.Arena(p1, p2, g, display=TixyGame.display)

print(arena.playGames(100, verbose=False))
