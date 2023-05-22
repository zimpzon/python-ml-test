import Arena
from MCTS import MCTS
from TixyGame import TixyGame

import numpy as np
from TixyNNetWrapper import TixyNetWrapper
from TixyPlayers import TixyGreedyPlayer, TixyHumanPlayer, TixyRandomPlayer
from utils import *

g = TixyGame(5, 5)

rp1 = TixyGreedyPlayer(g).play
rp2 = TixyRandomPlayer(g).play

h = TixyHumanPlayer(g).play

n1 = TixyNetWrapper(g)
n1.load_checkpoint('./temp/','best.pth.tar')

args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, is_training=False, temp=0))

arena = Arena.Arena(rp1, n1p, g, display=TixyGame.display)

arena.playGames(20, verbose=False)
