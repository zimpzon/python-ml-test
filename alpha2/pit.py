import Arena
from MCTS import MCTS
from TixyGame import TixyGame

import numpy as np
from TixyNNetWrapper import TixyNetWrapper
from TixyPlayers import TixyGreedyPlayer, TixyHumanPlayer, TixyRandomPlayer
from utils import dotdict

args = dotdict({'numMCTSPlay': 50, 'cpuct':1.0, 'maxMCTSDepth': 100})

g = TixyGame(5, 5)

rp1 = TixyGreedyPlayer(g).play
rp2 = TixyRandomPlayer(g).play

h = TixyHumanPlayer(g).play

n1 = TixyNetWrapper(g)
n1.load_checkpoint('./temp/','best.pth.tar')

mcts1 = MCTS(g, n1, args)

n1p = lambda x: np.argmax(mcts1.getActionProb(x, is_training=False, temp=0))

n2 = TixyNetWrapper(g)
n2.load_checkpoint('./temp/','checkpoint_3.pth.tar')

mcts2 = MCTS(g, n2, args)

n2p = lambda x: np.argmax(mcts2.getActionProb(x, is_training=False, temp=0))

arena = Arena.Arena(h, n1p, g, display=TixyGame.display)

arena.playGames(10, verbose=False)
