import Arena
from MCTS import MCTS
from TixyGame import TixyGame

import numpy as np
from TixyNNetWrapper import TixyNetWrapper as nn
from TixyPlayers import TixyGreedyPlayer, TixyRandomPlayer

args = dotdict({
    'numMCTSPlay': 50,
    'cpuct': 1,
})

g = TixyGame(5, 5)

n1 = nn(g)
n1.load_checkpoint(folder='./temp/', filename='best.pth.tar')
n1p = lambda x: np.argmax(n1.getactionprob(x, temp=0))

mcts = MCTS(g,n1, args)

rp1 = TixyRandomPlayer(g, 1).play
rp2 = TixyRandomPlayer(g, -1).play

# in round two wins for -1 counts towards player1 wins! This makes sure wins are counted per player type, not per player id
rp3 = TixyGreedyPlayer(g, 1).play
rp4 = TixyRandomPlayer(g, -1).play

arena = Arena.Arena(rp1, n1p, g, display=TixyGame.display)

print(arena.playGames(10, verbose=False))
