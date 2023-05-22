import logging

import coloredlogs

from Coach import Coach
from TixyGame import TixyGame as Game
from TixyNNetWrapper import TixyNetWrapper as nn
from utils import dotdict

log = logging.getLogger(__name__)

# ------------------- TODO -------------------
#  BUG?? When it starts hitting max depth progress seems to stop moving. But it did complete after a while.

#  ADD DIRICHLET NOISE WHEN TRAINING
#  MULTIPROCESSING!
#    or, do simulation in C#. Load model, run sum, save output, load output from python, continue. Can then easily run multithreaded and mucsh faster.
#    eval could also be C#, since it is quite slow. Should speed thing up a crazy amount.     
#  Train with random starting board? evail against previous must be symmetric, but training could be asymmetric

#  Useful stats/graph:
#   win rate in self-play (to see if player 1 or 2 wins all)
#   number of max depth reached
#   model accepted/rejected + win rate
#   model loss (mostly for fun? since target keeps changing this is less useful)
#   any way to measure performance of best model? greedy is not good enough to win ever (which is good!) could be a pure mcts?

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

# args = dotdict({
#     'numIters': 2,
#     'numEps': 2,              # Number of complete self-play games to simulate during a new iteration.
#     'tempThreshold': 1000,        #
#     'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
#     'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
#     'numMCTSSims': 2,          # Number of games moves for MCTS to simulate.
#     'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
#     'cpuct': 1,

#     'checkpoint': './temp/',
#     'load_model': False,
#     'load_folder_file': ('/dev/models/8x100x50_pwe','best.pth_pwe.tar'),
#     'numItersForTrainExamplesHistory': 20,
# })

args = dotdict({
    'numIters': 1000,
    'numEps': 20,               # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 1000,      #
    'updateThreshold': 0.55,    # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 200,        # PWE: This should be quite high? Alpha0 used 80000 for chess?
    'numMCTSPlay': 100,        # how low can we take this?
    'maxMCTSDepth': 100,        # how low can we take this?
    'arenaCompare': 30,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': True,
    'load_folder_file': ('./temp/', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(5, 5)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
