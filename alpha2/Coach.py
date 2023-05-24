import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from Arena import Arena
from MCTS import MCTS
from TixyGame import TixyGame
from TixyPlayers import TixyGreedyPlayer

log = logging.getLogger(__name__)


class Coach():

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def executeEpisode(self):

        # trainExamples is (state, pi, player)
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1

            # check if episode has been going for too long, probably reached a loop state
            # this is a draw, return as a loss to discourage states leading to this
            max_depth = self.args.maxMCTSDepth
            if (episodeStep > max_depth):
                print(f"episode was a draw ending at step {episodeStep}, returning 0")
                return [(x[0], x[1], 0) for x in trainExamples]

            temp = int(episodeStep < self.args.tempThreshold)
            if episodeStep == self.args.tempThreshold:
                log.info(f'Setting temperature to {temp}')

            pi = self.mcts.getActionProb(board, is_training=True, move_count=episodeStep)

            sym = self.game.getSymmetries(board, pi)
            for b, p in sym:
                trainExamples.append([b, p, self.curPlayer])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            # the player just taking a turn will always be the winning player if game is decided
            r = self.game.getGameEnded(board, 1)

            if r != 0:
                winning_player = -self.curPlayer # minus since we already switched player
                # print(f'Game ended with result {winning_player}')
                return [(x[0], x[1], 1 if x[2] == winning_player else -1) for x in trainExamples]

            board = self.game.turnBoard(board)

    def learn(self):

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    iterationTrainExamples += self.executeEpisode()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)

            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples)

            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='just-before-compare.pth.tar')

            nmcts = MCTS(self.game, self.nnet, self.args)

            log.info('PITTING AGAINST GREEDY PLAYER')
            new_against_rnd = MCTS(self.game, self.nnet, self.args)
            arena = Arena(TixyGreedyPlayer(self.game).play,
                          lambda x: np.argmax(new_against_rnd.getActionProb(x, is_training=False)),
                          self.game,
                          display = TixyGame.display)
            
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare, verbose=False)
            log.info('NEW/GREEDY : %d / %d (draws: %d)' % (nwins, pwins, draws))

            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, is_training=False)),
                          lambda x: np.argmax(nmcts.getActionProb(x, is_training=False)), self.game, display = TixyGame.display)
            
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare, verbose=False)

            log.info('NEW/PREV WINS : %d / %d (draws: %d)' % (nwins, pwins, draws))

            # do not want draws, count them as losses
            pwins += draws

            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
