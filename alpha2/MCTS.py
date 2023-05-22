import logging
import math

import numpy as np

from utils import Info

EPS = 1e-8

log = logging.getLogger(__name__)

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.q_for_stateaction = {}  # stores Q values for s,a (as defined in the paper)
        self.visitcount_stateaction = {}  # stores #times edge s,a was visited
        self.visitcount_state = {}  # stores #times board s was visited
        self.policy_for_state = {}  # stores initial policy (returned by neural net)

        self.gameended_state = {}  # stores game.getGameEnded ended for board s
        self.validmoves_state = {}  # stores game.getValidMoves for board s

    def getActionProb(self, board, is_training: bool, temp=1):
        count = self.args.numMCTSSims if is_training else self.args.numMCTSPlay
        good = 0
        bad = 0
        for i in range(count):
            result = self.search(board.copy())
            if abs(result) < 0.0002:
                bad += 1
            else:
                good += 1

        ratio = (bad + 0.001)/(good + 0.001)
        if ratio > 0.1:
            print(f"WARNING more than 0.1 reached maxDepth. below={good}, MAXED={bad}, ratio {ratio:0.4f}")

        s = self.game.stringRepresentation(board)
        counts = [self.visitcount_stateaction[(s, a)] if (s, a) in self.visitcount_stateaction else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, board, depth=0):

        # print("search depth " + str(depth))
        if (depth > 100):
            # print("max depth reached")
            return -0.0001
        
        s = self.game.stringRepresentation(board)

        if s not in self.gameended_state:
            game_ended = self.game.getGameEnded(board, 1)
            self.gameended_state[s] = game_ended

        if self.gameended_state[s] != 0:
            # terminal node
            # since MCTS starts from "real game" position, a depth of 1 is normal
            game_ended = self.gameended_state[s]
            # print(f"terminal node at depth {depth}, result={game_ended}")
            return -game_ended

        if s not in self.policy_for_state:
            # leaf node
            self.policy_for_state[s], v = self.nnet.predict(board)

            # this dirichlet noise code was made by Codepilot. Let's try it :-) changing 0.3 to my own guess.
            if depth == 0:
                self.policy_for_state[s] = self.policy_for_state[s] + np.random.dirichlet(np.ones(self.game.getActionSize()) * 0.8)
                self.policy_for_state[s] /= np.sum(self.policy_for_state[s])

            #self.policy_for_state[s] = [0.5] * len(self.policy_for_state[s])
            v = v[0]
            #v = 0.5

            valids = self.game.getValidMoves(board, 1)
            self.policy_for_state[s] = self.policy_for_state[s] * valids  # masking invalid moves
            sum_policy_state = np.sum(self.policy_for_state[s])
            if sum_policy_state > 0:
                self.policy_for_state[s] /= sum_policy_state  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                log.error("All valid moves were masked, doing a workaround.")
                self.policy_for_state[s] = self.policy_for_state[s] + valids
                self.policy_for_state[s] /= np.sum(self.policy_for_state[s])

            self.validmoves_state[s] = valids
            self.visitcount_state[s] = 0

            # print(f"expanded node at depth {depth}, v: {v:.8f}, player: {Info.getPlayerId(self.game, board, valids)}")
            return -v

        valids = self.validmoves_state[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.q_for_stateaction:
                    q_sa = self.q_for_stateaction[(s, a)]
                    sa_policy = self.policy_for_state[s][a]
                    visit_count_s = self.visitcount_state[s]
                    visit_count_sa = self.visitcount_stateaction[(s, a)]

                    u = (q_sa + self.args.cpuct * sa_policy * math.sqrt(visit_count_s) / (1 + visit_count_sa ))
                else:
                    sa_policy = self.policy_for_state[s][a]
                    visit_count_s = self.visitcount_state[s]

                    u = self.args.cpuct * sa_policy * math.sqrt(visit_count_s + EPS)  # Q = 0 ?

                u = float(u)
                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        # print(f"action {a}, depth: {depth}, player: {Info.getPlayerId(self.game, board, valids)}")

        if (s, a) not in self.visitcount_stateaction:
            self.visitcount_stateaction[(s, a)] = 1
        else:
            self.visitcount_stateaction[(s, a)] += 1

        self.visitcount_state[s] += 1
        next_s, _ = self.game.getNextState(board, 1, a)
        next_s = self.game.turnBoard(next_s)

        v = self.search(next_s, depth + 1)
        # print(f"backpropagating v: {v:.8f}, depth: {depth}, player: {Info.getPlayerId(self.game, board, valids)}")

        if (s, a) in self.q_for_stateaction:
            visit_count_sa = self.visitcount_stateaction[(s, a)]
            q_sa = self.q_for_stateaction[(s, a)]
            new_q = (visit_count_sa * q_sa + v) / (visit_count_sa + 1)

            self.q_for_stateaction[(s, a)] = new_q
            # self.visitcount_stateaction[(s, a)] += 1

        else:
            self.q_for_stateaction[(s, a)] = v
            # self.visitcount_stateaction[(s, a)] = 1

        # self.visitcount_state[s] += 1
        return -v
