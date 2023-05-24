import logging
import math

import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)

class MCTS():

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

    def getActionProb(self, board, is_training: bool, move_count=1):
        self.is_training = is_training

        count = self.args.numMCTSSims if is_training else self.args.numMCTSPlay

        for _ in range(count):
            self.search(board.copy(), cur_player=1)

        s = self.game.stringRepresentation(board)

        # visit counts for all actions in current state s = how many times the action was taken
        counts = [self.visitcount_stateaction[(s, a)] if (s, a) in self.visitcount_stateaction else 0 for a in range(self.game.getActionSize())]

        if not is_training:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [1 / float(len(counts)) for _ in counts]
            probs[bestA] = 1
            return probs

        temp = 1 if move_count < 20 else 0.1

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))

        if counts_sum == 0:
            # no actions were visited for this state
            # this happens/can happen at the very last simulation step
            print("WARNING: current main game state did not record any visitcounts, returning 1 for all actions")
            valids = self.game.getValidMoves(board, 1)
            probs = [1 / float(len(counts)) for _ in counts] * valids
            sum_probs = float(sum(probs))
            probs = [x / sum_probs for x in probs]
            return probs

        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, board, cur_player, depth=0):

        max_depth = self.args.maxMCTSDepth
        if (depth > max_depth):
            draw_value = 0
            # print(f"reached max depth {max_depth}, returning draw_value={draw_value}")
            return draw_value
        
        s = self.game.stringRepresentation(board)

        if s not in self.gameended_state:
            game_ended = self.game.getGameEnded(board, 1)
            self.gameended_state[s] = game_ended

        if self.gameended_state[s] != 0:
            # terminal node
            # since MCTS starts from "real game" position, a depth of 1 is normal
            game_ended = self.gameended_state[s]
            #print(f"terminal node at depth {depth}, result={game_ended}")

            self.visitcount_state[s] = 1
            return -game_ended
        
            # scale result by depth to favor faster wins. it will also punish faster loss more, and slow losses less.
            #return max(-game_ended * 2 - depth * 0.02, 0.25)

        if s not in self.policy_for_state:
            # leaf node
            self.policy_for_state[s], v = self.nnet.predict(board)
            v = v[0]

            # add noise to root node when training to avoid overfitting to a limited number of strategies.
            if depth == 0 and self.is_training:
                epsilon = 0.5 # noise ratio
                dirichlet_alpha = 0.5
                noise = np.random.dirichlet(np.ones(self.game.getActionSize()) * dirichlet_alpha)
                self.policy_for_state[s] = (1 - epsilon) * self.policy_for_state[s] + epsilon * noise
                self.policy_for_state[s] /= np.sum(self.policy_for_state[s])

            # mask invalid moves
            valids = self.game.getValidMoves(board, 1)
            self.policy_for_state[s] = self.policy_for_state[s] * valids

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

        v = self.search(next_s, cur_player * -1, depth + 1)

        # no penalty for draws, in a perfect game they the best moves and should not be discouraged
        val = v

        if (s, a) in self.q_for_stateaction:
            visit_count_sa = self.visitcount_stateaction[(s, a)]
            q_sa = self.q_for_stateaction[(s, a)]
            new_q = (visit_count_sa * q_sa + val) / (visit_count_sa + 1)

            self.q_for_stateaction[(s, a)] = new_q

        else:
            self.q_for_stateaction[(s, a)] = val

        return -v
