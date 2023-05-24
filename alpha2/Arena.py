import logging

from tqdm import tqdm

log = logging.getLogger(__name__)


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        cur_player = 1
        board = self.game.getInitBoard()
        it = 0
        
        while self.game.getGameEnded(board, 1) == 0:
            it += 1
            max_depth = 500 # pwe todo: hardcoded for now
            if (it > max_depth):
                return 0

            if verbose:
                assert self.display
                copy = self.game.turnBoard(board) if cur_player == -1 else board
                print("Turn ", str(it), "Player ", str(cur_player))
                self.display(copy)
                print("\n")

            action = players[cur_player + 1](board)
            valids = self.game.getValidMoves(board)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0
                
            board, cur_player = self.game.getNextState(board, cur_player, action)
            board = self.game.turnBoard(board)

        if verbose:
            assert self.display
            print("Final move ", str(it), "Player ", str(-cur_player))
            self.display(board)

        # the player just taking a turn will always be the winning player if game is decided
        winning_player = -cur_player # minus since we already switched player
        
        if verbose:
            assert self.display
            print("Game over: Turn ", str(it), "Result ", str(winning_player))
            copy = self.game.turnBoard(board) if cur_player == -1 else board
            self.display(copy)

        return winning_player

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1

        print("pl1: %d, pl2: %d, draw: %d" % (oneWon, twoWon, draws))
              
        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1

        print("pl1: %d, pl2: %d, draw: %d" % (oneWon, twoWon, draws))

        # prev, new
        return oneWon, twoWon, draws
