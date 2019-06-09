from mcts_pure import MCTS, MCTSPlayer
from policy_player import PolicyPlayer
from policy_net_keras import PolicyNet
from game import Game, BoardSlim
import numpy as np
import tqdm
PATH2MODEL = "/Users/danamir/tictactoe/AlphaZero_Gomoku/1555508174/policy14999.h5"
BOARD_SHAPE = (6,6)
playouts = [1]
num_games = 20

def load_player():
    network = PolicyNet(BOARD_SHAPE[0], BOARD_SHAPE[1], PATH2MODEL)
    player = PolicyPlayer(network, is_selfplay=False)
    return player


def evaluate_player(player, opponent, show_game=False):
    # for i in range(n_games):
    #     winner = self.game.start_play(current_player,
    #                                   pure_mcts_player,
    #                                   start_player=i % 2,
    #                                   is_shown=0)
    #     win_cnt[winner] += 1
    # win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
    # print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
    #     self.pure_mcts_playout_num,
    #     win_cnt[1], win_cnt[2], win_cnt[-1]))
    # return win_ratio
    stats = {"wins": 0, "length": [], "ties": 0}
    for i in tqdm.trange(num_games):
        opponent.reset_player()
        board = BoardSlim(width=BOARD_SHAPE[0], height=BOARD_SHAPE[1], n_in_row=4)
        game = Game(board)
        winner = game.start_play(player, opponent, start_player= i % 2, is_shown=(i == 0) and show_game)
        print("winner: ", winner)
        if winner == 1:
            stats["wins"] += 1
        elif winner == -1:
            stats["ties"] += 1
        stats["length"].append(BOARD_SHAPE[0] * BOARD_SHAPE[1] - len(board.availables))
    return stats



if __name__ == "__main__":
    player = load_player()
    for playout in playouts:
        mcts_player = MCTSPlayer(n_playout=playout)
        stats = evaluate_player(player, mcts_player, show_game=True)
        print()
        print("win ratio agains %d playouts: " % playout, stats["wins"] / num_games)
        print("tie ratio agains %d playouts: " % playout, stats["ties"] / num_games)
        print("average game length agains %d playouts: ", np.mean(stats["length"]))