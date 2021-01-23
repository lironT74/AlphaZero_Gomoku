from heuristic_player import Heuristic_player
from multiprocessing import Pool, set_start_method
from Game_boards_and_aux import *
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import pickle


def run_two_players_save_boards(player1, player2, width, height, n, game_board, n_games, sub_dir = "board_produce"):


    board_state, board_name, p1, p2, _, _ = game_board

    win_cnt = defaultdict(int)

    boards_to_save_alpha_beta = []

    for i in range(n_games):

        # print(f"game {i+1}:\t {player1.name} vs {player2.name} ({cur_time()})")

        i_board1, board1 = initialize_board_without_init_call(width, height, n, input_board=board_state, open_path_threshold=-1)
        game = Game(board1)

        path = f'/home/lirontyomkin/AlphaZero_Gomoku/{sub_dir}/{player1.name} vs {player2.name}/game_{i+1}/'



        winner, boards_list_for_gttt, game_length =            game.start_play_just_game_capture(path,
                                                               player1,
                                                               player2,
                                                               start_player=i % 2 + 1,
                                                               is_shown=1,
                                                               game_num=i+1)

        # -1 winner (end board) 0
        # -2 losser 1
        # -3 winner 2
        # -4 losser 3
        # -5 winner 4
        # -6 losser 5
        # -7 winner (was he able to promise victory here?) 6
        # -8 losser (was he able to promise victory here?) 7

        win_cnt[winner] += 1

        if winner != -1 and game_length > 15:

            for depth in range(7, 11, 2):

                current_player, curr_gttt_board = boards_list_for_gttt[-depth]

                if depth % 2 == 1:
                    assert current_player == winner
                else:
                    assert current_player == 3 - winner

                board_to_check_with_alpha_beta = {}
                board_to_check_with_alpha_beta['board_intial'] = curr_gttt_board
                board_to_check_with_alpha_beta['whos_turn'] = current_player
                board_to_check_with_alpha_beta['max_depth'] = depth - 1
                board_to_check_with_alpha_beta['game_num'] = i + 1

                boards_to_save_alpha_beta.append(board_to_check_with_alpha_beta)



    print(f"{player1.name} wins: {win_cnt[1]}, {player2.name} wins: {win_cnt[2]}, ties: {win_cnt[-1]}")


    gtt_path = f'/home/lirontyomkin/AlphaZero_Gomoku/{sub_dir}/gttt_boards/'

    with open(f'{gtt_path}/{player1.name} vs {player2.name}_gtttboards', 'wb') as f:
        pickle.dump(boards_to_save_alpha_beta, f)



if __name__ == '__main__':

    o_weight = 0.5
    density = Heuristic_player(name="density heuristic", heuristic="density", o_weight=o_weight)
    linear = Heuristic_player(name="linear heuristic", heuristic="linear", o_weight=o_weight)
    nonlinear = Heuristic_player(name="nonlinear heuristic", heuristic="nonlinear", o_weight=o_weight)
    interaction = Heuristic_player(name="interaction heuristic", heuristic="interaction", o_weight=o_weight)
    interaction_with_forcing = Heuristic_player(name="forcing heuristic", heuristic="interaction with forcing", o_weight=o_weight)


    players = [density, linear, nonlinear, interaction, interaction_with_forcing]

    width = 6
    height = 6
    n = 4
    game_board = EMPTY_BOARDS_DICT[width]
    n_games = 2000


    sub_dir = f"board_produce/{width}X{height}"

    jobs = []

    gtt_path = f'/home/lirontyomkin/AlphaZero_Gomoku/{sub_dir}/gttt_boards/'
    if not os.path.exists(gtt_path):
        os.makedirs(gtt_path)

    for i in range(len(players)):
        for j in range(i, len(players)):
            jobs.append((players[i], players[j], width, height, n, game_board, n_games, sub_dir))


    with Pool() as pool:
        pool.starmap(run_two_players_save_boards, jobs)
        pool.close()
        pool.join()


