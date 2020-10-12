from __future__ import print_function
import pickle
from multiprocessing import Pool, set_start_method
from mcts_alphaZero import *
from heuristic_player import Heuristic_player
from scipy.special import comb
import pandas as pd
import os
from mcts_pure import MCTSPlayer as PUREMCTS
import PIL


def compare_all_models_statistics(players_list, opponents, width=6, height=6, n=4, num_games=100):

    jobs = []
    for opponent_player in opponents:
        for board in PAPER_TRUNCATED_BOARDS:
            jobs.append((players_list, opponent_player, board, width, height, n, num_games, 2))

        for board in PAPER_FULL_BOARDS:
            jobs.append((players_list, opponent_player, board, width, height, n, num_games, 2))

        jobs.append((players_list, opponent_player, EMPTY_BOARD, width, height, n, num_games, 1))


    with Pool(len(jobs)) as pool:
        print(f"Using {pool._processes} workers. There are {len(jobs)} jobs: \n")
        pool.starmap(collect_statistics_againts_opponent, jobs)
        pool.close()
        pool.join()

    # collect_statistics_againts_opponent(*jobs[0])


def collect_statistics_againts_opponent(players_list, opponent_player, board, width, height, n, num_games, start_player):

    board_state, board_name, p1, p2, alternative_p1, alternative_p2 = board

    columns = [
        f"no. games",
        f"no. wins",
        f"no. losses",
        f"no. ties",
        f"win ratio",

        f"avg game len",
        f"avg game len (wins)",
        f"avg game len (losses)",

        f"avg shutter size",
        f"avg shutter size (wins)",
        f"avg shutter size (losses)",

        f"fraction of plays with shutter",
        f"fraction of plays with shutter (wins)",
        f"fraction of plays with shutter (losses)",

        f"avg shutter size (real last turn)",
        f"avg shutter size (real last turn - wins)",
        f"avg shutter size (real last turn - losses)",

        f"fraction of plays with shutter (real last turn)",
        f"fraction of plays with shutter (real last turn - wins)",
        f"fraction of plays with shutter (real last turn - losses)",

        # f"no. plays with shutter",
        # f"total no. of plays",
        #
        # f"no. plays with shutter (wins)",
        # f"total no. of plays (wins)",
        #
        #
        # f"no. plays with shutter (losses)",
        # f"total no. of plays (losses)",

    ]

    result_df = pd.DataFrame(0, index=[player.name for player in players_list], columns=columns)

    path = f"/home/lirontyomkin/AlphaZero_Gomoku/matches/statistics/vs {opponent_player.name}/{board_name}/"

    # already_saved_df = pd.read_excel(f"{path}all models {num_games} games results.xlsx", index_col=0)
    # already_saved_names = ["v7_1500", "v9_1500", "v10_1500", "v10_1500_random"]
    #
    # for model_name in already_saved_names:
    #     result_df.loc[model_name] = already_saved_df.loc[model_name]


    # for cur_player in players_list:
    #
    #     # if cur_player.name in already_saved_names:
    #     #     print(f"Skipping {cur_player.name} ({opponent_player.name})")
    #     #     continue
    #
    #     result =              save_games_statistics(width=width,
    #                           height=height,
    #                           n=n,
    #                           board_state=board_state,
    #                           board_name=board_name,
    #                           cur_player=cur_player,
    #                           opponent_player=opponent_player,
    #                           last_move_p1=p1,
    #                           last_move_p2=p2,
    #                           correct_move_p1=p1,
    #                           correct_move_p2=p2,
    #                           start_player=start_player,
    #                           num_games=num_games)
    #
    #     result_df.loc[cur_player.name] = result
    #
    # print(f"{opponent_player.name}")
    # print(result_df.to_string())
    #
    # result_df.to_excel(f"{path}all models {num_games} games results.xlsx")

    create_statistics_graphics(path, num_games, opponent_player.name)


def save_games_statistics(width, height, n, board_state, board_name, cur_player,
                          opponent_player, last_move_p1, last_move_p2, correct_move_p1,
                          correct_move_p2, start_player, num_games):

    i_board1, board1 = initialize_board_without_init_call(width, height, n, input_board=board_state)
    game1 = Game(board1)

    wins = []

    shutters_wins = []
    real_last_turn_shutters_cur_player_wins = []

    shutters_losses = []
    real_last_turn_shutters_cur_player_losses = []

    shutters_ties = []
    real_last_turn_shutters_cur_player_ties = []

    total_plays = 0
    total_plays_wins = 0
    total_plays_losses = 0

    if start_player == 1:
        player_by_index = {1: cur_player, 2: opponent_player}

    else:
        player_by_index = {2: cur_player, 1: opponent_player}

    # games_history = []

    path = f"/home/lirontyomkin/AlphaZero_Gomoku/matches/statistics/vs {opponent_player.name}/{board_name}/"
    games_history = pickle.load(open(f"{path}{cur_player.name}/full_{num_games}_games_stats", 'rb'))


    for i in range(num_games):
        # winner, game_length, shutter_sizes, real_last_move_shutter_sizes, game_history = game1.start_play(player1=player_by_index[1],
        #                                                      player2=player_by_index[2],
        #                                                      start_player=start_player,
        #                                                      is_shown=0,
        #                                                      start_board=i_board1,
        #                                                      last_move_p1=last_move_p1,
        #                                                      last_move_p2=last_move_p2,
        #                                                      correct_move_p1=correct_move_p1,
        #                                                      correct_move_p2=correct_move_p2,
        #                                                      is_random_last_turn_p1=player_by_index[1].is_random_last_turn,
        #                                                      is_random_last_turn_p2=player_by_index[2].is_random_last_turn,
        #                                                      savefig=0,
        #                                                      board_name=board_name,
        #                                                      return_statistics=1)
        #
        # games_history.append((winner, game_length, shutter_sizes, real_last_move_shutter_sizes, game_history))


        winner, game_length, shutter_sizes, real_last_move_shutter_sizes, game_history = games_history[i]


        plays = range(1, game_length + 1, 1)
        start_player_range = [(index, shutter) for index, shutter in zip(plays[0::2], shutter_sizes[start_player]) if shutter != -1]


        if winner != -1:
            wins.append((player_by_index[winner].name, game_length))

            if winner == start_player:
                shutters_wins.extend([x[1] for x in start_player_range])
                total_plays_wins += len(plays[0::2])
            else:
                shutters_losses.extend([x[1] for x in start_player_range])
                total_plays_losses += len(plays[0::2])

        else:
            shutters_ties.extend([x[1] for x in start_player_range])



        if cur_player.is_random_last_turn:

            start_player_range = [(index, shutter) for index, shutter in
                                  zip(plays[0::2], real_last_move_shutter_sizes[start_player])
                                  if
                                  shutter != -1]

            if winner == start_player:
                real_last_turn_shutters_cur_player_wins.extend([x[1] for x in start_player_range])
            elif winner == 3 - start_player:
                real_last_turn_shutters_cur_player_losses.extend([x[1] for x in start_player_range])
            else:
                real_last_turn_shutters_cur_player_ties.extend([x[1] for x in start_player_range])


        total_plays += len(plays[0::2])


    no_games = num_games
    no_wins = [x[0] for x in wins].count(cur_player.name)
    no_losses = [x[0] for x in wins].count(opponent_player.name)
    no_ties = num_games - no_wins - no_losses
    win_ratio = 1.0*(no_wins + 0.5*no_ties) / no_games

    average_game_length = np.average([x[1] for x in wins]) if len([x[1] for x in wins]) >0 else -1
    average_game_length_wins = np.average([x[1] for x in wins if x[0] == cur_player.name]) if len([x[1] for x in wins if x[0] == cur_player.name]) > 0 else -1
    average_game_length_losses = np.average([x[1] for x in wins if x[0] == opponent_player.name]) if len([x[1] for x in wins if x[0] == opponent_player.name]) > 0 else -1


    avg_shutter_size = np.average(shutters_wins + shutters_losses + shutters_ties) if len(shutters_wins + shutters_losses + shutters_ties) > 0 else -1
    avg_shutter_size_wins = np.average(shutters_wins) if len(shutters_wins) > 0 else -1
    avg_shutter_size_losses = np.average(shutters_losses) if len(shutters_losses) > 0 else -1

    plays_with_shutter_fraction = len(shutters_wins + shutters_losses + shutters_ties)/total_plays
    plays_with_shutter_fraction_wins = len(shutters_wins)/total_plays_wins
    plays_with_shutter_fraction_losses = len(shutters_losses)/total_plays_losses


    avg_shutter_size_real_last_turn = -1
    avg_shutter_size_wins_real_last_turn = -1
    avg_shutter_size_losses_real_last_turn = -1
    plays_with_shutter_fraction_real_last_turn = -1
    plays_with_shutter_fraction_wins_real_last_turn = -1
    plays_with_shutter_fraction_losses_real_last_turn = -1


    if cur_player.is_random_last_turn:

        avg_shutter_size_real_last_turn = np.average(real_last_turn_shutters_cur_player_wins +
                                                     real_last_turn_shutters_cur_player_losses +
                                                     real_last_turn_shutters_cur_player_ties) if len(
                                                    real_last_turn_shutters_cur_player_wins +
                                                    real_last_turn_shutters_cur_player_losses +
                                                    real_last_turn_shutters_cur_player_ties) > 0 else -1

        avg_shutter_size_wins_real_last_turn = np.average(real_last_turn_shutters_cur_player_wins) if len(real_last_turn_shutters_cur_player_wins) > 0 else -1
        avg_shutter_size_losses_real_last_turn = np.average(real_last_turn_shutters_cur_player_losses) if len(real_last_turn_shutters_cur_player_losses) > 0 else -1

        plays_with_shutter_fraction_real_last_turn = len(real_last_turn_shutters_cur_player_wins +
                                                     real_last_turn_shutters_cur_player_losses +
                                                     real_last_turn_shutters_cur_player_ties) / total_plays

        plays_with_shutter_fraction_wins_real_last_turn = len(real_last_turn_shutters_cur_player_wins) / total_plays_wins
        plays_with_shutter_fraction_losses_real_last_turn = len(real_last_turn_shutters_cur_player_losses) / total_plays_losses


    # path = f"/home/lirontyomkin/AlphaZero_Gomoku/matches/statistics/vs {opponent_player.name}/{board_name}/"
    #
    # if not os.path.exists(f"{path}{cur_player.name}/"):
    #     os.makedirs(f"{path}{cur_player.name}/")
    #
    # outfile = open(f"{path}{cur_player.name}/full_{num_games}_games_stats", 'wb')
    # pickle.dump(games_history, outfile)
    # outfile.close()


    result =    [no_games,
                 no_wins,
                 no_losses,
                 no_ties,
                 win_ratio,

                 average_game_length,
                 average_game_length_wins,
                 average_game_length_losses,

                 avg_shutter_size,
                 avg_shutter_size_wins,
                 avg_shutter_size_losses,

                 plays_with_shutter_fraction,
                 plays_with_shutter_fraction_wins,
                 plays_with_shutter_fraction_losses,

                 avg_shutter_size_real_last_turn,
                 avg_shutter_size_wins_real_last_turn,
                 avg_shutter_size_losses_real_last_turn,

                 plays_with_shutter_fraction_real_last_turn,
                 plays_with_shutter_fraction_wins_real_last_turn,
                 plays_with_shutter_fraction_losses_real_last_turn]

    return result


def create_statistics_graphics(path, num_games, opponent_name):

    width = 30
    height = 8
    font_size = 27

    save_plays_with_shutter_results(path, num_games, width,height, font_size, opponent_name)
    save_shutter_size(path, num_games, width,height, font_size, opponent_name)
    save_save_game_len(path, num_games, width,height, font_size, opponent_name)
    save_game_results(path, num_games, width,height, font_size, opponent_name)


def save_plays_with_shutter_results(path, num_games, width, height, font_size, opponent_name):
    mpl.rcParams.update({'font.size': font_size})

    data = pd.read_excel(f"{path}all models {num_games} games results.xlsx", index_col=0)

    fig = plt.figure(constrained_layout=True)
    fig.suptitle(f"Fraction of plays with shutter results - {opponent_name}", y=1.05)

    fig.set_size_inches(width, height)
    grid = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[20, 1])

    ind = np.arange(len(data.index) - 1)
    width = 0.25

    ax = fig.add_subplot(grid[0, 0])

    ax.bar(ind-width, data["fraction of plays with shutter"][:-1], width=width, label="fraction of plays with shutter", color="blue")
    ax.bar(ind, data["fraction of plays with shutter (wins)"][:-1], width=width, label="fraction of plays with shutter - wins", color="green")
    ax.bar(ind + width, data["fraction of plays with shutter (losses)"][:-1], width=width, label="fraction of plays with shutter - losses", color="red")


    ax.bar(len(data.index)- 0.5 - 3*width, data["fraction of plays with shutter"][-1], width=width, color="blue")
    ax.bar(len(data.index)- 0.5 - 2*width, data["fraction of plays with shutter (real last turn)"][-1], label="real last turn", width=width, color="cornflowerblue")

    ax.bar(len(data.index)- 0.5 - 0.5*width, data["fraction of plays with shutter (wins)"][-1], width=width, color="green")
    ax.bar(len(data.index)- 0.5 + 0.5*width, data["fraction of plays with shutter (real last turn - wins)"][-1], label="real last turn", width=width, color="greenyellow")

    ax.bar(len(data.index)- 0.5 + 2*width, data["fraction of plays with shutter (losses)"][-1], width=width, color="red")
    ax.bar(len(data.index)- 0.5 + 3*width, data["fraction of plays with shutter (real last turn - losses)"][-1], label="real last turn", width=width, color="lightcoral")


    ax.set_xticks(list(np.arange(len(data.index) - 1)) + [len(data.index) - 0.5])
    ax.set_xticklabels(data.index)

    ax.set_ylim([0,1])

    lax = fig.add_subplot(grid[1, 0])
    h, l = ax.get_legend_handles_labels()
    lax.legend(h, l, borderaxespad=0, loc="center", fancybox=True, shadow=True, ncol=2)
    lax.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = PIL.Image.open(buf)

    plt.savefig(f"{path}Fraction of plays with shutter results.png", bbox_inches='tight')
    plt.close('all')


def save_shutter_size(path, num_games, width, height, font_size, opponent_name):
    mpl.rcParams.update({'font.size': font_size})

    data = pd.read_excel(f"{path}all models {num_games} games results.xlsx", index_col=0)

    fig = plt.figure(constrained_layout=True)
    fig.suptitle(f"Shutter sizes results  - {opponent_name}", y=1.05)

    fig.set_size_inches(width, height)
    grid = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[20, 1])

    ind = np.arange(len(data.index) - 1)
    width = 0.25

    ax = fig.add_subplot(grid[0, 0])

    ax.bar(ind-width, data["avg shutter size"][:-1], width=width, label="average shutter size", color="blue")
    ax.bar(ind, data["avg shutter size (wins)"][:-1], width=width, label="average shutter size - wins", color="green")
    ax.bar(ind + width, data["avg shutter size (losses)"][:-1], width=width, label="average shutter size - losses", color="red")


    ax.bar(len(data.index)- 0.5 - 3*width, data["avg shutter size"][-1], width=width, color="blue")
    ax.bar(len(data.index)- 0.5 - 2*width, data["avg shutter size (real last turn)"][-1], label="real last turn", width=width, color="cornflowerblue")

    ax.bar(len(data.index)- 0.5 - 0.5*width, data["avg shutter size (wins)"][-1], width=width, color="green")
    ax.bar(len(data.index)- 0.5 + 0.5*width, data["avg shutter size (real last turn - wins)"][-1], label="real last turn", width=width, color="greenyellow")

    ax.bar(len(data.index)- 0.5 + 2*width, data["avg shutter size (losses)"][-1], width=width, color="red")
    ax.bar(len(data.index)- 0.5 + 3*width, data["avg shutter size (real last turn - losses)"][-1], label="real last turn", width=width, color="lightcoral")


    ax.set_xticks(list(np.arange(len(data.index) - 1)) + [len(data.index) - 0.5])
    ax.set_xticklabels(data.index)

    lax = fig.add_subplot(grid[1, 0])
    h, l = ax.get_legend_handles_labels()
    lax.legend(h, l, borderaxespad=0, loc="center", fancybox=True, shadow=True, ncol=2)
    lax.axis("off")


    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = PIL.Image.open(buf)

    plt.savefig(f"{path}Shutter sizes results.png", bbox_inches='tight')
    plt.close('all')


def save_save_game_len(path, num_games, width, height, font_size, opponent_name):
    mpl.rcParams.update({'font.size': font_size})

    data = pd.read_excel(f"{path}all models {num_games} games results.xlsx", index_col=0)

    fig = plt.figure(constrained_layout=True)
    fig.suptitle(f"Games length results - {opponent_name}",  y=1.05)

    fig.set_size_inches(width, height)
    grid = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[20, 1])

    ind = np.arange(len(data.index))
    width = 0.25

    ax = fig.add_subplot(grid[0, 0])

    ax.bar(ind-width, data["avg game len"], width=width, label="average game len", color="blue")
    ax.bar(ind, data["avg game len (wins)"], width=width, label="average game len - wins", color="green")
    ax.bar(ind + width, data["avg game len (losses)"], width=width, label="average game len - losses", color="red")


    ax.set_xticks(ind)
    ax.set_xticklabels(data.index)

    lax = fig.add_subplot(grid[1, 0])
    h, l = ax.get_legend_handles_labels()
    lax.legend(h, l, borderaxespad=0, loc="center", fancybox=True, shadow=True)
    lax.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = PIL.Image.open(buf)

    plt.savefig(f"{path}Games lengths results.png", bbox_inches='tight')
    plt.close('all')


def save_game_results(path, num_games, width, height, font_size, opponent_name):
    mpl.rcParams.update({'font.size': font_size})

    data = pd.read_excel(f"{path}all models {num_games} games results.xlsx", index_col=0)

    fig = plt.figure(constrained_layout=True)
    fig.suptitle(f"Games results - {opponent_name}", y=1.05)

    fig.set_size_inches(width, height)
    grid = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[20,1])

    ind = np.arange(len(data.index))
    width = 0.5

    ax = fig.add_subplot(grid[0, 0])

    ax.set_ylim([0,num_games*1.1])

    ax.bar(ind, data["no. wins"], width=width, label="wins", color="green")
    ax.bar(ind, data["no. losses"], width=width, label="losses", color="red", bottom=data["no. wins"])
    ax.bar(ind, data["no. ties"], width=width, label = "ties", color="yellow", bottom=data["no. wins"]+data["no. losses"])

    ax.set_xticks(ind)
    ax.set_xticklabels(data.index)

    lax = fig.add_subplot(grid[1, 0])
    h, l = ax.get_legend_handles_labels()
    lax.legend(h,l, borderaxespad=0, loc="center", fancybox=True, shadow=True)
    lax.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = PIL.Image.open(buf)

    plt.savefig(f"{path}Games results.png", bbox_inches='tight')
    plt.close('all')


def call_collage_statistics_results(board_name, opponents):

    opponents_names = [opp.name for opp in opponents]

    path_collage = f"/home/lirontyomkin/AlphaZero_Gomoku/matches/statistics/{board_name} summery/"

    if not os.path.exists(path_collage):
        os.makedirs(path_collage)

    listofimages1 = [f"/home/lirontyomkin/AlphaZero_Gomoku/matches/statistics/vs {opponent_name}/{board_name}/Fraction of plays with shutter results.png"
                    for opponent_name in opponents_names]

    listofimages2 = [
        f"/home/lirontyomkin/AlphaZero_Gomoku/matches/statistics/vs {opponent_name}/{board_name}/Games lengths results.png"
        for opponent_name in opponents_names]

    listofimages3 = [
        f"/home/lirontyomkin/AlphaZero_Gomoku/matches/statistics/vs {opponent_name}/{board_name}/Games results.png"
        for opponent_name in opponents_names]

    listofimages4 = [
        f"/home/lirontyomkin/AlphaZero_Gomoku/matches/statistics/vs {opponent_name}/{board_name}/Shutter sizes results.png"
        for opponent_name in opponents_names]


    create_collages_boards(listofimages1, "Fraction of plays with shutter results", path_collage)
    create_collages_boards(listofimages2, "Games lengths results", path_collage)
    create_collages_boards(listofimages3, "Games results", path_collage)
    create_collages_boards(listofimages4, "Shutter sizes results", path_collage)


def create_collages_boards(listofimages, fig_name, path):

    im_check = PIL.Image.open(listofimages[0])
    width1, height1 = im_check.size

    cols = 1
    rows = 4

    width = width1 * cols
    height = height1 * rows

    thumbnail_width = width // cols
    thumbnail_height = height // rows

    size = thumbnail_width, thumbnail_height
    new_im = PIL.Image.new('RGB', (width, height))
    ims = []
    for p in listofimages:
        im = PIL.Image.open(p)
        im.thumbnail(size)
        ims.append(im)
    i = 0
    x = 0
    y = 0
    for col in range(cols):
        for row in range(rows):
            # print(i, x, y)
            new_im.paste(ims[i], (x, y))
            i += 1
            y += thumbnail_height
        x += thumbnail_width
        y = 0

    new_im.save(path + f"{fig_name}.png")



if __name__ == '__main__':
    n_playout = 400

    v7 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p3_v7/current_policy_1500.model',
          'v7_1500', 3, True, False)
    v9 = ( '/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p3_v9/current_policy_1500.model',
           'v9_1500', 3,True, False)
    v10 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v10/current_policy_1500.model',
           'v10_1500', 4, True, False)
    v10_random = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v10/current_policy_1500.model',
           'v10_1500_random', 4, True, True)


    policy_v7 = PolicyValueNet(6, 6, model_file=v7[0], input_plains_num=v7[2])
    player_v7 = MCTSPlayer(policy_v7.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v7[3],
                           name=v7[1], input_plains_num=v7[2], is_random_last_turn=v7[4])
    policy_v9 = PolicyValueNet(6, 6, model_file=v9[0], input_plains_num=v9[2])
    player_v9 = MCTSPlayer(policy_v9.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v9[3],
                               name=v9[1], input_plains_num=v9[2], is_random_last_turn=v9[4])
    policy_v10 = PolicyValueNet(6, 6, model_file=v10[0], input_plains_num=v10[2])
    player_v10 = MCTSPlayer(policy_v10.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v10[3],
                               name=v10[1], input_plains_num=v10[2], is_random_last_turn=v10[4])
    policy_v10_random = PolicyValueNet(6, 6, model_file=v10_random[0], input_plains_num=v10_random[2])
    player_v10_random = MCTSPlayer(policy_v10_random.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v10_random[3],
                               name=v10_random[1], input_plains_num=v10_random[2], is_random_last_turn=v10_random[4])


    v12 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v12/current_policy_5000.model',
           'v12_5000', 4, True, False)
    v14 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v14/current_policy_5000.model',
           'v14_5000', 4, True, False)
    v16 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v16/current_policy_5000.model',
           'v16_5000', 4, True, False)
    v18 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v18/current_policy_5000.model',
           'v18_5000', 4, True, False)
    v20 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v20/current_policy_5000.model',
           'v20_5000', 4, True, False)
    v22 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v22/current_policy_5000.model',
           'v22_5000', 4, True, False)

    policy_v12 = PolicyValueNet(6, 6, model_file=v12[0], input_plains_num=v12[2])
    player_v12 = MCTSPlayer(policy_v12.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v12[3],
                            name=v12[1], input_plains_num=v12[2], is_random_last_turn=v12[4])

    policy_v14 = PolicyValueNet(6, 6, model_file=v14[0], input_plains_num=v14[2])
    player_v14 = MCTSPlayer(policy_v14.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v14[3],
                            name=v14[1], input_plains_num=v14[2], is_random_last_turn=v14[4])

    policy_v16 = PolicyValueNet(6, 6, model_file=v16[0], input_plains_num=v16[2])
    player_v16 = MCTSPlayer(policy_v16.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v16[3],
                            name=v16[1], input_plains_num=v16[2], is_random_last_turn=v16[4])

    policy_v18 = PolicyValueNet(6, 6, model_file=v18[0], input_plains_num=v18[2])
    player_v18 = MCTSPlayer(policy_v18.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v18[3],
                            name=v18[1], input_plains_num=v18[2], is_random_last_turn=v18[4])

    policy_v20 = PolicyValueNet(6, 6, model_file=v20[0], input_plains_num=v20[2])
    player_v20 = MCTSPlayer(policy_v20.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v20[3],
                            name=v20[1], input_plains_num=v20[2], is_random_last_turn=v20[4])

    policy_v22 = PolicyValueNet(6, 6, model_file=v22[0], input_plains_num=v22[2])
    player_v22 = MCTSPlayer(policy_v22.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v22[3],
                            name=v22[1], input_plains_num=v22[2], is_random_last_turn=v22[4])


    players_list = [player_v7, player_v9, player_v10, player_v12, player_v14,
                    player_v16, player_v18, player_v20, player_v22, player_v10_random]


    opponent_player_1 = Heuristic_player(name="forcing heuristic", heuristic="interaction with forcing")
    opponent_player_2 = PUREMCTS(c_puct=5, n_playout=500, name="pure MCTS 500")


    v9_5000 = ( '/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p3_v9/current_policy_5000.model',
           'v9_5000_no_MCTS', 3, True, False)
    policy_opponent_3 = PolicyValueNet(6, 6, model_file=v9_5000[0], input_plains_num=v9_5000[2])
    opponent_player_3 = MCTSPlayer(policy_opponent_3.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v9_5000[3],
                               name=v9_5000[1], input_plains_num=v9_5000[2], is_random_last_turn=v9_5000[4])


    v10_5000 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v10/current_policy_5000.model',
           f'v10_5000_no_MCTS', 4, True, False)
    policy_opponent_4 = PolicyValueNet(6, 6, model_file=v10_5000[0], input_plains_num=v10_5000[2])
    opponent_player_4 = MCTSPlayer(policy_opponent_4.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v10_5000[3],
                               name=v10_5000[1], input_plains_num=v10_5000[2], is_random_last_turn=v10_5000[4])


    opponents = [opponent_player_1, opponent_player_2, opponent_player_3, opponent_player_4]


    set_start_method("spawn")
    compare_all_models_statistics(players_list, opponents, width=6, height=6, n=4, num_games=1000)


    BOARDS = [EMPTY_BOARD, BOARD_1_FULL, BOARD_2_FULL, BOARD_1_TRUNCATED, BOARD_2_TRUNCATED]
    for board in BOARDS:
        board_name = board[1]
        call_collage_statistics_results(board_name, opponents)