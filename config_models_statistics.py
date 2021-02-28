from __future__ import print_function
from mcts_alphaZero import *
from heuristic_player import Heuristic_player
from mcts_pure import MCTSPlayer as PUREMCTS



discription_dict = {
    "v7_1500" : "v7_1500",
    "v9_1500" : "v9_1500",
    "v10_1500": "v10_1500",
    "v10_1500_random" : "v10_1500_random",

    "v9_5000" : "v9_5000",
    "v10_5000": "v10_5000",


    "v23_5000": "v23\nsim:50\nshutter:1\nfull:yes\niter:5000",
    "v24_5000": "v24\nsim:50\nshutter:0\nfull:yes\niter:5000",
    "v25_5000": "v25\nsim:50\nshutter:1\nfull:no\niter:5000",
    "v26_5000": "v26\nsim:50\nshutter:0\nfull:no\niter:5000",
    "v27_5000": "v27\nsim:25\nshutter:1\nfull:yes\niter:5000",
    "v28_5000": "v28\nsim:25\nshutter:0\nfull:yes\niter:5000",
    "v29_5000": "v29\nsim:25\nshutter:1\nfull:no\niter:5000",
    "v30_5000": "v30\nsim:25\nshutter:0\nfull:no\niter:5000",
    "v31_5000": "v31\nsim:100\nshutter:1\nfull:yes\niter:5000",
    "v32_5000": "v32\nsim:100\nshutter:0\nfull:yes\niter:5000",
    "v33_5000": "v33\nsim:100\nshutter:1\nfull:no\niter:5000",
    "v34_5000": "v34\nsim:100\nshutter:0\nfull:no\niter:5000",


    "v_01_1500": "v_01\nsim:400\n3 plains\niter:1500",
    "v_02_1500": "v_02\nsim:400\n4 plains\niter:1500",
    "v_02_1500_random": "v_02_random:\nsim:400\n4 plains\niter:1500",

    "v_03_5000": "v_03\nsim:100\nshutter:2\nfull:yes\niter:5000",
    "v_04_5000": "v_04\nsim:100\nshutter:1\nfull:yes\niter:5000",
    "v_05_5000": "v_05\nsim:100\nshutter:0\nfull:yes\niter:5000",

    "v_06_5000": "v_06\nsim:100\nshutter:2\nfull:no\niter:5000",
    "v_07_5000": "v_07\nsim:100\nshutter:1\nfull:no\niter:5000",
    "v_08_5000": "v_08\nsim:100\nshutter:0\nfull:no\niter:5000",

}

colors_dict = {
    "v7_1500" : "grey",
    "v9_1500" : "grey",
    "v10_1500": "grey",
    "v10_1500_random" : "grey",
    "v9_5000": "grey",
    "v10_5000": "grey",

    "v23_5000": "green",
    "v24_5000": "green",
    "v25_5000": "green",
    "v26_5000": "green",
    "v27_5000": "blue",
    "v28_5000": "blue",
    "v29_5000": "blue",
    "v30_5000": "blue",
    "v31_5000": "red",
    "v32_5000": "red",
    "v33_5000": "red",
    "v34_5000": "red",


    "v_01_1500": "grey",
    "v_02_1500": "grey",
    "v_02_1500_random": "grey",

    "v_03_5000": "green",
    "v_04_5000": "green",
    "v_05_5000": "green",

    "v_06_5000": "orange",
    "v_07_5000": "orange",
    "v_08_5000": "orange",

}

# veteran_models_6 = (["v7_1500", "v9_1500", "v10_1500", "v10_1500_random"], "veteran_models")
veteran_models_6 = (["v9_1500", "v10_1500", "v9_5000", "v10_5000"], "veteran_models")

all_new_12_models_6 = (["v27_5000", "v28_5000", "v29_5000", "v30_5000", "v23_5000", "v24_5000", "v25_5000", "v26_5000", "v31_5000", "v32_5000", "v33_5000", "v34_5000"], "all_new_12_models", False)

mcts_25_models_6 = (["v27_5000", "v28_5000", "v29_5000", "v30_5000"], "mcts_25_models", False)
mcts_50_models_6 = (["v23_5000", "v24_5000", "v25_5000", "v26_5000"], "mcts_50_models", False)
mcts_100_models_6 = (["v31_5000", "v32_5000", "v33_5000", "v34_5000"], "mcts_100_models", False)

full_boards_models_6 = (["v27_5000", "v28_5000", "v23_5000", "v24_5000", "v31_5000", "v32_5000"], "full_boards_models", True)
non_full_boards_models_6 = (["v29_5000", "v30_5000", "v25_5000", "v26_5000", "v33_5000", "v34_5000"], "non_full_boards_models", True)

shutter_0_models_6 = (["v28_5000", "v30_5000", "v24_5000", "v26_5000", "v32_5000", "v34_5000"], "shutter_0_models", True)
shutter_1_models_6 = (["v27_5000", "v29_5000", "v23_5000", "v25_5000", "v31_5000", "v33_5000"], "shutter_1_models", True)


models_variations_6 = [all_new_12_models_6, mcts_25_models_6, mcts_50_models_6, mcts_100_models_6, full_boards_models_6,
                       non_full_boards_models_6, shutter_0_models_6, shutter_1_models_6]



# veteran_models_10 = (["v_01_1500", "v_02_1500", "v_02_1500_random"], "veteran_models")
veteran_models_10 = (["v_01_1500", "v_02_1500"], "veteran_models")


all_new_6_models_10 = (["v_03_5000", "v_04_5000", "v_05_5000", "v_06_5000", "v_07_5000", "v_08_5000"], "all_new_6_models", False)

full_boards_models_10 = (["v_03_5000", "v_04_5000", "v_05_5000"], "full_boards_models", True)
non_full_boards_models_10 = (["v_06_5000", "v_07_5000", "v_08_5000"], "non_full_boards_models", True)

shutter_0_models_10 = (["v_05_5000", "v_08_5000"], "shutter_0_models", True)
shutter_1_models_10 = (["v_04_5000", "v_07_5000" ], "shutter_1_models", True)
shutter_2_models_10 = (["v_03_5000", "v_06_5000"], "shutter_2_models", True)


models_variations_10 = [all_new_6_models_10, full_boards_models_10, non_full_boards_models_10,
                       shutter_0_models_10, shutter_1_models_10, shutter_2_models_10]


all_models_variations = {6: models_variations_6, 10 : models_variations_10}
veteran_models = {6: veteran_models_6, 10 : veteran_models_10}



def who_started_dict(board_name):
    X_boards = [
        "board 3 full",
        "board 5 full",
        "board 3 truncated",
        "board 5 truncated",
        "empty board"
    ]

    O_boards = [
        "board 1 full",
        "board 2 full",
        "board 1 truncated",
        "board 2 truncated",
        "board 4 full",
        "board 4 truncated",
    ]

    if board_name in X_boards:
        return 1
    elif board_name in O_boards:
        return 2
    else:
        raise Exception(f"{board_name} is not a valid board")




n_playout = 400

v7 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p3_v7/current_policy_1500.model',
      'v7_1500', 3, True, False)
v9 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p3_v9/current_policy_1500.model',
      'v9_1500', 3, True, False)
v10 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v10/current_policy_1500.model',
       'v10_1500', 4, True, False)
v10_random = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v10/current_policy_1500.model',
              'v10_1500_random', 4, True, True)


v9_5000 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p3_v9/current_policy_5000.model',
      'v9_5000', 3, True, False)
v10_5000 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v10/current_policy_5000.model',
       'v10_5000', 4, True, False)




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

v23 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v23/current_policy_5000.model',
       'v23_5000', 4, True, False, 1)
v24 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v24/current_policy_5000.model',
       'v24_5000', 4, True, False, 0)
v25 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v25/current_policy_5000.model',
       'v25_5000', 4, True, False, 1)
v26 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v26/current_policy_5000.model',
       'v26_5000', 4, True, False, 0)
v27 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v27/current_policy_5000.model',
       'v27_5000', 4, True, False, 1)
v28 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v28/current_policy_5000.model',
       'v28_5000', 4, True, False, 0)
v29 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v29/current_policy_5000.model',
       'v29_5000', 4, True, False, 1)
v30 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v30/current_policy_5000.model',
       'v30_5000', 4, True, False, 0)
v31 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v31/current_policy_5000.model',
       'v31_5000', 4, True, False, 1)
v32 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v32/current_policy_5000.model',
       'v32_5000', 4, True, False, 0)
v33 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v33/current_policy_5000.model',
       'v33_5000', 4, True, False, 1)
v34 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v34/current_policy_5000.model',
       'v34_5000', 4, True, False, 0)


all_new_12_models_6_policies = [v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34]





opponent_player_forcing_6X6 = Heuristic_player(name="forcing heuristic", heuristic="interaction with forcing")
opponent_player_mcts_500_6X6 = PUREMCTS(c_puct=5, n_playout=500, name="pure MCTS 500")
opponent_player_mcts_1000_6X6 = PUREMCTS(c_puct=5, n_playout=1000, name="pure MCTS 1000")

v9_5000_opp = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p3_v9/current_policy_5000.model',
           'v9_5000_no_MCTS', 3, True, False)
policy_opponent_4_6X6 = PolicyValueNet(6, 6, model_file=v9_5000_opp[0], input_plains_num=v9_5000_opp[2])
opponent_player_v9_5000_6X6 = MCTSPlayer(policy_opponent_4_6X6.policy_value_fn, c_puct=5, n_playout=n_playout,
                                     no_playouts=v9_5000_opp[3],
                                     name=v9_5000_opp[1], input_plains_num=v9_5000_opp[2], is_random_last_turn=v9_5000_opp[4])

v10_5000_opp = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v10/current_policy_5000.model',
            f'v10_5000_no_MCTS', 4, True, False)
policy_opponent_5_6X6 = PolicyValueNet(6, 6, model_file=v10_5000_opp[0], input_plains_num=v10_5000_opp[2])
opponent_player_v10_5000_6X6 = MCTSPlayer(policy_opponent_5_6X6.policy_value_fn, c_puct=5, n_playout=n_playout,
                                      no_playouts=v10_5000_opp[3],
                                      name=v10_5000_opp[1], input_plains_num=v10_5000_opp[2], is_random_last_turn=v10_5000_opp[4])












v_01 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_10_10_5_p3_v0_1/current_policy_1500.model',
        'v_01_1500', 3, True, False)
v_02 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_10_10_5_p4_v0_2/current_policy_1500.model',
        'v_02_1500', 4, True, False)
v_02_random = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_10_10_5_p4_v0_2/current_policy_1500.model',
               'v_02_1500_random', 4, True, True)

v_03 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_10_10_5_p4_v0_3/current_policy_5000.model',
        'v_03_5000', 4, True, False)
v_04 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_10_10_5_p4_v0_4/current_policy_5000.model',
        'v_04_5000', 4, True, False)
v_05 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_10_10_5_p4_v0_5/current_policy_5000.model',
        'v_05_5000', 4, True, False)
v_06 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_10_10_5_p4_v0_6/current_policy_5000.model',
        'v_06_5000', 4, True, False)
v_07 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_10_10_5_p4_v0_7/current_policy_5000.model',
        'v_07_5000', 4, True, False)
v_08 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_10_10_5_p4_v0_8/current_policy_5000.model',
        'v_08_5000', 4, True, False)




opponent_player_forcing_10X10 = Heuristic_player(name="forcing heuristic", heuristic="interaction with forcing")
opponent_player_mcts_500_10X10 = PUREMCTS(c_puct=5, n_playout=500, name="pure MCTS 500")
opponent_player_3_mcts_1000_10X10 = PUREMCTS(c_puct=5, n_playout=1000, name="pure MCTS 1000")

v_01_5000 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_10_10_5_p3_v0_1/current_policy_5000.model',
             'v_01_5000_no_MCTS', 3, True, False)
policy_opponent_4_10X10 = PolicyValueNet(10, 10, model_file=v_01_5000[0], input_plains_num=v_01_5000[2])
opponent_player_v_01_5000_10X10 = MCTSPlayer(policy_opponent_4_10X10.policy_value_fn, c_puct=5, n_playout=n_playout,
                                       no_playouts=v_01_5000[3],
                                       name=v_01_5000[1], input_plains_num=v_01_5000[2],
                                       is_random_last_turn=v_01_5000[4])

v_02_5000 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_10_10_5_p4_v0_2/current_policy_5000.model',
             f'v_02_5000_no_MCTS', 4, True, False)
policy_opponent_5_10X10 = PolicyValueNet(10, 10, model_file=v_02_5000[0], input_plains_num=v_02_5000[2])
opponent_player_v_02_5000_10X10 = MCTSPlayer(policy_opponent_5_10X10.policy_value_fn, c_puct=5, n_playout=n_playout,
                                       no_playouts=v_02_5000[3],
                                       name=v_02_5000[1], input_plains_num=v_02_5000[2],
                                       is_random_last_turn=v_02_5000[4])


