# from multiprocessing import Pool
# import pandas as pd
# from Game_boards_and_aux import *
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import os
# import matplotlib.pyplot as plt
# import io
# import PIL
#
#
#
# def EMD_model_comparison(model1_name, input_plains_num_1, max_model1_iter, model1_check_freq, tell_last_move1,
#                          model2_name, input_plains_num_2, max_model2_iter, model2_check_freq, tell_last_move2,
#                          board, n=4, width=6, height=6, **kwargs):
#
#     BOARDS = [BOARD_1_FULL, BOARD_2_FULL, BOARD_1_TRUNCATED, BOARD_2_TRUNCATED, EMPTY_BOARD_6X6]
#
#
#     last_move_str_1 = " with last move " if tell_last_move1 else " "
#     last_move_str_2 = " with last move" if tell_last_move2 else ""
#
#     save_path = f'/home/lirontyomkin/AlphaZero_Gomoku/models emd comparison/{model1_name}{last_move_str_1}and {model2_name}{last_move_str_2}/'
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#
#
#     sub_models_1 = list(range(model1_check_freq, max_model1_iter + model1_check_freq, model1_check_freq))
#     sub_models_2 = list(range(model2_check_freq, max_model2_iter + model2_check_freq, model2_check_freq))
#
#
#     index = [f"{model1_name}__{i}" for i in sub_models_1]
#     columns = [f"{model2_name}__{i}" for i in sub_models_2]
#
#     # result = np.random.rand(len(index), len(columns))
#     result = np.empty((len(index), len(columns)))
#
#     board_state, board_name, last_move_p1, last_move_p2, _, _ = board
#
#     if tell_last_move1:
#         board1 = initialize_board_with_init_and_last_moves(height, width, input_board=board_state, n_in_row=n, last_move_p1=last_move_p1,
#                                                            last_move_p2=last_move_p2)
#     else:
#         board1 = initialize_board_with_init_and_last_moves(height, width, input_board=board_state, n_in_row=n, last_move_p1=None,
#                                                            last_move_p2=None)
#
#     if tell_last_move2:
#         board2 = initialize_board_with_init_and_last_moves(height, width, input_board=board_state, n_in_row=n, last_move_p1=last_move_p1,
#                                                            last_move_p2=last_move_p2)
#     else:
#         board2 = initialize_board_with_init_and_last_moves(height, width, input_board=board_state, n_in_row=n, last_move_p1=None,
#                                                            last_move_p2=None)
#
#     for index_i, i in enumerate(range(model1_check_freq, max_model1_iter + model1_check_freq, model1_check_freq)):
#         for index_j, j in enumerate(range(model2_check_freq, max_model2_iter + model2_check_freq, model2_check_freq)):
#             result[index_i, index_j] = EMD_between_two_models_on_board(
#                                    model1_name=model1_name, input_plains_num_1=input_plains_num_1, i1=i,
#                                    model2_name=model2_name, input_plains_num_2=input_plains_num_2, i2=j,
#                                    board1=board1, board2=board2, width=width,height=height)
#
#
#     Emd_upper_bound = 0.001
#     if np.max(result) > Emd_upper_bound:
#         raise Exception("Enlarge upper boundary in EMD colorbar")
#
#     df = pd.DataFrame(result, index=index, columns=columns)
#     df.to_csv(f"{save_path}on {board_name}.csv",index = True, header=True)
#
#     fig, ax = plt.subplots(1, figsize=(20, 20))
#     fontsize = 12
#
#     im = ax.imshow(result, cmap='hot', interpolation='nearest', norm=plt.Normalize(vmin=0, vmax=Emd_upper_bound))
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#
#     sm = plt.cm.ScalarMappable(cmap='hot', norm=plt.Normalize(vmin=0, vmax=Emd_upper_bound))
#     fig.colorbar(sm, ax=ax, cax=cax).ax.tick_params(labelsize=fontsize*2)
#
#     ax.set_title(f"EMD of {model1_name}{last_move_str_1}and {model2_name}{last_move_str_2}\non {board_name}", fontsize=3*fontsize)
#     ax.set_xticks(list(range(len(sub_models_1))))
#     ax.set_yticks(list(range(len(sub_models_1))))
#
#     ax.set_xticklabels([str(i) for i in sub_models_1], rotation=90, fontsize=fontsize)
#     ax.set_yticklabels([str(i) for i in sub_models_2], fontsize=fontsize)
#
#     ax.set_xlabel(model1_name, fontsize=fontsize*2.5)
#     ax.set_ylabel(model2_name, rotation=90, fontsize=fontsize*2.5)
#
#     buf = io.BytesIO()
#     plt.savefig(buf, format='jpeg')
#     buf.seek(0)
#     image = PIL.Image.open(buf)
#
#     plt.savefig(f"{save_path}on {board_name}.png")
#
#     print(f"Done {model1_name} and {model2_name} on {board_name}")
#
#     plt.close('all')
#
#
#
# def generate_models_emd_comparison():
#
#     BOARDS = [BOARD_1_FULL, BOARD_2_FULL, BOARD_1_TRUNCATED, BOARD_2_TRUNCATED, EMPTY_BOARD_6X6]
#
#     arguments = []
#
#     for board in BOARDS:
#         arguments.append(("pt_6_6_4_p4_v10", 4, 5000, 50, True, "pt_6_6_4_p3_v7", 3, 5000, 50, False, board, 4, 6, 6))
#
#         # EMD_model_comparison(model1_name="pt_6_6_4_p4_v10", input_plains_num_1=4, max_model1_iter=5000,
#         #                      model1_check_freq=50, tell_last_move1=True,
#         #                      model2_name="pt_6_6_4_p3_v7", input_plains_num_2=3, max_model2_iter=5000,
#         #                      model2_check_freq=50, tell_last_move2=False,
#         #                      game_board=board, n=4, width=6, height=6)
#
#         arguments.append(("pt_6_6_4_p4_v10", 4, 5000, 50, True, "pt_6_6_4_p3_v9", 3, 5000, 50, False, board, 4, 6, 6))
#
#         # EMD_model_comparison(model1_name="pt_6_6_4_p4_v10", input_plains_num_1=4, max_model1_iter=5000,
#         #                      model1_check_freq=50, tell_last_move1=True,
#         #                      model2_name="pt_6_6_4_p3_v9", input_plains_num_2=3, max_model2_iter=5000,
#         #                      model2_check_freq=50, tell_last_move2=False,
#         #                      game_board=board, n=4, width=6, height=6)
#
#         arguments.append(("pt_6_6_4_p3_v7", 3, 5000, 50, False, "pt_6_6_4_p3_v9", 3, 5000, 50, False, board, 4, 6, 6))
#
#         # EMD_model_comparison(model1_name="pt_6_6_4_p3_v7", input_plains_num_1=3, max_model1_iter=5000,
#         #                      model1_check_freq=50, tell_last_move1=False,
#         #                      model2_name="pt_6_6_4_p3_v9", input_plains_num_2=3, max_model2_iter=5000,
#         #                      model2_check_freq=50, tell_last_move2=False,
#         #                      game_board=board, n=4, width=6, height=6)
#
#         arguments.append(("pt_6_6_4_p4_v10", 4, 5000, 50, False, "pt_6_6_4_p3_v7", 3, 5000, 50, False, board, 4, 6, 6))
#
#         # EMD_model_comparison(model1_name="pt_6_6_4_p4_v10", input_plains_num_1=4, max_model1_iter=5000,
#         #                      model1_check_freq=50, tell_last_move1=False,
#         #                      model2_name="pt_6_6_4_p3_v7", input_plains_num_2=3, max_model2_iter=5000,
#         #                      model2_check_freq=50, tell_last_move2=False,
#         #                      game_board=board, n=4, width=6, height=6)
#
#         arguments.append(("pt_6_6_4_p4_v10", 4, 5000, 50, False, "pt_6_6_4_p3_v9", 3, 5000, 50, False, board, 4, 6, 6))
#
#         # EMD_model_comparison(model1_name="pt_6_6_4_p4_v10", input_plains_num_1=4, max_model1_iter=5000,
#         #                      model1_check_freq=50, tell_last_move1=False,
#         #                      model2_name="pt_6_6_4_p3_v9", input_plains_num_2=3, max_model2_iter=5000,
#         #                      model2_check_freq=50, tell_last_move2=False,
#         #                      game_board=board, n=4, width=6, height=6)
#
#
#     with Pool(5) as pool:
#         print(f"Using {pool._processes} workers. There are {len(arguments)} jobs: \n")
#         pool.starmap(EMD_model_comparison, arguments)
#         pool.close()
#
#
#
# if __name__ == "__main__":
#     generate_models_emd_comparison()
