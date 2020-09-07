



import numpy as np
from game import Board



def graphic(board, player1, player2):
    """Draw the board and show game info"""
    width = board.width
    height = board.height

    print("Player", player1, "with X".rjust(3))
    print("Player", player2, "with O".rjust(3))
    print()
    for x in range(width):
        print("{0:8}".format(x), end='')
    print('\r\n')
    for i in range(height - 1, -1, -1):
        print("{0:4d}".format(i), end='')
        for j in range(width):
            loc = i * width + j
            p = board.states.get(loc, -1)
            if p == player1:
                print('X'.center(8), end='')
            elif p == player2:
                print('O'.center(8), end='')
            else:
                print('_'.center(8), end='')
        print('\r\n\r\n')


n = 4
width, height = 6,6

# initial_board = np.array([[0,0,0,0,0,0],
#                           [0,2,0,0,0,0],
#                           [0,0,0,0,0,0],
#                           [0,0,0,0,0,0],
#                           [2,1,0,0,2,0],
#                           [0,0,1,1,1,2]])

initial_board = np.array([
    [0, 1, 0, 2, 0, 0],
    [0, 2, 1, 1, 0, 0],
    [1, 2, 2, 2, 1, 0],
    [2, 0, 1, 1, 2, 0],
    [1, 0, 2, 2, 2, 0],
    [0, 0, 0, 0, 1, 0]])


# initial_board = np.array([[0,0,0],
#                           [0,1,0],
#                           [0,0,2]])
#


i_board = np.zeros((2, height, width))
i_board[0] = initial_board == 1
i_board[1] = initial_board == 2

board = Board(width=width, height=height, n_in_row=n)

board.init_board(start_player=1, initial_state=i_board)

# graphic(board, 1, 2)

print(initial_board)

print("Current player is {}".format(board.current_player))

print(board.calc_all_heuristics(exp=1, o_weight=1))