import numpy as np
from collections import Counter

def dbg_list(list, info):
    print(f'{info} len: {len(list)}, counts: {Counter(tuple(list))}')


def print_board(state, move):
    # state is 9 * 5 * 5
    player_id = 1 if state[0] == 0 else 2

    cnt = 0
    move_cnt = 0
    dst = [0] * 25
    for y in range(5):
        for x in range(5):
            for plane in range(8):
                idx = (y * 5 + x) + ((plane + 1) * 25) # plane 0 is player
                idx_move = idx - 25 # no plane 0
                if state[idx] > 0:
                    dst[y * 5 + x] = 'TIXYtixy'[plane]
                    cnt += 1

                if move[idx_move] > 0:
                    dst[y * 5 + x] = '*'
                    move_cnt += 1

    print(f'count: {cnt}, move count: {move_cnt}, player: {player_id}')
    print(np.array(dst).reshape(5, 5))
