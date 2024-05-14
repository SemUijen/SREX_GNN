
# TODO: very slow option, but works for now
def best_to_srexParameters(best_idx, label_shape):
    Max_to_move, numR_P1, numR_P2 = label_shape
    i = 0
    for numRoutesMove in range(1, Max_to_move + 1):
        for idx1 in range(0, numR_P1):
            for idx2 in range(0, numR_P2):
                print(i)
                if i == best_idx:
                    return numRoutesMove, idx1, idx2
                i += 1

    return "nothing found"