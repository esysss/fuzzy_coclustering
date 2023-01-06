import numpy as np

def merge1(Pusers, Pitems, k):
    Pik = Pusers[k]
    user_keeper = Pusers.copy()
    Pusers[Pusers > 0] = 1
    Pitems[Pitems > 0] = 1

    temp = Pusers + Pitems
    temp = np.where(temp == 2)[0]
    user_keeper = user_keeper[temp]

    if len(user_keeper) == 0:
        return 0

    if Pik == max(user_keeper):
        return 1
    else :
        return 0


def merge2(Pusers, k):
    return Pusers[k]


def merge3(Pusers, Pitems, k):
    # return (Pusers[k] * Pitems[k]) / np.sum(Pusers * Pitems)
    return (Pusers[k] + Pitems[k]) / np.sum(Pusers + Pitems)