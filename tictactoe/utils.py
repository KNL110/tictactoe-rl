import pickle


def canonical_state(board, player):
    """View the board from `player`'s perspective: player's own pieces become 1,
    opponent's become -1. This lets a single table serve both X and O in self-play."""
    return tuple(player * c for c in board)


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
