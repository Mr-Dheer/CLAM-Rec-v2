"""
Leave-one-out data partition (faithful to A-LLMRec) plus cold/warm tagging for RQ1.

Partition (per user, chronological):
  - test  = last item
  - valid = second-to-last item
  - train = everything before
  Users with < 3 interactions keep all in train and have empty valid/test.

Cold/warm tagging (RQ1):
  Each *test case* is tagged by the popularity of its TARGET (test) item, measured
  as that item's interaction count in the TRAINING portion only (no leakage from
  valid/test). An item with train-count <= cold_threshold is "cold". This lets us
  slice Hit@1/NDCG by whether the ground-truth item is cold or warm.
"""

from collections import Counter, defaultdict


def data_partition(path):
    """Return [user_train, user_valid, user_test, usernum, itemnum]."""
    usernum = itemnum = 0
    User = defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            u, i = line.rstrip().split(" ")
            u, i = int(u), int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)

    user_train, user_valid, user_test = {}, {}, {}
    for user in User:
        n = len(User[user])
        if n < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = [User[user][-2]]
            user_test[user] = [User[user][-1]]
    return [user_train, user_valid, user_test, usernum, itemnum]


def item_train_counts(user_train):
    """Interaction count per item over the TRAINING portion only."""
    c = Counter()
    for u, seq in user_train.items():
        for i in seq:
            c[i] += 1
    return c


def tag_cold_warm(user_test, user_train, cold_threshold=5):
    """Map user_id -> {'target': item_id, 'train_count': int, 'cold': bool}.

    Only users with a non-empty test item are included.
    """
    counts = item_train_counts(user_train)
    tags = {}
    for u, test_items in user_test.items():
        if not test_items:
            continue
        target = test_items[0]
        cnt = counts.get(target, 0)
        tags[u] = {"target": target, "train_count": cnt, "cold": cnt <= cold_threshold}
    return tags, counts


def cold_warm_summary(tags):
    """Quick distribution summary for logging/reporting."""
    n = len(tags)
    cold = sum(1 for t in tags.values() if t["cold"])
    return {
        "n_test_users": n,
        "cold": cold,
        "warm": n - cold,
        "cold_frac": cold / n if n else 0.0,
    }
