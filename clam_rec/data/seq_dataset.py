"""Sequence datasets for training and inference (ported from A-LLMRec utils)."""

import numpy as np
from torch.utils.data import Dataset


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


class SeqDataset(Dataset):
    """Training: for each user, a (seq, pos, neg) leave-one-out style sample."""

    def __init__(self, user_train, num_user, num_item, max_len):
        self.user_train = user_train
        self.num_user = num_user
        self.num_item = num_item
        self.max_len = max_len

    def __len__(self):
        return self.num_user

    def __getitem__(self, idx):
        user_id = idx + 1
        seq = np.zeros([self.max_len], dtype=np.int32)
        pos = np.zeros([self.max_len], dtype=np.int32)
        neg = np.zeros([self.max_len], dtype=np.int32)
        nxt = self.user_train[user_id][-1]
        i = self.max_len - 1
        ts = set(self.user_train[user_id])
        for item in reversed(self.user_train[user_id][:-1]):
            seq[i] = item
            pos[i] = nxt
            if nxt != 0:
                neg[i] = random_neq(1, self.num_item + 1, ts)
            nxt = item
            i -= 1
            if i == -1:
                break
        return user_id, seq, pos, neg


class SeqDatasetInference(Dataset):
    """Inference: build eval sequence (train + valid item) with the test item as pos."""

    def __init__(self, user_train, user_valid, user_test, use_users, num_item, max_len):
        self.user_train = user_train
        self.user_valid = user_valid
        self.user_test = user_test
        self.use_users = use_users
        self.num_item = num_item
        self.max_len = max_len

    def __len__(self):
        return len(self.use_users)

    def __getitem__(self, idx):
        user_id = self.use_users[idx]
        seq = np.zeros([self.max_len], dtype=np.int32)
        i = self.max_len - 1
        seq[i] = self.user_valid[user_id][0]
        i -= 1
        for item in reversed(self.user_train[user_id]):
            seq[i] = item
            i -= 1
            if i == -1:
                break
        pos = self.user_test[user_id][0]
        # negatives here are unused for LLM generation (candidates built in-model),
        # but kept for API symmetry with SASRec forward.
        neg = np.zeros([3], dtype=np.int32)
        return user_id, seq, pos, neg
