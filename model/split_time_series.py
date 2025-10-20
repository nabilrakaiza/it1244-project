import os
import math
import numpy as np
import pandas as pd

# --------- CONFIG ---------
TRANSFORMERS = {
    "trans_1": "../dataset/trans_1.csv",
    "trans_2": "../dataset/trans_2.csv",
}
GROUP_SIZE = 96 * 7        # one week if data is hourly (tune if needed)
TRAIN_FRACTION = 0.80      # 80% of groups go to train pool
VAL_FRACTION_OF_TRAIN = 0.10  # 10% of train groups held out for validation
RANDOM_SEED = 42
# --------------------------

def load_df(path):
    df = pd.read_csv(path)
    # If you have a timestamp column, sort by it to be safe:
    # df = df.sort_values("timestamp").reset_index(drop=True)
    return df.reset_index(drop=True)

def make_group_ids(n_rows, group_size):
    n_groups = math.ceil(n_rows / group_size)
    groups = np.repeat(np.arange(n_groups), group_size)[:n_rows]
    return groups, n_groups

def split_groups(n_groups, train_frac, val_frac_of_train, seed):
    rng = np.random.RandomState(seed)
    all_groups = np.arange(n_groups)
    rng.shuffle(all_groups)

    k_train = int(round(train_frac * n_groups))
    train_groups = np.sort(all_groups[:k_train])
    test_groups  = np.sort(all_groups[k_train:])

    # carve validation from train groups
    n_val = max(1, int(round(val_frac_of_train * len(train_groups))))
    perm = train_groups.copy()
    rng.shuffle(perm)
    val_groups = np.sort(perm[:n_val])
    pure_train_groups = np.array(sorted(set(train_groups) - set(val_groups)))
    return pure_train_groups, val_groups, test_groups

def indices_for_groups(groups, chosen_groups):
    mask = np.isin(groups, chosen_groups)
    return np.where(mask)[0]

def save_indices(out_dir, name, idx):
    os.makedirs(out_dir, exist_ok=True)
    pd.Series(idx, name="row_index").to_csv(
        os.path.join(out_dir, f"indices_{name}.csv"), index=False
    )

def main():
    for tf_name, path in TRANSFORMERS.items():
        df = load_df(path)
        groups, n_groups = make_group_ids(len(df), GROUP_SIZE)
        g_train, g_val, g_test = split_groups(
            n_groups, TRAIN_FRACTION, VAL_FRACTION_OF_TRAIN, RANDOM_SEED
        )

        idx_train = indices_for_groups(groups, g_train)
        idx_val   = indices_for_groups(groups, g_val)
        idx_test  = indices_for_groups(groups, g_test)

        out_dir = f"../splits/{tf_name}"
        save_indices(out_dir, "train", idx_train)
        save_indices(out_dir, "val",   idx_val)
        save_indices(out_dir, "test",  idx_test)

        # quick sanity print
        print(
            f"{tf_name}: rows={len(df)} | groups={n_groups} | "
            f"train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}"
        )

if __name__ == "__main__":
    main()
