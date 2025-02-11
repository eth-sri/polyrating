import numpy as np
import pandas as pd
from tqdm import tqdm
import math
from sklearn.linear_model import LogisticRegression
from helper import prepare_lmsys_data

# code adjusted from https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH

def compute_mle_elo(
    df, SCALE=400, BASE=10, INIT_RATING=1000, sample_weight=None # code from LM 
):
    
    ptbl_a_win = pd.pivot_table(
        df[df["winner"] == "model_a"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )
    # if no tie, create a zero matrix
    if sum(df["winner"].isin(["tie", "tie (bothbad)"])) == 0:
        ptbl_tie = pd.DataFrame(0, index=ptbl_a_win.index, columns=ptbl_a_win.columns)
    else:
        ptbl_tie = pd.pivot_table(
            df[df["winner"].isin(["tie", "tie (bothbad)"])],
            index="model_a",
            columns="model_b",
            aggfunc="size",
            fill_value=0,
        )
        ptbl_tie = ptbl_tie + ptbl_tie.T

    ptbl_b_win = pd.pivot_table(
        df[df["winner"] == "model_b"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )
    ptbl_win = ptbl_a_win * 2 + ptbl_b_win.T * 2 + ptbl_tie

    models = pd.Series(np.arange(len(ptbl_win.index)), index=ptbl_win.index)

    p = len(models)
    X = np.zeros([p * (p - 1) * 2, p])
    Y = np.zeros(p * (p - 1) * 2)

    cur_row = 0
    sample_weights = []
    for m_a in ptbl_win.index:
        for m_b in ptbl_win.columns:
            if m_a == m_b:
                continue
            # if nan skip
            if math.isnan(ptbl_win.loc[m_a, m_b]) or math.isnan(ptbl_win.loc[m_b, m_a]):
                continue
            X[cur_row, models[m_a]] = +math.log(BASE)
            X[cur_row, models[m_b]] = -math.log(BASE)
            Y[cur_row] = 1.0
            sample_weights.append(ptbl_win.loc[m_a, m_b])

            X[cur_row + 1, models[m_a]] = math.log(BASE)
            X[cur_row + 1, models[m_b]] = -math.log(BASE)
            Y[cur_row + 1] = 0.0
            sample_weights.append(ptbl_win.loc[m_b, m_a])
            cur_row += 2
    X = X[:cur_row]
    Y = Y[:cur_row]

    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-6)
    lr.fit(X, Y, sample_weight=sample_weights)
    elo_scores = SCALE * lr.coef_[0] + INIT_RATING
    if "mixtral-8x7b-instruct-v0.1" in models.index:
        elo_scores += 1114 - elo_scores[models["mixtral-8x7b-instruct-v0.1"]]
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)

def get_bootstrap_result(battles, func_compute_elo, num_round):
    rows = []
    for i in tqdm(range(num_round), desc="bootstrap"):
        rows.append(func_compute_elo(battles.sample(frac=1.0, replace=True, random_state=i)))
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]


if __name__ == '__main__':
    data = prepare_lmsys_data()
    categories = ['is_english', 'is_chinese', 'is_hard', 'is_code']

    n_bootstrap = 1000

    full_result = get_bootstrap_result(data, compute_mle_elo, n_bootstrap)
    elo_mean_full = full_result.mean()
    elo_std = full_result.std()
    # get percentile for each model
    el_95_quantile = np.percentile(full_result, 95, axis=0)
    el_5_quantile = np.percentile(full_result, 5, axis=0)
    full_elo = pd.DataFrame({"mean": elo_mean_full, "std": elo_std, 'lower': elo_mean_full - el_95_quantile, 
                            'upper': elo_mean_full - el_5_quantile})
    categories_elo = []
    for category in categories:
        category_result = get_bootstrap_result(data[data[category]], compute_mle_elo, n_bootstrap)
        merge = pd.merge(full_result.T, category_result.T, left_index=True, right_index=True, suffixes=('', '_category'))
        for i in range(n_bootstrap):
            merge[f'{i}_diff'] = merge[f'{i}_category'] - merge[str(i)]
        # remove all columns in merge that are not diff
        merge = merge[[col for col in merge.columns if 'diff' in col]].T
        # take the mean of the diff columns
        modifier_mean = merge.mean()
        modifier_std = merge.std()
        modifier_95_quantile = np.percentile(merge, 95, axis=0)
        modifier_5_quantile = np.percentile(merge, 5, axis=0)
        categories_elo.append(pd.DataFrame({"mean": modifier_mean, "std": modifier_std, 
                                            'lower': modifier_mean - modifier_95_quantile,
                                            'upper': modifier_mean - modifier_5_quantile}))

    full_elo.to_csv('results/leaderboard_univariate.csv')
    for i, category in enumerate(categories):
        categories_elo[i].to_csv(f'results/leaderboard_univariate_{category}.csv')