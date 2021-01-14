import pandas as pd
import numpy as np
import gc
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from tqdm.notebook import tqdm
import lightgbm as lgb
# import riiideducation
import matplotlib.pyplot as plt

import random
import os

from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import cpu_count
import pickle as pk
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import os
import bisect
#from python_utils import dd


SVAE_NAME = 'lgb.seed42.200.pkl'

def questions_and_lecture_parsing():
    q = pd.read_csv('/Users/han/Desktop/Kaggle-riiid-test-answer-prediction/questions.csv').fillna('')
    bundle_dict = q.groupby('bundle_id')['question_id'].count().to_dict()
    q.loc[:, 'small_part'] = -1
    cnt = 0
    for part in range(1, 8):
        end = q[q.part == part][q[q.part == part]['question_id'].diff(-1) != -1]['question_id'].values
        start = q[q.part == part][q[q.part == part]['question_id'].diff() != 1]['question_id'].values
        for s, t in zip(start, end):
            q.loc[s:t, 'small_part'] = cnt
            cnt += 1

    q['part'] -= 1

    for i in range(3):
        q[f'tags{i}'] = q['tags'].apply(lambda x: [int(item) for item in x.split()][i] if len(x.split()) > i else np.nan)
    for i in range(1, 3):
        q[f'tags{i}'] = q[f'tags{i - 1}'] * 200 + q[f'tags{i}']

    q['tags_count'] = q['tags'].apply(lambda x: len(x.split()))
    return q

def mdiv(a, b):
    if b == 0:
        return np.nan
    return a*1./b

def ll(predicted, actual, eps=1e-14):
    predicted = np.clip(predicted, eps, 1-eps)
    loss = -1*(actual * np.log(predicted) + (1 - actual) * np.log(1-predicted))
    return loss


def merge_parallel(results):
    dfs = [item[0] for item in results]
    dicts = [item[1] for item in results]
    train_data = pd.concat(dfs, axis=0)
    df_train = train_data[(train_data.tag == 1)]
    df_valid = train_data[(train_data.tag == 0)]

    feature_dicts = dicts[0]

    for item in dicts[1:]:
        for feature_name in item.keys():
            feature_dicts[feature_name].update(item[feature_name])

    return df_train, df_valid, feature_dicts


# Random seed
SEED = 42


# Function to seed everything
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


seed_everything(SEED)


# Funcion for user stats with loops
def make_data(df, global_avg_q_time_dict, prior_question_elapsed_time_mean, global_content_cnt_dict, update=True):
    #     # Client dictionaries
    last_u_content_id_dict = defaultdict(float)
    last_u_container_id_dict = defaultdict(float)
    hist_u_answered_correctly_cnt_dict = defaultdict(float)
    hist_u_elapsed_time_sum_dict = defaultdict(float)
    hist_u_explanation_sum_dict = defaultdict(float)
    hist_u_same_part_correctly_cnt_dict = defaultdict(dd)
    hist_u_same_content_id_correctly_cnt_dict = defaultdict(dd)
    timestamp_u = defaultdict(list)
    hist_u_answered_correctly_sum_dict = defaultdict(float)
    hist_u_score_sum_dict = defaultdict(float)
    hist_u_same_part_correctly_sum_dict = defaultdict(dd)
    hist_u_same_content_id_correctly_sum_dict = defaultdict(dd)
    last_u_last_incorrect_timestamp_dict = defaultdict(float)
    hist_u_last_incorrect_cnt_dict = defaultdict(float)
    hist_u_lag_time_sum_dict = defaultdict(float)

    # -----------------------------------------------------------------------
    last_u_diff_container_id = np.zeros(len(df), dtype=np.float32)
    timestamp_u_recency_1 = np.zeros(len(df), dtype=np.float32)
    timestamp_u_gap_time_ratio = np.zeros(len(df), dtype=np.float32)
    timestamp_u_avg_time_ratio = np.zeros(len(df), dtype=np.float32)
    timestamp_u_lag_time_ratio = np.zeros(len(df), dtype=np.float32)
    timestamp_u_lag_time = np.zeros(len(df), dtype=np.float32)

    hist_u_lag_time_raito = np.zeros(len(df), dtype=np.float32)
    hist_u_answered_correctly_ratio = np.zeros(len(df), dtype=np.float32)
    hist_u_elapsed_time_ratio = np.zeros(len(df), dtype=np.float32)
    hist_u_explanation_ratio = np.zeros(len(df), dtype=np.float32)
    hist_u_score_ratio = np.zeros(len(df), dtype=np.float32)
    hist_u_same_part_sum = np.zeros(len(df), dtype=np.float32)
    hist_u_same_part_cnt = np.zeros(len(df), dtype=np.float32)
    hist_u_same_part_ratio = np.zeros(len(df), dtype=np.float32)
    hist_u_same_content_id_sum = np.zeros(len(df), dtype=np.float32)
    hist_u_same_content_id_cnt = np.zeros(len(df), dtype=np.float32)
    hist_u_same_content_id_ratio = np.zeros(len(df), dtype=np.float32)
    hist_u_last_incorrect_timestamp = np.zeros(len(df), dtype=np.float32)
    hist_u_last_incorrect_cnt = np.zeros(len(df), dtype=np.float32)
    # User Question
    answered_correctly_uq_count = np.zeros(len(df), dtype=np.int32)
    # -----------------------------------------------------------------------

    for num, row in tqdm(enumerate(df[['user_id', 'answered_correctly', 'content_id', 'prior_question_elapsed_time',
                                       'prior_question_had_explanation', 'timestamp', 'contentid_mean',
                                       'task_container_id', 'part']].values)):

        last_u_content_id = last_u_content_id_dict.get(row[0], np.nan)
        last_u_container_id = last_u_container_id_dict.get(row[0], np.nan)

        last_u_diff_container_id[num] = row[7] - last_u_container_id  # 1

        last_u_sum_time = row[3] * global_content_cnt_dict.get(last_u_content_id, 1)
        sum_time_consum = global_avg_q_time_dict.get(row[2],
                                                     prior_question_elapsed_time_mean) * global_content_cnt_dict.get(
            row[2], 1)

        # Client features assignation
        # ------------------------------------------------------------------

        if len(timestamp_u[row[0]]) == 0:
            timestamp_u_recency_1[num] = np.nan  # 2
            timestamp_u_gap_time_ratio[num] = np.nan  # 3
            timestamp_u_avg_time_ratio[num] = np.nan  # 4
            timestamp_u_lag_time_ratio[num] = np.nan  # 5
            timestamp_u_lag_time[num] = np.nan  # 6

        elif len(timestamp_u[row[0]]) == 1:
            timestamp_u_recency_1[num] = row[5] - timestamp_u[row[0]][0]
            timestamp_u_gap_time_ratio[num] = mdiv(sum_time_consum, (row[5] - timestamp_u[row[0]][0]))
            timestamp_u_avg_time_ratio[num] = mdiv((row[5] - timestamp_u[row[0]][0]),
                                                   global_content_cnt_dict.get(row[2], 1))
            timestamp_u_lag_time_ratio[num] = np.nan
            timestamp_u_lag_time[num] = np.nan

        elif len(timestamp_u[row[0]]) == 2:
            timestamp_u_recency_1[num] = row[5] - timestamp_u[row[0]][1]
            timestamp_u_gap_time_ratio[num] = mdiv(sum_time_consum, (row[5] - timestamp_u[row[0]][1]))
            timestamp_u_avg_time_ratio[num] = mdiv((row[5] - timestamp_u[row[0]][1]),
                                                   global_content_cnt_dict.get(row[2], 1))
            timestamp_u_lag_time_ratio[num] = mdiv((timestamp_u[row[0]][1] - timestamp_u[row[0]][0] - last_u_sum_time),
                                                   (row[5] - timestamp_u[row[0]][1] + 1))
            timestamp_u_lag_time[num] = (timestamp_u[row[0]][1] - timestamp_u[row[0]][0] - last_u_sum_time)

        elif len(timestamp_u[row[0]]) == 3:
            timestamp_u_recency_1[num] = row[5] - timestamp_u[row[0]][2]
            timestamp_u_gap_time_ratio[num] = mdiv(sum_time_consum, (row[5] - timestamp_u[row[0]][2]))
            timestamp_u_avg_time_ratio[num] = mdiv((row[5] - timestamp_u[row[0]][2]),
                                                   global_content_cnt_dict.get(row[2], 1))
            timestamp_u_lag_time_ratio[num] = mdiv((timestamp_u[row[0]][2] - timestamp_u[row[0]][1] - sum_time_consum),
                                                   (row[5] - timestamp_u[row[0]][2] + 1))
            timestamp_u_lag_time[num] = (timestamp_u[row[0]][2] - timestamp_u[row[0]][1] - last_u_sum_time)

        if timestamp_u_lag_time[num] is not np.nan:
            hist_u_lag_time_sum_dict[row[0]] += max(0, min(timestamp_u_lag_time[num], 300 * 1000))
        hist_u_lag_time_raito[num] = mdiv(hist_u_lag_time_sum_dict[row[0]], hist_u_answered_correctly_cnt_dict[row[0]])

        hist_u_answered_correctly_ratio[num] = mdiv(hist_u_answered_correctly_sum_dict[row[0]],
                                                    hist_u_answered_correctly_cnt_dict[row[0]])
        hist_u_elapsed_time_ratio[num] = mdiv(hist_u_elapsed_time_sum_dict[row[0]],
                                              hist_u_answered_correctly_cnt_dict[row[0]])
        hist_u_explanation_ratio[num] = mdiv(hist_u_explanation_sum_dict[row[0]],
                                             hist_u_answered_correctly_cnt_dict[row[0]])

        hist_u_score_ratio[num] = mdiv(hist_u_score_sum_dict[row[0]], hist_u_answered_correctly_cnt_dict[row[0]])

        hist_u_same_part_sum[num] = hist_u_same_part_correctly_sum_dict[row[0]][row[8]]
        hist_u_same_part_cnt[num] = hist_u_same_part_correctly_cnt_dict[row[0]][row[8]]
        hist_u_same_part_ratio[num] = mdiv(hist_u_same_part_sum[num], hist_u_same_part_cnt[num])

        hist_u_same_content_id_sum[num] = hist_u_same_content_id_correctly_sum_dict[row[0]][row[2]]
        hist_u_same_content_id_cnt[num] = hist_u_same_content_id_correctly_cnt_dict[row[0]][row[2]]
        hist_u_same_content_id_ratio[num] = mdiv(hist_u_same_content_id_sum[num], hist_u_same_content_id_cnt[num])

        hist_u_last_incorrect_timestamp[num] = row[5] - last_u_last_incorrect_timestamp_dict[row[0]]
        hist_u_last_incorrect_cnt[num] = hist_u_last_incorrect_cnt_dict[row[0]]

        # ------------------------------------------------------------------
        # Client features updates
        hist_u_answered_correctly_cnt_dict[row[0]] += 1
        hist_u_elapsed_time_sum_dict[row[0]] += row[3]
        hist_u_explanation_sum_dict[row[0]] += int(row[4])
        hist_u_same_part_correctly_cnt_dict[row[0]][row[8]] += 1
        hist_u_same_content_id_correctly_cnt_dict[row[0]][row[2]] += 1

        if len(timestamp_u[row[0]]) == 0 or row[5] != timestamp_u[row[0]][-1]:
            last_u_content_id_dict[row[0]] = row[2]
            if len(timestamp_u[row[0]]) == 3:
                timestamp_u[row[0]].pop(0)
                timestamp_u[row[0]].append(row[5])
            else:
                timestamp_u[row[0]].append(row[5])
        if len(timestamp_u[row[0]]) != 0 or row[5] == timestamp_u[row[0]][-1]:
            last_u_container_id_dict[row[0]] = row[7]

        # Flag for training and inference
        if update:
            # ------------------------------------------------------------------
            # Client features updates
            hist_u_answered_correctly_sum_dict[row[0]] += row[1]
            hist_u_score_sum_dict[row[0]] += (1 if row[1] == 1 else -1) * ll(row[6], row[1])
            hist_u_same_part_correctly_sum_dict[row[0]][row[8]] += row[1]
            hist_u_same_content_id_correctly_sum_dict[row[0]][row[2]] += row[1]
            if row[1] == 0:
                last_u_last_incorrect_timestamp_dict[row[0]] = row[5]
                hist_u_last_incorrect_cnt_dict[row[0]] = 0
            else:
                hist_u_last_incorrect_cnt_dict[row[0]] += 1

    user_df = pd.DataFrame({
        'last_u_diff_container_id': last_u_diff_container_id,
        'timestamp_u_recency_1': timestamp_u_recency_1,
        'timestamp_u_gap_time_ratio': timestamp_u_gap_time_ratio,
        'timestamp_u_avg_time_ratio': timestamp_u_avg_time_ratio,
        'timestamp_u_lag_time_ratio': timestamp_u_lag_time_ratio,
        'timestamp_u_lag_time': timestamp_u_lag_time,

        'hist_u_lag_time_raito': hist_u_lag_time_raito,
        'hist_u_answered_correctly_ratio': hist_u_answered_correctly_ratio,
        'hist_u_elapsed_time_ratio': hist_u_elapsed_time_ratio,
        'hist_u_explanation_ratio': hist_u_explanation_ratio,
        'hist_u_score_ratio': hist_u_score_ratio,
        'hist_u_same_part_sum': hist_u_same_part_sum,
        'hist_u_same_part_cnt': hist_u_same_part_cnt,
        'hist_u_same_part_ratio': hist_u_same_part_ratio,
        'hist_u_same_content_id_sum': hist_u_same_content_id_sum,
        'hist_u_same_content_id_cnt': hist_u_same_content_id_cnt,
        'hist_u_same_content_id_ratio': hist_u_same_content_id_ratio,
        'hist_u_last_incorrect_timestamp': hist_u_last_incorrect_timestamp,
        'hist_u_last_incorrect_cnt': hist_u_last_incorrect_cnt,

        'timestamp': df['timestamp'].values,
        'content_id': df['content_id'].values,
        'task_container_id': df['task_container_id'].values,
        'prior_question_elapsed_time': df['prior_question_elapsed_time'].values,
        'part': df['part'].values,
        'tags0': df['tags0'].values,
        'tags1': df['tags1'].values,
        'tags2': df['tags2'].values,
        'tags_count': df['tags_count'].values,
        'contentid_mean': df['contentid_mean'].values,
        'tag': df['tag'].values,
        'answered_correctly': df['answered_correctly'].values,
    })

    features_dicts = {
        'last_u_content_id_dict': last_u_content_id_dict,
        'last_u_container_id_dict': last_u_container_id_dict,
        'hist_u_answered_correctly_cnt_dict': hist_u_answered_correctly_cnt_dict,
        'hist_u_elapsed_time_sum_dict': hist_u_elapsed_time_sum_dict,
        'hist_u_explanation_sum_dict': hist_u_explanation_sum_dict,
        'hist_u_same_part_correctly_cnt_dict': hist_u_same_part_correctly_cnt_dict,
        'hist_u_same_content_id_correctly_cnt_dict': hist_u_same_content_id_correctly_cnt_dict,
        'timestamp_u': timestamp_u,
        'hist_u_answered_correctly_sum_dict': hist_u_answered_correctly_sum_dict,
        'hist_u_score_sum_dict': hist_u_score_sum_dict,
        'hist_u_same_part_correctly_sum_dict': hist_u_same_part_correctly_sum_dict,
        'hist_u_same_content_id_correctly_sum_dict': hist_u_same_content_id_correctly_sum_dict,
        'last_u_last_incorrect_timestamp_dict': last_u_last_incorrect_timestamp_dict,
        'hist_u_last_incorrect_cnt_dict': hist_u_last_incorrect_cnt_dict,
        'hist_u_lag_time_sum_dict': hist_u_lag_time_sum_dict,
    }

    return user_df, features_dicts


# Funcion for user stats with loops
def add_features(df,
                 last_u_content_id_dict,
                 last_u_container_id_dict,
                 hist_u_answered_correctly_cnt_dict,
                 hist_u_elapsed_time_sum_dict,
                 hist_u_explanation_sum_dict,
                 hist_u_same_part_correctly_cnt_dict,
                 hist_u_same_content_id_correctly_cnt_dict,
                 timestamp_u,
                 hist_u_answered_correctly_sum_dict,
                 hist_u_score_sum_dict,
                 hist_u_same_part_correctly_sum_dict,
                 hist_u_same_content_id_correctly_sum_dict,
                 last_u_last_incorrect_timestamp_dict,
                 hist_u_last_incorrect_cnt_dict,
                 hist_u_lag_time_sum_dict,
                 global_avg_q_time_dict, prior_question_elapsed_time_mean, global_content_cnt_dict,
                 update=True):
    # -----------------------------------------------------------------------
    last_u_diff_container_id = np.zeros(len(df), dtype=np.float32)
    timestamp_u_recency_1 = np.zeros(len(df), dtype=np.float32)
    timestamp_u_gap_time_ratio = np.zeros(len(df), dtype=np.float32)
    timestamp_u_avg_time_ratio = np.zeros(len(df), dtype=np.float32)
    timestamp_u_lag_time_ratio = np.zeros(len(df), dtype=np.float32)
    timestamp_u_lag_time = np.zeros(len(df), dtype=np.float32)

    hist_u_lag_time_raito = np.zeros(len(df), dtype=np.float32)
    hist_u_answered_correctly_ratio = np.zeros(len(df), dtype=np.float32)
    hist_u_elapsed_time_ratio = np.zeros(len(df), dtype=np.float32)
    hist_u_explanation_ratio = np.zeros(len(df), dtype=np.float32)
    hist_u_score_ratio = np.zeros(len(df), dtype=np.float32)
    hist_u_same_part_sum = np.zeros(len(df), dtype=np.float32)
    hist_u_same_part_cnt = np.zeros(len(df), dtype=np.float32)
    hist_u_same_part_ratio = np.zeros(len(df), dtype=np.float32)
    hist_u_same_content_id_sum = np.zeros(len(df), dtype=np.float32)
    hist_u_same_content_id_cnt = np.zeros(len(df), dtype=np.float32)
    hist_u_same_content_id_ratio = np.zeros(len(df), dtype=np.float32)
    hist_u_last_incorrect_timestamp = np.zeros(len(df), dtype=np.float32)
    hist_u_last_incorrect_cnt = np.zeros(len(df), dtype=np.float32)
    # User Question
    answered_correctly_uq_count = np.zeros(len(df), dtype=np.int32)
    # -----------------------------------------------------------------------

    for num, row in tqdm(enumerate(df[['user_id', 'answered_correctly', 'content_id', 'prior_question_elapsed_time',
                                       'prior_question_had_explanation', 'timestamp', 'contentid_mean',
                                       'task_container_id', 'part']].values)):

        last_u_content_id = last_u_content_id_dict.get(row[0], np.nan)
        last_u_container_id = last_u_container_id_dict.get(row[0], np.nan)

        last_u_diff_container_id[num] = row[7] - last_u_container_id  # 1

        last_u_sum_time = row[3] * global_content_cnt_dict.get(last_u_content_id, 1)
        sum_time_consum = global_avg_q_time_dict.get(row[2],
                                                     prior_question_elapsed_time_mean) * global_content_cnt_dict.get(
            row[2], 1)

        # Client features assignation
        # ------------------------------------------------------------------

        if len(timestamp_u[row[0]]) == 0:
            timestamp_u_recency_1[num] = np.nan  # 2
            timestamp_u_gap_time_ratio[num] = np.nan  # 3
            timestamp_u_avg_time_ratio[num] = np.nan  # 4
            timestamp_u_lag_time_ratio[num] = np.nan  # 5
            timestamp_u_lag_time[num] = np.nan  # 6

        elif len(timestamp_u[row[0]]) == 1:
            timestamp_u_recency_1[num] = row[5] - timestamp_u[row[0]][0]
            timestamp_u_gap_time_ratio[num] = mdiv(sum_time_consum, (row[5] - timestamp_u[row[0]][0]))
            timestamp_u_avg_time_ratio[num] = mdiv((row[5] - timestamp_u[row[0]][0]),
                                                   global_content_cnt_dict.get(row[2], 1))
            timestamp_u_lag_time_ratio[num] = np.nan
            timestamp_u_lag_time[num] = np.nan

        elif len(timestamp_u[row[0]]) == 2:
            timestamp_u_recency_1[num] = row[5] - timestamp_u[row[0]][1]
            timestamp_u_gap_time_ratio[num] = mdiv(sum_time_consum, (row[5] - timestamp_u[row[0]][1]))
            timestamp_u_avg_time_ratio[num] = mdiv((row[5] - timestamp_u[row[0]][1]),
                                                   global_content_cnt_dict.get(row[2], 1))
            timestamp_u_lag_time_ratio[num] = mdiv((timestamp_u[row[0]][1] - timestamp_u[row[0]][0] - last_u_sum_time),
                                                   (row[5] - timestamp_u[row[0]][1] + 1))
            timestamp_u_lag_time[num] = (timestamp_u[row[0]][1] - timestamp_u[row[0]][0] - last_u_sum_time)

        elif len(timestamp_u[row[0]]) == 3:
            timestamp_u_recency_1[num] = row[5] - timestamp_u[row[0]][2]
            timestamp_u_gap_time_ratio[num] = mdiv(sum_time_consum, (row[5] - timestamp_u[row[0]][2]))
            timestamp_u_avg_time_ratio[num] = mdiv((row[5] - timestamp_u[row[0]][2]),
                                                   global_content_cnt_dict.get(row[2], 1))
            timestamp_u_lag_time_ratio[num] = mdiv((timestamp_u[row[0]][2] - timestamp_u[row[0]][1] - sum_time_consum),
                                                   (row[5] - timestamp_u[row[0]][2] + 1))
            timestamp_u_lag_time[num] = (timestamp_u[row[0]][2] - timestamp_u[row[0]][1] - last_u_sum_time)

        if timestamp_u_lag_time[num] is not np.nan:
            hist_u_lag_time_sum_dict[row[0]] += max(0, min(timestamp_u_lag_time[num], 300 * 1000))
        hist_u_lag_time_raito[num] = mdiv(hist_u_lag_time_sum_dict[row[0]], hist_u_answered_correctly_cnt_dict[row[0]])

        hist_u_answered_correctly_ratio[num] = mdiv(hist_u_answered_correctly_sum_dict[row[0]],
                                                    hist_u_answered_correctly_cnt_dict[row[0]])
        hist_u_elapsed_time_ratio[num] = mdiv(hist_u_elapsed_time_sum_dict[row[0]],
                                              hist_u_answered_correctly_cnt_dict[row[0]])
        hist_u_explanation_ratio[num] = mdiv(hist_u_explanation_sum_dict[row[0]],
                                             hist_u_answered_correctly_cnt_dict[row[0]])

        hist_u_score_ratio[num] = mdiv(hist_u_score_sum_dict[row[0]], hist_u_answered_correctly_cnt_dict[row[0]])

        hist_u_same_part_sum[num] = hist_u_same_part_correctly_sum_dict[row[0]][row[8]]
        hist_u_same_part_cnt[num] = hist_u_same_part_correctly_cnt_dict[row[0]][row[8]]
        hist_u_same_part_ratio[num] = mdiv(hist_u_same_part_sum[num], hist_u_same_part_cnt[num])

        hist_u_same_content_id_sum[num] = hist_u_same_content_id_correctly_sum_dict[row[0]][row[2]]
        hist_u_same_content_id_cnt[num] = hist_u_same_content_id_correctly_cnt_dict[row[0]][row[2]]
        hist_u_same_content_id_ratio[num] = mdiv(hist_u_same_content_id_sum[num], hist_u_same_content_id_cnt[num])

        hist_u_last_incorrect_timestamp[num] = row[5] - last_u_last_incorrect_timestamp_dict[row[0]]
        hist_u_last_incorrect_cnt[num] = hist_u_last_incorrect_cnt_dict[row[0]]

        # ------------------------------------------------------------------
        # Client features updates
        hist_u_answered_correctly_cnt_dict[row[0]] += 1
        hist_u_elapsed_time_sum_dict[row[0]] += row[3]
        hist_u_explanation_sum_dict[row[0]] += int(row[4])
        hist_u_same_part_correctly_cnt_dict[row[0]][row[8]] += 1
        hist_u_same_content_id_correctly_cnt_dict[row[0]][row[2]] += 1

        if len(timestamp_u[row[0]]) == 0 or row[5] != timestamp_u[row[0]][-1]:
            last_u_content_id_dict[row[0]] = row[2]
            if len(timestamp_u[row[0]]) == 3:
                timestamp_u[row[0]].pop(0)
                timestamp_u[row[0]].append(row[5])
            else:
                timestamp_u[row[0]].append(row[5])
        if len(timestamp_u[row[0]]) != 0 or row[5] == timestamp_u[row[0]][-1]:
            last_u_container_id_dict[row[0]] = row[7]

        # Flag for training and inference
        if update:
            # ------------------------------------------------------------------
            # Client features updates
            hist_u_answered_correctly_sum_dict[row[0]] += row[1]
            hist_u_score_sum_dict[row[0]] += (1 if row[1] == 1 else -1) * ll(row[6], row[1])
            hist_u_same_part_correctly_sum_dict[row[0]][row[8]] += row[1]
            hist_u_same_content_id_correctly_sum_dict[row[0]][row[2]] += row[1]
            if row[1] == 0:
                last_u_last_incorrect_timestamp_dict[row[0]] = row[5]
                hist_u_last_incorrect_cnt_dict[row[0]] = 0
            else:
                hist_u_last_incorrect_cnt_dict[row[0]] += 1

    user_df = pd.DataFrame({
        'last_u_diff_container_id': last_u_diff_container_id,
        'timestamp_u_recency_1': timestamp_u_recency_1,
        'timestamp_u_gap_time_ratio': timestamp_u_gap_time_ratio,
        'timestamp_u_avg_time_ratio': timestamp_u_avg_time_ratio,
        'timestamp_u_lag_time_ratio': timestamp_u_lag_time_ratio,
        'timestamp_u_lag_time': timestamp_u_lag_time,

        'hist_u_lag_time_raito': hist_u_lag_time_raito,
        'hist_u_answered_correctly_ratio': hist_u_answered_correctly_ratio,
        'hist_u_elapsed_time_ratio': hist_u_elapsed_time_ratio,
        'hist_u_explanation_ratio': hist_u_explanation_ratio,
        'hist_u_score_ratio': hist_u_score_ratio,
        'hist_u_same_part_sum': hist_u_same_part_sum,
        'hist_u_same_part_cnt': hist_u_same_part_cnt,
        'hist_u_same_part_ratio': hist_u_same_part_ratio,
        'hist_u_same_content_id_sum': hist_u_same_content_id_sum,
        'hist_u_same_content_id_cnt': hist_u_same_content_id_cnt,
        'hist_u_same_content_id_ratio': hist_u_same_content_id_ratio,
        'hist_u_last_incorrect_timestamp': hist_u_last_incorrect_timestamp,
        'hist_u_last_incorrect_cnt': hist_u_last_incorrect_cnt,
    })

    df = pd.concat([df.reset_index(drop=True), user_df], axis=1)
    return df


def update_features(df,
                    hist_u_answered_correctly_sum_dict,
                    hist_u_score_sum_dict,
                    hist_u_same_part_correctly_sum_dict,
                    hist_u_same_content_id_correctly_sum_dict,
                    last_u_last_incorrect_timestamp_dict,
                    hist_u_last_incorrect_cnt_dict):
    for row in df[['user_id', 'answered_correctly', 'content_id', 'prior_question_elapsed_time',
                   'prior_question_had_explanation', 'timestamp', 'contentid_mean',
                   'task_container_id', 'part', 'content_type_id']].values:
        if row[-1] == 0:
            # ------------------------------------------------------------------
            # Client features updates
            hist_u_answered_correctly_sum_dict[row[0]] += row[1]
            hist_u_score_sum_dict[row[0]] += (1 if row[1] == 1 else -1) * ll(row[6], row[1])
            hist_u_same_part_correctly_sum_dict[row[0]][row[8]] += row[1]
            hist_u_same_content_id_correctly_sum_dict[row[0]][row[2]] += row[1]
            if row[1] == 0:
                last_u_last_incorrect_timestamp_dict[row[0]] = row[5]
                hist_u_last_incorrect_cnt_dict[row[0]] = 0
            else:
                hist_u_last_incorrect_cnt_dict[row[0]] += 1

    return


def read_and_preprocess(feature_engineering=False):
    nrows = None
    tags = '../input/folds/tag_data_full.pkl'
    question_file = '/Users/han/Desktop/Kaggle-riiid-test-answer-prediction/questions.csv'

    train_data = pd.read_csv('/Users/han/Desktop/Kaggle-riiid-test-answer-prediction/train.csv',
                             nrows=None,
                             dtype={
                                 'row_id': 'int64',
                                 'timestamp': 'int64',
                                 'user_id': 'int32',
                                 'content_id': 'int16',
                                 'content_type_id': 'int8',
                                 'task_container_id': 'int16',
                                 'user_answer': 'int8',
                                 'answered_correctly': 'int8',
                                 'prior_question_elapsed_time': 'float32',
                                 'prior_question_had_explanation': 'boolean'
                             }
                             )

    tags = pd.read_pickle(tags)
    train_data.loc[:, 'tag'] = tags

    # Read data
    feld_needed = ['timestamp', 'user_id', 'answered_correctly', 'content_id', 'content_type_id',
                   'prior_question_elapsed_time', 'prior_question_had_explanation', 'task_container_id', 'tag']
    train = train_data[(train_data.tag == 1)][feld_needed]
    valid = train_data[(train_data.tag == 0)][feld_needed]

    # Filter by content_type_id to discard lectures
    train = train.loc[train.content_type_id == False].reset_index(drop=True)
    valid = valid.loc[valid.content_type_id == False].reset_index(drop=True)

    # Changing dtype to avoid lightgbm error
    train['prior_question_had_explanation'] = train.prior_question_had_explanation.fillna(False).astype('int8')
    valid['prior_question_had_explanation'] = valid.prior_question_had_explanation.fillna(False).astype('int8')

    # Fill prior question elapsed time with the mean
    prior_question_elapsed_time_mean = train['prior_question_elapsed_time'].dropna().mean()
    train['prior_question_elapsed_time'].fillna(prior_question_elapsed_time_mean, inplace=True)
    valid['prior_question_elapsed_time'].fillna(prior_question_elapsed_time_mean, inplace=True)

    # Merge with question dataframe
    questions_df = questions_and_lecture_parsing()
    questions_df['part'] = questions_df['part'].astype(np.int32)
    questions_df['bundle_id'] = questions_df['bundle_id'].astype(np.int32)

    global_content_cnt_dict = questions_df.bundle_id.value_counts().to_dict()

    train = pd.merge(train, questions_df, left_on='content_id', right_on='question_id', how='left')
    valid = pd.merge(valid, questions_df, left_on='content_id', right_on='question_id', how='left')

    ##### global target encoding features ######
    contentid_count = train['content_id'].value_counts()

    contentidmean_dict = train.groupby('content_id')['answered_correctly'].mean()
    contentidmean_dict = contentidmean_dict[contentid_count > 10].to_dict()
    train['contentid_mean'] = train['content_id'].map(lambda x: contentidmean_dict.get(x, np.nan))
    valid['contentid_mean'] = valid['content_id'].map(lambda x: contentidmean_dict.get(x, np.nan))

    #########

    prior_question_elapsed_time_mean = train['prior_question_elapsed_time'].dropna().mean()

    #### global avg time
    ret = defaultdict(list)
    question_data = train_data[train_data.content_type_id == 0].copy()
    question_data = question_data[question_data.timestamp.diff() != 0]
    elapsed_time = question_data.prior_question_elapsed_time.shift(-1).values
    bundle_id = question_data.content_id.astype(int).values

    for x, y in zip(bundle_id, elapsed_time):
        if not np.isnan(y):
            if not x in ret: ret[x] = []
            ret[x].append(y)
    global_avg_q_time_dict = {}
    for x, y in ret.items():
        if len(y) > 100:
            y = sorted(y)
            global_avg_q_time_dict[x] = np.mean(y)
    ####
    # Fill prior question elapsed time with the mean
    train['prior_question_elapsed_time'].fillna(prior_question_elapsed_time_mean, inplace=True)
    valid['prior_question_elapsed_time'].fillna(prior_question_elapsed_time_mean, inplace=True)

    print('User feature calculation started...')
    print('\n')
    train_data = pd.concat([train, valid], axis=0).reset_index(drop=True)
    user_id = train_data['user_id'].unique()
    n_parts = 4
    n_cores = 20
    period = len(user_id) // (n_cores * n_parts - 1)
    userid_groups = [user_id[i * period:(i + 1) * period] for i in range(n_cores * n_parts)]

    results_all = []

    def my_make_data(df):
        return make_data(df, global_avg_q_time_dict, prior_question_elapsed_time_mean, global_content_cnt_dict)

    for i in range(n_parts):
        print('PART {}'.format(i))
        split_dfs = [train_data[train_data.user_id.isin(ids)] for ids in userid_groups[i * n_cores:(i + 1) * n_cores]]
        print([item.shape for item in split_dfs])
        results = Parallel(n_jobs=n_cores)(delayed(my_make_data)(item) for item in split_dfs)
        results_all.extend(results)

    df_train, df_valid, features_dicts = merge_parallel(results_all)

    gc.collect()

    print('User feature calculation completed...')
    print('\n')
    global_dicts = {
        'global_avg_q_time_dict': global_avg_q_time_dict,
        'prior_question_elapsed_time_mean': prior_question_elapsed_time_mean,
        'global_content_cnt_dict': global_content_cnt_dict,
        'contentidmean_dict': contentidmean_dict
    }

    return df_train, df_valid, questions_df, prior_question_elapsed_time_mean, features_dicts, global_dicts


# Function for training and evaluation
def train_and_evaluate(train, valid, feature_engineering=False):
    TARGET = 'answered_correctly'
    # Features to train and predict
    FEATURES = [
        'timestamp', 'content_id', 'task_container_id', 'prior_question_elapsed_time',
        'part',
        # 'small_part',
        'tags0', 'tags1', 'tags2', 'tags_count',

        # 'tags0_mean',
        'contentid_mean',
        # 'contentid_std',
        # 'contentid_skew',
        # 'part_mean',
        'last_u_diff_container_id',
        'timestamp_u_recency_1',
        'timestamp_u_gap_time_ratio',
        'timestamp_u_avg_time_ratio',
        'timestamp_u_lag_time_ratio',
        'timestamp_u_lag_time',

        'hist_u_lag_time_raito',
        'hist_u_answered_correctly_ratio',
        'hist_u_elapsed_time_ratio',
        'hist_u_explanation_ratio',
        'hist_u_score_ratio',
        # 'hist_u_same_part_sum',
        # 'hist_u_same_part_cnt',
        'hist_u_same_part_ratio',
        # 'hist_u_same_content_id_sum',
        # 'hist_u_same_content_id_cnt',
        'hist_u_same_content_id_ratio',
        'hist_u_last_incorrect_timestamp',
        'hist_u_last_incorrect_cnt',
    ]

    gc.collect()
    print(f'Traning with {train.shape[0]} rows and {len(FEATURES)} features')
    drop_cols = list(set(train.columns) - set(FEATURES))
    y_train = train[TARGET]
    y_val = valid[TARGET]
    # Drop unnecessary columns
    train.drop(drop_cols, axis=1, inplace=True)
    valid.drop(drop_cols, axis=1, inplace=True)
    gc.collect()

    lgb_train = lgb.Dataset(train[FEATURES], y_train)
    lgb_valid = lgb.Dataset(valid[FEATURES], y_val)
    del train, y_train
    gc.collect()

    params = {
        'objective': 'binary',
        'boosting': 'gbdt',
        'metric': 'auc',
        'max_depth': -1,
        'learning_rate': 0.18,
        'num_leaves': 95,  # 80,
        'subsample': 0.83524,  # 0.8,
        'feature_fraction': 0.4,  # 0.8,
        'bagging_freq': 5,
        'num_threads': 20,
        'reg_alpha': 90.312,
        'reg_lambda': 7.880,
    }

    model = lgb.train(
        params=params,
        feature_name=FEATURES,
        categorical_feature=['part', 'content_id'],
        train_set=lgb_train,
        num_boost_round=200,
        valid_sets=[lgb_train, lgb_valid],
        early_stopping_rounds=50,
        verbose_eval=50
    )

    print('Our Roc Auc score for the validation data is:', roc_auc_score(y_val, model.predict(valid[FEATURES])))

    feature_importance = model.feature_importance()
    feature_importance = pd.DataFrame({'Features': FEATURES, 'Importance': feature_importance}).sort_values(
        'Importance', ascending=False)

    #     fig = plt.figure(figsize = (10, 10))
    #     fig.suptitle('Feature Importance', fontsize = 20)
    #     plt.tick_params(axis = 'x', labelsize = 12)
    #     plt.tick_params(axis = 'y', labelsize = 12)
    #     plt.xlabel('Importance', fontsize = 15)
    #     plt.ylabel('Features', fontsize = 15)
    #     sns.barplot(x = feature_importance['Importance'], y = feature_importance['Features'], orient = 'h')
    #     plt.show()

    return TARGET, FEATURES, model


# Using time series api that simulates production predictions
def inference(TARGET, FEATURES, model, questions_df, prior_question_elapsed_time_mean, features_dicts, global_dicts):
    # Get feature dict
    last_u_content_id_dict = features_dicts['last_u_content_id_dict']
    last_u_container_id_dict = features_dicts['last_u_container_id_dict']
    hist_u_answered_correctly_cnt_dict = features_dicts['hist_u_answered_correctly_cnt_dict']
    hist_u_elapsed_time_sum_dict = features_dicts['hist_u_elapsed_time_sum_dict']
    hist_u_explanation_sum_dict = features_dicts['hist_u_explanation_sum_dict']
    hist_u_same_part_correctly_cnt_dict = features_dicts['hist_u_same_part_correctly_cnt_dict']
    hist_u_same_content_id_correctly_cnt_dict = features_dicts['hist_u_same_content_id_correctly_cnt_dict']
    timestamp_u = features_dicts['timestamp_u']
    hist_u_answered_correctly_sum_dict = features_dicts['hist_u_answered_correctly_sum_dict']
    hist_u_score_sum_dict = features_dicts['hist_u_score_sum_dict']
    hist_u_same_part_correctly_sum_dict = features_dicts['hist_u_same_part_correctly_sum_dict']
    hist_u_same_content_id_correctly_sum_dict = features_dicts['hist_u_same_content_id_correctly_sum_dict']
    last_u_last_incorrect_timestamp_dict = features_dicts['last_u_last_incorrect_timestamp_dict']
    hist_u_last_incorrect_cnt_dict = features_dicts['hist_u_last_incorrect_cnt_dict']
    hist_u_lag_time_sum_dict = features_dicts['hist_u_lag_time_sum_dict']

    # Get global dict
    global_avg_q_time_dict = global_dicts['global_avg_q_time_dict']
    prior_question_elapsed_time_mean = global_dicts['prior_question_elapsed_time_mean']
    global_content_cnt_dict = global_dicts['global_content_cnt_dict']
    contentidmean_dict = global_dicts['contentidmean_dict']

    # Get api iterator and predictor
    env = riiideducation.make_env()
    iter_test = env.iter_test()
    set_predict = env.predict

    previous_test_df = None
    for (test_df, sample_prediction_df) in iter_test:
        if previous_test_df is not None:
            previous_test_df[TARGET] = eval(test_df["prior_group_answers_correct"].iloc[0])
            previous_test_df = pd.merge(previous_test_df, questions_df, left_on='content_id', right_on='question_id',
                                        how='left')
            previous_test_df['contentid_mean'] = previous_test_df['content_id'].map(
                lambda x: contentidmean_dict.get(x, np.nan))
            update_features(previous_test_df,
                            hist_u_answered_correctly_sum_dict,
                            hist_u_score_sum_dict,
                            hist_u_same_part_correctly_sum_dict,
                            hist_u_same_content_id_correctly_sum_dict,
                            last_u_last_incorrect_timestamp_dict,
                            hist_u_last_incorrect_cnt_dict)
        previous_test_df = test_df.copy()
        test_df = test_df[test_df['content_type_id'] == 0].reset_index(drop=True)
        test_df['prior_question_had_explanation'] = test_df.prior_question_had_explanation.fillna(False).astype('int8')
        test_df['prior_question_elapsed_time'].fillna(prior_question_elapsed_time_mean, inplace=True)
        test_df = pd.merge(test_df, questions_df, left_on='content_id', right_on='question_id', how='left')
        test_df[TARGET] = 0
        test_df['contentid_mean'] = test_df['content_id'].map(lambda x: contentidmean_dict.get(x, np.nan))
        test_df = add_features(test_df,
                               last_u_content_id_dict,
                               last_u_container_id_dict,
                               hist_u_answered_correctly_cnt_dict,
                               hist_u_elapsed_time_sum_dict,
                               hist_u_explanation_sum_dict,
                               hist_u_same_part_correctly_cnt_dict,
                               hist_u_same_content_id_correctly_cnt_dict,
                               timestamp_u,
                               hist_u_answered_correctly_sum_dict,
                               hist_u_score_sum_dict,
                               hist_u_same_part_correctly_sum_dict,
                               hist_u_same_content_id_correctly_sum_dict,
                               last_u_last_incorrect_timestamp_dict,
                               hist_u_last_incorrect_cnt_dict,
                               hist_u_lag_time_sum_dict,
                               global_avg_q_time_dict, prior_question_elapsed_time_mean, global_content_cnt_dict,
                               update=False)
        test_df[TARGET] = model.predict(test_df[FEATURES])
        set_predict(test_df[['row_id', TARGET]])

    print('Job Done')


# %%

train, valid, questions_df, prior_question_elapsed_time_mean, features_dicts, global_dicts = read_and_preprocess(
    feature_engineering=True)

# %%

train_data = pd.read_csv('../input/riiid-test-answer-prediction/train.csv',
                         nrows=None,
                         dtype={
                             'row_id': 'int64',
                             'timestamp': 'int64',
                             'user_id': 'int32',
                             'content_id': 'int16',
                             'content_type_id': 'int8',
                             'task_container_id': 'int16',
                             'user_answer': 'int8',
                             'answered_correctly': 'int8',
                             'prior_question_elapsed_time': 'float32',
                             'prior_question_had_explanation': 'boolean'
                         }
                         )
all_user_ids = train_data['user_id'].unique()
os.makedirs('/root/hist_caches', exist_ok='ignore')
for user_id in tqdm(all_user_ids):
    tmp = {
        'last_u_content_id_dict': features_dicts['last_u_content_id_dict'][user_id],
        'last_u_container_id_dict': features_dicts['last_u_container_id_dict'][user_id],
        'hist_u_answered_correctly_cnt_dict': features_dicts['hist_u_answered_correctly_cnt_dict'][user_id],
        'hist_u_elapsed_time_sum_dict': features_dicts['hist_u_elapsed_time_sum_dict'][user_id],
        'hist_u_explanation_sum_dict': features_dicts['hist_u_explanation_sum_dict'][user_id],
        'hist_u_same_part_correctly_cnt_dict': features_dicts['hist_u_same_part_correctly_cnt_dict'][user_id],
        'hist_u_same_content_id_correctly_cnt_dict': features_dicts['hist_u_same_content_id_correctly_cnt_dict'][
            user_id],
        'timestamp_u': features_dicts['timestamp_u'][user_id],
        'hist_u_answered_correctly_sum_dict': features_dicts['hist_u_answered_correctly_sum_dict'][user_id],
        'hist_u_score_sum_dict': features_dicts['hist_u_score_sum_dict'][user_id],
        'hist_u_same_part_correctly_sum_dict': features_dicts['hist_u_same_part_correctly_sum_dict'][user_id],
        'hist_u_same_content_id_correctly_sum_dict': features_dicts['hist_u_same_content_id_correctly_sum_dict'][
            user_id],
        'last_u_last_incorrect_timestamp_dict': features_dicts['last_u_last_incorrect_timestamp_dict'][user_id],
        'hist_u_last_incorrect_cnt_dict': features_dicts['hist_u_last_incorrect_cnt_dict'][user_id],
        'hist_u_lag_time_sum_dict': features_dicts['hist_u_lag_time_sum_dict'][user_id],
    }
    pd.to_pickle(tmp, '/root/hist_caches/{}.pkl'.format(user_id))

# %%

TARGET, FEATURES, model = train_and_evaluate(train, valid, feature_engineering=True)

# %%

pd.to_pickle([TARGET, FEATURES, model, questions_df, prior_question_elapsed_time_mean, features_dicts, global_dicts],
             SVAE_NAME, protocol=4)

# %%

TARGET, FEATURES, model, questions_df, prior_question_elapsed_time_mean, features_dicts, global_dicts = pd.read_pickle(
    'meta.test.pkl')
