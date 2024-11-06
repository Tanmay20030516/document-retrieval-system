import pandas as pd
import numpy as np


def get_dataframe_subset(retrieved_ids, df):
    df_sub = df.set_index('ID').loc[retrieved_ids, :].reset_index()
    df_sub['Model_Sentence'] = "[TITLE] " + df_sub['Title'].str.lower() + " [ABSTRACT] " + df_sub['Abstract'].str.lower()
    return df_sub


def get_dataframe_to_display(retrieved_ids, df):
    df_sub = df.set_index('ID').loc[retrieved_ids, :].reset_index()
    df_sub['Link'] = "https://arxiv.org/pdf/" + df_sub['ID']
    df_sub = df_sub[['Title', 'Link', 'Authors', 'Abstract']]
    return df_sub


def re_rank(scores, retrieved_ids):
    retrieved_ids = np.array(retrieved_ids)
    new_rank_indices = scores.argsort()[::-1]
    re_ranked_ids = retrieved_ids[new_rank_indices]
    return re_ranked_ids.tolist()


def create_cross_encoder_input(query, retrieved_ids, df):
    res = [query]*len(retrieved_ids)
    df_sub = get_dataframe_subset(retrieved_ids, df)
    text = df_sub[['Model_Sentence']]
    text['Query'] = pd.Series(res)
    text = text[['Query', 'Model_Sentence']].copy()
    return text.to_dict('split').get('data')


def cross_encoder_rerank(model, query, retrieved_ids, df):
    cross_enc_input = create_cross_encoder_input(query, retrieved_ids, df)
    scores = model.predict(cross_enc_input)
    return scores