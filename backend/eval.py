# generate 200 song evaluation set based on audio feature clustering and release era
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from preprocess import preprocess
import singlestage, twostage

# EDA
def cluster_exploration(df:pd.DataFrame):
    # prepping audio features
    print('\n—— audio feature cluster exploration ——\n')
    audio_df = df.copy()[[
        'track_id', 'track_name', 'artist_names', 'main_genres',
        'danceability', 'energy', 'loudness', 'mode', 'speechiness',
        'acousticness', 'instrumentalness', 'valence', 'tempo'
    ]].dropna()
    scaler = StandardScaler()
    audio_scaled = scaler.fit_transform(audio_df[[
        'danceability', 'energy', 'loudness', 'mode', 'speechiness',
        'acousticness', 'instrumentalness', 'valence', 'tempo'
    ]])    

    # clustering audio features; finding optimal k centroids using silhouette scores
    scores={}
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42
    ) # t-SNE instead of PCA for small nonlinear data

    for k in range(5, 15):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        audio_df['cluster'] = kmeans.fit_predict(audio_scaled)
        scores[k] = silhouette_score(audio_scaled, audio_df['cluster'])

        # graphing clusters in 2d
        audio_2d = tsne.fit_transform(audio_scaled)

        plt.figure(figsize=(10,8))
        plt.scatter(
            audio_2d[:,0], # column vector
            audio_2d[:,1],
            c=audio_df["cluster"],
            cmap="tab20",
            s=10
        )
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.title("Audio Feature Clusters (t-SNE)")
        os.makedirs("./backend/charts", exist_ok=True)
        plt.savefig("./backend/charts/audio-clusters.png", dpi=300, bbox_inches="tight")

        # cluster_profiles = (
        #     audio_df.drop(columns=['track_id','track_name','artist_names','main_genres',])
        #     .groupby("cluster")
        #     .mean()
        # )
        # print(cluster_profiles)

        # if k < 7:
        #     print(f"\n*======* {k} Clusters *======*")
        #     for c in range(k):
        #         print(f"\n=== Cluster {c} ===")
        #         print(
        #             audio_df[audio_df.cluster == c][
        #                 ["track_name","artist_names","main_genres"]
        #             ].sample(5, random_state=42)
        #         )
    print('Silhouette Scores:', scores)
    plt.figure(figsize=(10,8))
    plt.plot(
        range(5, 15),
        list(scores.values()),
    )
    plt.xlabel("Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score v Clusters (k)")
    os.makedirs("./backend/charts", exist_ok=True)
    plt.savefig("./backend/charts/silhouette-v-k.png", dpi=300, bbox_inches="tight")

def create_query_set(df:pd.DataFrame, k:int=10, songs:int=200, save:bool=True, output_dir:str='./'):

    eval_df = df.copy().dropna()
    eval_df['release_yr'] = eval_df['release_date'].str.extract(r'^(\d{4})-?').astype(int)

    audio_scaled = StandardScaler().fit_transform(eval_df[[
        'danceability', 'energy', 'loudness', 'mode', 'speechiness',
        'acousticness', 'instrumentalness', 'valence', 'tempo'
    ]]) 
    eval_df['cluster'] = (
        KMeans(n_clusters=k, random_state=42, n_init='auto')
        .fit_predict(audio_scaled)
    )

    # sample clusters by release era strata proportions
    eval_df["release_era"] = pd.cut(
            eval_df["release_yr"],
            bins=[0, 1999, 2009, 2019, float("inf")],
            labels=["<2000", "2000-2009", "2010-2019", "2020+"]
        )
    strata = eval_df.groupby(['cluster', 'release_era'], observed=True).size()
    target_vals = strata / len(eval_df) * songs
    n = target_vals.astype(int)

    remaining = songs - n.sum()
    n.loc[(target_vals - n).nlargest(remaining).index] += 1 # round up the remaining largest buckets

    eval_set = pd.concat([
        eval_df[
            (eval_df['cluster'] == cluster) &
            (eval_df['release_era'] == era)
        ].sample(n=count, random_state=42)
        for (cluster, era), count in n.items() # each cluster x era strata
        if count > 0
    ])[['track_id','track_name','artist_names','main_genres', 'release_yr','release_era','cluster']]

    eval_set = eval_set.reset_index(drop=True)
    if save:
        eval_set.to_csv(os.path.join(output_dir, 'data', 'eval-queries.csv'))

    return eval_set

def create_recs_set(df:pd.DataFrame, eval_set:pd.DataFrame, save:bool=True, output_dir:str='./'):
    precomputed = preprocess(df)
    rows = []

    for _, query in eval_set.iterrows():
        track = query['track_id']
        single_recs = singlestage.recommend(df, track, 10, precomputed)
        two_recs = twostage.recommend(df, track, 100, 10, precomputed)

        for model_name, recs in [('singlestage', single_recs), ('twostage', two_recs)]:
            if isinstance(recs, str): # error guard
                continue
            for rank, (idx, rec) in enumerate(recs.iterrows(), 1):
                if rec['track_id'] == track:  # skip if rec is the same song as query
                    continue
                rows.append({
                    'query_id': track,
                    'query_name': query['track_name'],
                    'query_artist': query['artist_names'],
                    'rec_rank': rank,
                    'rec_id': rec['track_id'],
                    'rec_name': rec['track_name'],
                    'rec_artist': rec['artist_names'],
                    'model': model_name,
                    'relevance_score': '' # manual scoring for now
                })

    eval_recs = pd.DataFrame(rows)
    shuffled = []
    for qid, group in eval_recs.groupby('query_id', sort=False):
        shuffled.append(group.sample(frac=1, random_state=42))
    eval_recs = pd.concat(shuffled).reset_index(drop=True)

    if save:
        eval_recs.to_csv(os.path.join(output_dir, 'data', 'eval-recs.csv'), index=False)
        print(f'Saved {len(eval_recs)} recommendations to eval-recs.csv')

    return eval_recs

def evaluate_model(eval_recs: pd.DataFrame, model_name: str):
    scored = eval_recs[
        (eval_recs['model'] == model_name) &
        (eval_recs['relevance_score'].notna()) &
        (eval_recs['relevance_score'] != '')
    ].copy()
    scored['relevance_score'] = scored['relevance_score'].astype(float)
    scored = scored.sort_values(['query_id', 'rec_rank'])

    scores_by_query = (
        scored
        .groupby('query_id')['relevance_score']
        .apply(list)
    )
    ndcgs, precisions, mean_rels = [], [], []

    for query_id, rels in scores_by_query.items():
        if len(rels) != 10:
            raise ValueError(
                f"Query {query_id} has {len(rels)} recommendations, expected 10"
            )
        rels = np.array(rels)

        mean_rels.append(rels.mean())
        precisions.append(np.mean(rels >= 2))

        # ndcg calcs
        discounts = np.log2(np.arange(2, 12))
        dcg = np.sum(rels / discounts)

        ideal_rels = np.sort(rels)[::-1]
        idcg = np.sum(ideal_rels / discounts)

        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    results = {
        'model': model_name,
        'ndcg@10': np.mean(ndcgs),
        'precision@10': np.mean(precisions),
        'mean_relevance@10': np.mean(mean_rels),
        'n_queries': len(ndcgs)
    }

    print(f"\n—— {model_name} evaluation ——")
    print(f"  NDCG@10:           {results['ndcg@10']:.4f}")
    print(f"  Precision@10:      {results['precision@10']:.4f}")
    print(f"  Mean Relevance@10: {results['mean_relevance@10']:.4f}")
    print(f"  Queries evaluated: {results['n_queries']}")

    return results

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'data', 'top-10k-spotify-songs-2025-07-detailed.csv')
df = pd.read_csv(file_path)

# eval_set = create_query_set(df=df, output_dir=script_dir)
eval_set = pd.read_csv(os.path.join(script_dir, 'data', 'eval-queries.csv'), index_col=0)
create_recs_set(df=df, eval_set=eval_set, output_dir=script_dir)