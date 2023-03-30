import pickle
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.spatial.distance import cdist

from ann.recommender import AnnoyRecommender
#from config.config import recommender_conf, path_conf
import numpy as np

path = '/home/mikhail/Projects/your-second-recsys/lecture_4/data/'

def load_object(path):
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    return obj


def read_vectors_and_mappings(
    user_vectors_path, item_vectors_path, user_map_path, item_map_path
):
    return (
        load_object(user_vectors_path),
        load_object(item_vectors_path),
        load_object(user_map_path),
        load_object(item_map_path),
    )

def create_user_mappings_remap():
    path = '/home/mikhail/Projects/your-second-recsys/lecture_4/data/user_mappings_remap.pkl'

    keys = list(user_map.keys())
    map = {user_map[key]: key for key in keys}

    with open(path, 'wb') as handle:
        pickle.dump(map, handle, protocol=pickle.HIGHEST_PROTOCOL)


user_vectors, item_vectors, user_map, item_map = read_vectors_and_mappings(path+'user_vectors.pkl',
                                                                           path+'item_vectors.pkl',
                                                                           path+'user_mappings.pkl',
                                                                           path+'item_mappings.pkl')

user_mappings_remap = load_object(path+'user_mappings_remap.pkl')


recommender = AnnoyRecommender(
        item_vectors=item_vectors, # пользователи для которых строится рекомендация
        user_vectors=user_vectors, # пользователи которых рекомендуют
        user_id_user_index_id_mapping=user_map,
        item_id_item_index_id_mapping=item_map,
        user_mappings_remap=user_mappings_remap,
        sim_function=lambda x, y: 1 - cdist(x, y, metric='cosine'),
        top_k=50,
        dim=32,
        metric="angular",
        n_trees=50,
        n_jobs=-1,
        search_k=-1,
        n_neighbors=200,
    )
# user_id = 14064
user_id = 9506 # id пользователя из списка items, для которого нужно порекомендовать пользователей из списка users
k = 20 # кол-во пользователей, которые будут рекомендованы

recommendations = recommender.recommend_single_user(user_id, k)
print(recommendations)


#
# class Response(BaseModel):
#     user_id: int
#     item_ids: List[int]
#
#
# class Request(BaseModel):
#     user_id: int
#     item_whitelist: List[int]
#
#
# app = FastAPI(docs_url="/docs", redoc_url="/redoc")
#
#
# @app.on_event("startup")
# async def startup():
#     user_vectors, item_vectors, user_map, item_map = read_vectors_and_mappings(**path_conf)
#     app.state.recommender = AnnoyRecommender(
#         item_vectors=item_vectors,
#         user_vectors=user_vectors,
#         user_id_user_index_id_mapping=user_map,
#         item_id_item_index_id_mapping=item_map,
#         sim_function=lambda x, y: 1 - cdist(x, y, metric='cosine'),
#         **recommender_conf
#     )
#     app.state.recommender.fit()
#
# @app.post("/api/v1/recommend_for_user", response_model=Response)
# async def recommend_for_user(request: Request):
#     try:
#         recommendations = app.state.recommender.recommend_single_user(
#             request.user_id, request.item_whitelist
#         )
#     except KeyError:
#         raise HTTPException(status_code=404, detail="Item or user not found")
#     return Response(user_id=request.user_id, item_ids=recommendations)
#
# @app.post("/api/v1/recommend_bruteforce", response_model=Response)
# async def recommend_bruteforce(request: Request):
#     recommendations = app.state.recommender.recommend_bruteforce_single_user(
#         request.user_id, request.item_whitelist
#     )
#     return Response(user_id=request.user_id, item_ids=recommendations)
