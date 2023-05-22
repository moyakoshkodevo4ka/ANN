from annoy import AnnoyIndex

class ItemRecommender:
    def __init__(self, num_features, metric='angular'):
        self.num_features = num_features
        self.metric = metric
        self.item_index = AnnoyIndex(num_features, metric)

    def fit(self, item_vectors, trees):
        for i, vector in enumerate(item_vectors):
            self.item_index.add_item(i, vector)
        self.item_index.build(trees)

    def recommend_items(self, purchased_items, num_recommendations=5):
        similar_items = set()
        for item_id in purchased_items:
            similar_items.update(self.item_index.get_nns_by_item(item_id, num_recommendations))
        recommended_items = similar_items.difference(purchased_items)
        return list(recommended_items)[:num_recommendations]

    def find_similar(self, item_id, num_recommendations):
        return self.item_index.get_nns_by_item(item_id, num_recommendations)


item_vectors = [
    [0.2, 0.5, 0.3],
    [0.8, 0.1, 0.5],
    [0.1, 0.4, 0.9],
    [0.6, 0.2, 0.7],
    [0.4, 0.6, 0.2]
]

recommender = ItemRecommender(num_features=3)
recommender.fit(item_vectors, 10)

purchased_items = [1, 3]
recommended_items = recommender.recommend_items(purchased_items, num_recommendations=3)
print(recommended_items)
