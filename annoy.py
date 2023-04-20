from annoy import AnnoyIndex
import numpy as np
 
 
class AnnoyRecommender:
    def init(self, n_factors=50, n_trees=100):
        self.n_factors = n_factors
        self.n_trees = n_trees
        self.index = None
        self.item_factors = None
 
    def fit(self, X, metric='euclidean'):
        self.index = AnnoyIndex(self.n_factors, metric)
 
        for i in range(X.shape[0]):
            self.index.add_item(i, X[i])
        self.index.build(self.n_trees)
 
        self.item_factors = X.copy()
 
    def recommend(self, item_id, n=10):
        similar_items = self.index.get_nns_by_item(item_id, n)
 
        return similar_items
 
 
recommender = AnnoyRecommender()
X = np.random.rand(100, 50)
 
recommender.fit(X)
similar_items = recommender.recommend(0, n=10)
 
print(similar_items)