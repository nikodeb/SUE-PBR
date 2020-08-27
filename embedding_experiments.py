import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.manifold import TSNE, SpectralEmbedding
import matplotlib.pyplot  as plt

loc_emb_file = 'experiments/randomtests/t_sse_del_2020-06-26_0/models/user_embedding.npy'
u_emb = np.load(loc_emb_file)
print('User embedding size:', str(u_emb.shape))

# print('Fitting PCA')
# pca = PCA(n_components=2, svd_solver='full')
# dim_red = pca.fit_transform(u_emb)

tsne = TSNE(n_components=2, init='random', random_state=0, perplexity=50)
dim_red = tsne.fit_transform(u_emb)

# se = SpectralEmbedding(n_components=2, affinity='rbf', random_state=0, gamma=0.01)
# dim_red = se.fit_transform(u_emb)

x = dim_red[:,0]
y = dim_red[:,1]
plt.plot(x, y, ".", markersize=2)
plt.show()

print('Visualising')
#
# def dump_variances(x):
#     mean = np.mean(x, axis=0)
#     cov = np.cov(x, rowvar=False)
#     var = np.var(x, axis=0)
#     u_emb_stats = {'mean': mean,
#                    'cov': cov,
#                    'var': var}
#     pickle.dump(u_emb_stats, open("u_emb_stats.p", "wb"))
#     print(x)
#
# loc_emb_file = 'experiments/randomtests/t_sse_del_2020-06-26_1/models/user_embedding.npy'

