import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

sales = pd.read_csv("data/vgsales.csv")
print(sales.info())

# # Drop data that is not needed
drop = ["Rank", "Name", "Platform", "Year", "Genre", "Global_Sales"]
sales_droped = sales.drop(labels=drop, axis = 1)
print(sales_droped.info())

EA = sales_droped[sales_droped.Publisher == 'Electronic Arts']
ACT = sales_droped[sales_droped.Publisher == 'Activision']
Total = EA.append(ACT)
print(EA.info())
print(ACT.info())
print(Total.info())


# drop = ["Rank", "Name", "Platform", "Year", "Genre", "Global_Sales"]
# sales_droped = sales.drop(labels=drop, axis = 1)
# print(sales_droped.info())
# print(sales_droped.describe())

# sales_droped.hist(bins=20)
# plt.show()

# corr_matrix = titanic_filled.corr()
# corr_relationships = corr_matrix["Survived"].sort_values(ascending = False)
# print(corr_relationships)
#
# pd.plotting.scatter_matrix(titanic_filled[corr_relationships.index])
# plt.show()

# titanic_labels = titanic_filled["Survived"]
# titanic_filled_data = titanic_filled.drop(labels="Survived", axis = 1)
#
# num_pipeline = Pipeline([
#     ('std_scaler', StandardScaler())
# ])
#
# titanic_data = num_pipeline.fit_transform(titanic_filled_data)
#
# from sklearn.decomposition import PCA
# pca = PCA(n_components = 2)
# pca.fit(titanic_data)
# PCAX = pca.transform(titanic_data)
# print(pca.explained_variance_ratio_.sum())
# plt.scatter(PCAX[:, 0], PCAX[:, 1], c = titanic_labels)
# plt.show()
#
# from sklearn.manifold import TSNE
# tsne = TSNE(n_components=2, verbose=0, perplexity=100, n_iter=1000)
# tsne_results = tsne.fit_transform(titanic_data)
# plt.scatter(PCAX[:, 0], PCAX[:, 1], c=titanic_labels)
# plt.show()
