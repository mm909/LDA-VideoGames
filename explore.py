import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

sales = pd.read_csv("data/vgsales.csv")
print(sales.info())
# print(sales.Publisher.value_counts())

# Electronic Arts                 1351
# Activision                       975
# Namco Bandai Games               932
# Ubisoft                          921
# Konami Digital Entertainment     832

# Drop data that is not needed
drop = ["Name", "Year", "Platform", "Genre", "Global_Sales"]
sales = sales.drop(labels=drop, axis = 1)
# print(sales.info())

Class1 = sales[sales.Publisher == 'Nintendo']
Class2 = sales[sales.Publisher == 'Ubisoft']
Total = Class1.append(Class2)
# print(Class1.info())
# print(Class2.info())
print(Total.info())

le = LabelEncoder()
le.fit(Total.Publisher.values)
Total.Publisher = le.transform(Total.Publisher.values)

# Total.hist(bins=20)
# plt.show()

corr_matrix = Total.corr()
corr_relationships = corr_matrix["Publisher"].sort_values(ascending = False)
print(corr_relationships)

# pd.plotting.scatter_matrix(Total[corr_relationships.index])
# plt.show()
