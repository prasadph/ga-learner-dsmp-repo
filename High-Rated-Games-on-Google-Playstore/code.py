# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





#Code starts here
data = pd.read_csv(path)
data.Rating.hist()
data = data[data['Rating']<=5]
data.Rating.hist()
#Code ends here


# --------------
# code starts here
total_null = data.isnull().sum()
percent_null = total_null/data.isnull().count()
missing_data = pd.concat([total_null,percent_null], keys=['Total', 'Percent'],axis=1)
print(missing_data)
data.dropna(inplace=True)
total_null_1 = data.isnull().sum()
percent_null_1 = total_null_1/data.isnull().count()
missing_data_1 = pd.concat([total_null_1,percent_null_1], keys=['Total', 'Percent'],axis=1)
print(missing_data_1)
# code ends here


# --------------

#Code starts here
catplot = sns.catplot(x="Category",y="Rating",data=data, kind="box", height = 10)
catplot.set_titles("Rating vs Category [BoxPlot]")
catplot.set_xticklabels(rotation=90)

#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here

data['Installs'] = data['Installs'].str.replace(',', '').str.replace('+','').astype(int)
le = LabelEncoder()
data['Installs'] = le.fit_transform(data['Installs'])
regplot = sns.regplot(x="Installs", y="Rating",data=data)
regplot.set_title("Rating vs Installs [RegPlot]")
# regplot.set_
#Code ends here


# --------------
#Code starts here
data.Price = data.Price.str.replace('$','').astype('float')

pr_regplot = sns.regplot(data=data, x='Price', y="Rating")
pr_regplot.set_title("Rating vs Price [RegPlot]")
#Code ends here


# --------------

#Code starts here

data.Genres = data.Genres.apply(lambda x: x.split(';')[0])

gr_mean = data[['Genres','Rating']].groupby(by="Genres",as_index=False).mean()
# print(gr_mean.describe())
gr_mean.sort_values("Rating",inplace=True)
print(gr_mean.iloc[0])
print(gr_mean.iloc[-1])
#Code ends here


# --------------

#Code starts here
data['Last Updated'] = pd.to_datetime(data['Last Updated'])
max_date = data['Last Updated'].max()
data['Last Updated Days'] = (max_date -data['Last Updated']).dt.days
data['Last Updated Days'].head() 



#Code ends here


