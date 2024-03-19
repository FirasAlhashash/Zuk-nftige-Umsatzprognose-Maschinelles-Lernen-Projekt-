#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importieren der erforderlichen Python-Bibliotheken und des Datensatzes

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/advertising.csv")
print(data.head())


# In[2]:


#Schauen wir an, ob dieser Datensatz Nullwerte enthält oder nicht

print(data.isnull().sum())


# In[5]:


#die Beziehung zwischen dem Betrag, der für Werbung im Fernsehen ausgegeben wird, und den verkauften Einheiten visualisieren

import plotly.express as px
import plotly.graph_objects as go
figure = px.scatter(data_frame = data, x="Sales", y="TV", size="TV", trendline="ols")
figure.show()


# In[6]:


#Visualisieren das Verhältnis zwischen den Werbeausgaben für Zeitungen und den verkauften Einheiten


figure = px.scatter(data_frame = data, x="Sales",
                    y="Newspaper", size="Newspaper", trendline="ols")
figure.show()


# In[7]:


#Visualisieren das Verhältnis zwischen den Ausgaben für Werbung im Radio und den verkauften Einheiten:


figure = px.scatter(data_frame = data, x="Sales",
                    y="Radio", size="Radio", trendline="ols")
figure.show()


# In[8]:


#die Korrelation aller Spalten mit der Verkaufsspalte

correlation = data.corr()
print(correlation["Sales"].sort_values(ascending=False))


# In[21]:


#Trainieren ein maschinelles Lernmodell, um die zukünftigen Verkäufe eines Produkts vorherzusagen
# Assuming 'data' is a Pandas DataFrame

x = np.array(data.drop(["Sales"], axis=1))
y = np.array(data["Sales"])

# Splitting the dataset into the Training set and Test set
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)


# In[22]:


model = LinearRegression()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))


# In[26]:


# Werte in das Modell entsprechend den Funktionen, mit denen wir es trainiert haben, und prognostizieren, wie viele Einheiten des Produktsbasierend auf dem für seine Werbung auf verschiedenen Plattformen ausgegebenen Betrag verkauft werden können
#features = [[TV, Radio, Newspaper]]

features = np.array([[230.1, 37.8, 69.2]])
print(model.predict(features))


# In[ ]:





# In[ ]:




