#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Se importan la librerias a utilizar
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt


# In[2]:


#Se importan la librerias a utilizar
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt


# In[6]:


#Importamos los datos de la misma librería de scikit-learn
boston = datasets.load_boston()


# In[7]:


#Verifico la información contenida en el dataset
print('Información en el dataset:')
print(boston.keys())


# In[8]:


#Verifico las características del dataset
print('Características del dataset:')
print(boston.DESCR)


# In[9]:


#Verifico la cantidad de datos que hay en los dataset
print('Cantidad de datos:')
print(boston.data.shape)


# In[10]:


#Verifico la información de las columnas
print('Nombres columnas:')
print(boston.feature_names)


# In[11]:


#Seleccionamos solamente la columna 5 del dataset
X = boston.data[:, np.newaxis, 5]


# In[12]:


#Defino los datos correspondientes a las etiquetas
y = boston.target


# In[13]:


#Graficamos los datos correspondientes
plt.scatter(X, y)
plt.xlabel('Número de habitaciones')
plt.ylabel('Valor medio')
plt.show()


# In[14]:


from sklearn.model_selection import train_test_split
#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[15]:


#Defino el algoritmo a utilizar
lr = linear_model.LinearRegression()


# In[16]:


#Entreno el modelo
lr.fit(X_train, y_train)


# In[17]:


#Realizo una predicción
Y_pred = lr.predict(X_test)


# In[18]:


#Graficamos los datos junto con el modelo
plt.scatter(X_test, y_test)
plt.plot(X_test, Y_pred, color='red', linewidth=3)
plt.title('Regresión Lineal Simple')
plt.xlabel('Número de habitaciones')
plt.ylabel('Valor medio')
plt.show()


# In[19]:


print('DATOS DEL MODELO REGRESIÓN LINEAL SIMPLE')
print()
print('Valor de la pendiente o coeficiente "a":')
print(lr.coef_)
print('Valor de la intersección o coeficiente "b":')
print(lr.intercept_)
print()
print('La ecuación del modelo es igual a:')
print('y = ', lr.coef_, 'x ', lr.intercept_)


# In[20]:


print('Precisión del modelo:')
print(lr.score(X_train, y_train))


# In[ ]:




