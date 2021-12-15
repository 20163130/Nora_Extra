#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Se importan la librerias a utilizar
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt


# In[3]:


#Se importan la librerias a utilizar
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt


# In[4]:


#Importamos los datos de la misma librería de scikit-learn
boston = datasets.load_boston()


# In[5]:


#Verifico la información contenida en el dataset
print('Información en el dataset:')
print(boston.keys())


# In[6]:


#Verifico las características del dataset
print('Características del dataset:')
print(boston.DESCR)


# In[7]:


#Verifico la cantidad de datos que hay en los dataset
print('Cantidad de datos:')
print(boston.data.shape)


# In[8]:


#Verifico la información de las columnas
print('Nombres columnas:')
print(boston.feature_names)


# In[9]:


#Seleccionamos solamente la columna 5 del dataset
X = boston.data[:, np.newaxis, 5]


# In[10]:


#Defino los datos correspondientes a las etiquetas
y = boston.target


# In[11]:


#Graficamos los datos correspondientes
plt.scatter(X, y)
plt.xlabel('Número de habitaciones')
plt.ylabel('Valor medio')
plt.show()


# In[12]:


from sklearn.model_selection import train_test_split
#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[13]:


#Defino el algoritmo a utilizar
lr = linear_model.LinearRegression()


# In[14]:


#Entreno el modelo
lr.fit(X_train, y_train)


# In[15]:


#Realizo una predicción
Y_pred = lr.predict(X_test)


# In[16]:


#Graficamos los datos junto con el modelo
plt.scatter(X_test, y_test)
plt.plot(X_test, Y_pred, color='red', linewidth=3)
plt.title('Regresión Lineal Simple')
plt.xlabel('Número de habitaciones')
plt.ylabel('Valor medio')
plt.show()


# In[17]:


print('DATOS DEL MODELO REGRESIÓN LINEAL SIMPLE')
print()
print('Valor de la pendiente o coeficiente "a":')
print(lr.coef_)
print('Valor de la intersección o coeficiente "b":')
print(lr.intercept_)
print()
print('La ecuación del modelo es igual a:')
print('y = ', lr.coef_, 'x ', lr.intercept_)


# In[18]:


print('Precisión del modelo:')
print(lr.score(X_train, y_train))


# In[19]:


#Seleccionamos solamente la columna 6 del dataset
X = boston.data[:, np.newaxis, 6]


# In[20]:


#Defino los datos correspondientes a las etiquetas
y = boston.target


# In[21]:


#Graficamos los datos correspondientes
plt.scatter(X, y)
plt.xlabel('Antiguedad')
plt.ylabel('Valor medio')
plt.show()


# In[22]:


from sklearn.model_selection import train_test_split
#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[23]:


#Defino el algoritmo a utilizar
lr = linear_model.LinearRegression()


# In[24]:


#Entreno el modelo
lr.fit(X_train, y_train)


# In[25]:


#Realizo una predicción
Y_pred = lr.predict(X_test)


# In[26]:


#Defino los datos correspondientes a las etiquetas
y = boston.target


# In[28]:


#Graficamos los datos junto con el modelo
plt.scatter(X_test, y_test)
plt.plot(X_test, Y_pred, color='red', linewidth=3)
plt.title('Regresión Lineal Simple')
plt.xlabel('Antiguedad')
plt.ylabel('Valor medio')
plt.show()


# In[29]:


print('DATOS DEL MODELO REGRESIÓN LINEAL SIMPLE')
print()
print('Valor de la pendiente o coeficiente "a":')
print(lr.coef_)
print('Valor de la intersección o coeficiente "b":')
print(lr.intercept_)
print()
print('La ecuación del modelo es igual a:')
print('y = ', lr.coef_, 'x ', lr.intercept_)


# In[30]:


print('Precisión del modelo:')
print(lr.score(X_train, y_train))


# In[ ]:




