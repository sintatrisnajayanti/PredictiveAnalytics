# -*- coding: utf-8 -*-
"""ProyekPertamaMLTerapan.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11GvPvbP3hqx_fbgw6zaTOBts1kgaHwKz

#**PROYEK PERTAMA MACHINE LEARNING TERAPAN**
##"Prediksi Biaya Asuransi Kesehatan"

Oleh: 
Anak Agung Sinta Trisnajayanti, 
sintatrisnajayanti@gmail.com, 
Universitas Udayana.
"""

!pip install opendatasets

import opendatasets as od

dataset_url= 'https://www.kaggle.com/noordeen/insurance-premium-prediction'
od.download('https://www.kaggle.com/noordeen/insurance-premium-prediction')

"""Menggunakan opendatasets untuk mengambil data dari kaggle

#**Import Data**
Import library yang digunakan
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline
import seaborn as sns

"""Kemudian mengambil dan membaca dataset dengan pd.read.csv"""

data_asuransi = pd.read_csv('/content/insurance-premium-prediction/insurance.csv')
data_asuransi.head()

"""Untuk melihat jumlah data pada dataset"""

data_asuransi.shape

"""#**Data Information**

*  age: Usia dari Nasabah 
*  sex: Jenis kelamin dari nasabah
      * female: Perempuan
      * male : Laki- Laki 
*   bmi: Indeks massa tubuh dari nasabah. Memberikan pemahaman tentang tubuh, bobot yang relatif tinggi atau rendah
*   children: Jumlah anak yang ditanggung oleh asuransi kesehatan  (jumlah tanggungan
*   smoker: apakah nasabah perokok atau tidak
      * yes: perokok
      * no: tidak perokok
*   region: Daerah perumahan dari nasabah
      * southeast: Tenggara
      * southwest: Barat daya
      * northwest: Barat laut
      * northeast: Timur laut
*	expenses: biaya yang dibayarkan nasabah ke perusahaan asuransi

Mengecek nilai null dan melihat tipe datanya. Dimana jumlah fitur numerik adalah 4 dan fitur kategorikal adalah 3
"""

data_asuransi.info()

"""Untuk mengecek data atau informasi statik dan melihat apakah terdapat nilai 0"""

data_asuransi.describe()

"""Setelah melihat pada tabel ternyata terdapat nilai 0 pada fitur atau kolom children. Akan tetapi nilai 0 pada children tidak merupakan missing value karena kemungkinan nasabah memang tidak memiliki anak. Maka dari itu kita hanya melihat berapa jumlah data bernilai 0 pada kolom children dan tidak melakukan drop kolom"""

children = (data_asuransi.children == 0).sum()
 
print("Nilai 0 di kolom children ada: ", children)

data_asuransi.loc[(data_asuransi['children']==0)]

"""Disini bisa dilihat jumlah dari data masih sama seperti awal"""

data_asuransi.shape

"""#**Data Analysis**
Sekarang memisahkan fitur numerik dan fitur kategorikal
"""

numerical_features = ['age', 'bmi', 'children', 'expenses']
categorical_features = ['sex', 'smoker', 'region']

"""Mengecek outlier fitur numerik menggunakan boxplot"""

for col in numerical_features :
  sns.boxplot(x=data_asuransi[col])
  plt.show()

"""Pada fitur numerikal diatas terdapat beberapa outliers. Sekarang kita gunakan IQR method untuk mengatasi hal tersebut"""

Q1 = data_asuransi.quantile(0.20)
Q3 = data_asuransi.quantile(0.80)
IQR=Q3-Q1
data_asuransi=data_asuransi[~((data_asuransi<(Q1-1.5*IQR))|(data_asuransi>(Q3+1.5*IQR))).any(axis=1)]

data_asuransi.shape

data_asuransi.head()

"""Sekarang kita analisis data pada fitur kategorikal (Univariate) menggunakan countplot"""

feature = categorical_features[0]
count = data_asuransi[feature].value_counts()
percent = 100*data_asuransi[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

"""Gambar diatas menunjukkan persentase perempuan dan laki- laki sama pada data nasabah"""

feature = categorical_features[1]
count = data_asuransi[feature].value_counts()
percent = 100*data_asuransi[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

"""Kemudian persentase dari jumlah nasabah yang tidak perokok lebih besar """

feature = categorical_features[2]
count = data_asuransi[feature].value_counts()
percent = 100*data_asuransi[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

"""Disini juga dapat dilihat yang mendominasi untuk daerah yaitu southeast

Untuk melihat visualisasi data dari fitur numerik (Univariate) kali ini menggunakan hist
"""

data_asuransi.hist(bins=50, figsize=(15,10))
plt.show()

"""Melihat hubungan antara semua fitur ditampilkan pada visualisasi data dengan heatmap"""

sns.heatmap(data_asuransi.corr(), cmap = 'Wistia', annot= True)

"""Disini untuk melihat lebih jelas mengenai keterhubungan setiap fiturnya saya menerapkan Multivariate pada fitur numerik dengan hue smoker dan sex untuk membedakan warna sebagai kategorinya"""

sns.pairplot(data_asuransi,hue='smoker')

sns.pairplot(data_asuransi,hue='sex')

"""Dari kedua gambar di atas bisa dilihat bahwa perokok relatif muda yaitu dibawah 30 tahun, nasabah memiliki sedikit anak, biaya asuransi lebih tinggi yang perokok ini memiliki kekuatan dalam prediksi. Kemudian ada hubungan juga antara expenses-bmi, dan expenses-age

Melihat keterhubungan antara expenses-bmi, dan expenses-age menggunakan Implot
"""

sns.lmplot(x='age', y='expenses', data=data_asuransi, hue='smoker')
sns.lmplot(x='bmi', y='expenses', data=data_asuransi, hue='smoker')

"""#**Data Preparation**
Pada data preparation ini saya menggunakan One-Hot_Encoding yaitu salah satu metode encoding yang akan merepresentasikan data bertipe kategori menjadi biner yang bernilai integer 0 dan 1
"""

for col in categorical_features:
  data_asuransi = pd.concat([data_asuransi, pd.get_dummies(data_asuransi[col], prefix=col, drop_first=True)],axis=1)
  data_asuransi = data_asuransi.drop(col,axis=1)

data_asuransi

"""Membagi data menjadi data latih dan data uji"""

from sklearn.model_selection import train_test_split
X = data_asuransi.drop(["expenses"],axis =1)
y = data_asuransi["expenses"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

"""Melakukan standarisasi"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""Melihat data X_train"""

pd.DataFrame(X_train).head()

"""Melihat data y_train"""

pd.DataFrame(y_train).head()

"""#**Modeling**
Import library
"""

from math import sqrt 
from sklearn.model_selection import cross_val_predict  
from sklearn.metrics import r2_score, mean_squared_error   
from sklearn.svm import SVR  
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

"""Membuat model summary yang nantinya akan digunakan untuk membandingkan model/solusi yang akan digunakan. Dimana Untuk membandingkan ketiga model ini akan dilakukan penghitungan nilai dari Training Accuracy, Testing Accuracy, RMSE Training Data, RMSE Testing Data, dan Accuracy dari prediksi"""

def model_summary(model, model_name, cvn=20):
    print(model_name)
    y_pred_model_train = model.predict(X_train)
    y_pred_model_test = model.predict(X_test)
    accuracy_model_train = r2_score(y_train, y_pred_model_train)
    print("Training Accuracy: ", accuracy_model_train)
    accuracy_model_test = r2_score(y_test, y_pred_model_test)
    print("Testing Accuracy: ", accuracy_model_test)
    RMSE_model_train = sqrt(mean_squared_error(y_train, y_pred_model_train))
    print("RMSE Training Data: ", RMSE_model_train)
    RMSE_model_test = sqrt(mean_squared_error(y_test, y_pred_model_test))
    print("RMSE Testing Data: ", RMSE_model_test)

    y_pred_cv_model = cross_val_predict(model, X, y, cv=cvn)
    accuracy_cv_model = r2_score(y, y_pred_cv_model)
    print("Accuracy untuk", cvn,"- Prediksi: ", accuracy_cv_model)

"""Menjalankan semua model hingga mencapai performa yang baik"""

support_vector_reg = SVR(gamma="auto", kernel="linear", C=1000)  
support_vector_reg.fit(X_train, y_train)  
model_summary(support_vector_reg, "Support_Vector_Regression")

decision_tree_reg = DecisionTreeRegressor(max_depth=3, random_state=5)  
decision_tree_reg.fit(X_train, y_train) 
model_summary(decision_tree_reg, "Decision_Tree_Regression")

random_forest_reg = RandomForestRegressor(n_estimators=300, max_depth=3, random_state=5)  
random_forest_reg.fit(X_train, y_train) 
model_summary(random_forest_reg, "Random_Forest_Regression")

"""Dari sini kita dapat menyimpulkan bahwa model yang lebih akurat dalam memprediksi harga/biaya asuransi pada proyek ini adalah menggunakan model Random Forest Regression dengan memiliki accuracy prediksi tertinggi yaitu 81% dan nilai RMSE terendah. """