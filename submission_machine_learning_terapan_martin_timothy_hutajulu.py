# -*- coding: utf-8 -*-
"""Submission - Machine Learning Terapan - Martin Timothy Hutajulu.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1GyCpC34298uHW2UFrgRbZYYL8l2tNJnQ

# **Prediksi Gangguan Pada Transmission Line Berdasarkan Berbagai Kondisi Tegangan dan Arus Pada Sistem 3 Phase**

## 1.Import Library yang dibutuhkan
"""

# Import library untuk menganalisis data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import library untuk Data Preparation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Import model yang dibutuhkan
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

"""## 2.Data Understanding

### 2.1. Data Loading

Untuk memahami dataset yang akan digunakan, maka diperlukan proses loading data terlebih dahulu

Dataset yang digunakan pada proyek ini yakni
https://www.kaggle.com/datasets/esathyaprakash/electrical-fault-detection-and-classification

pada dataset tersebut terdapat dua file csv yakni
1. classData.csv
2. detect_dataset.csv

Adapun dataset yang dipilih penulis hanya dataset nomor 2 yakni *classData.csv*
"""

# Membaca dataset dari detect_dataset.csv
detect_fault = pd.read_csv('/content/classData.csv')
detect_fault

"""### 2.2. Exploratory Data Analysis (EDA)

Exploratory data analysis atau sering disingkat EDA merupakan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data.
"""

# Memeriksa setiap informasi pada dataset yang digunakan
detect_fault.info()

# Melihat parameter statistik dari dataset yang digunakan
detect_fault.describe()

# Memeriksa missing value dari dataset yang digunakan
detect_fault.isnull().sum()

# Memeriksa duplikat data pada dataset yang digunakan
detect_fault.duplicated().sum()

"""#### 2.2.1. EDA - Penanganan Missing Value dan Outliers"""

# Memeriksa outliers pada kolom 'G'
sns.boxplot(x = detect_fault['G'])

# Memeriksa outliers pada kolom 'C'
sns.boxplot(x = detect_fault['C'])

# Memeriksa outliers pada kolom 'B'
sns.boxplot(x = detect_fault['B'])

# Memeriksa outliers pada kolom 'A'
sns.boxplot(x = detect_fault['A'])

# Memeriksa outliers pada kolom 'Ia'
sns.boxplot(x = detect_fault['Ia'])

# Memeriksa outliers pada kolom 'Ib'
sns.boxplot(x = detect_fault['Ib'])

# Memeriksa outliers pada kolom 'Ic'
sns.boxplot(x = detect_fault['Ic'])

# Memeriksa outliers pada kolom 'Va'
sns.boxplot(x = detect_fault['Va'])

# Memeriksa outliers pada kolom 'Vb'
sns.boxplot(x = detect_fault['Vb'])

# Memeriksa outliers pada kolom 'Vc'
sns.boxplot(x = detect_fault['Vc'])

# Mengatasi Outliers menggunakan method IQR
numeric_cols = detect_fault.select_dtypes(include=['int64', 'float64']).columns
Q1 = detect_fault[numeric_cols].quantile(0.25)
Q3 = detect_fault[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

detect_fault = detect_fault[~((detect_fault[numeric_cols] < (Q1 - 1.5 * IQR)) | (detect_fault[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
detect_fault.shape

detect_fault

# Menggabungkan Kolom G, B, C, A menjadi Kolom Output
fault_map = {
    (1.0, 0.0, 0.0, 1.0): 'LG fault (Between Phase A and Gnd)',
    (0.0, 0.0, 1.0, 1.0): 'LL fault (Between Phase A and Phase B)',
    (1.0, 0.0, 1.0, 1.0): 'LLG Fault (Between Phases A,B and ground)',
    (0.0, 1.0, 1.0, 1.0): 'LLL Fault(Between all three phases)',
    (1.0, 1.0, 1.0, 1.0): 'LLLG fault( Three phase symmetrical fault)',
    (0.0, 0.0, 0.0, 0.0): 'No Fault'
}

# membuat kolom baru bernama 'output' menggunakan fault_map untuk kolom G, B, C, A
detect_fault['Output'] = detect_fault.apply(lambda row: fault_map.get(tuple(row[['G', 'B', 'C', 'A']].values), 'Unknown Fault'), axis=1)

# Melakukan drop original kolom G, B, C, A
detect_fault.drop(columns=['G', 'B', 'C', 'A'], inplace=True)
detect_fault

"""#### 2.2.2. EDA - Unvariate Analysis"""

# Memeriksa Frekuensi kemunculan jenis jenis fault
plt.figure(figsize=(8, 8))
plt.pie(detect_fault['Output'].value_counts(), autopct='%0.2f', labels=detect_fault['Output'].value_counts().index)
plt.title('Frequency of Fault and No Fault')
plt.show()

# Memeriksa masing - masing fitur menggunakan histogram
detect_fault.hist(bins=50, figsize=(20,15))
plt.show()

"""#### 2.2.3. EDA - Multivariate Analysis"""

# Memeriksa korelasi antar kolom
numeric_cols = detect_fault.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = detect_fault[numeric_cols].corr()

plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

"""## 3.Data Preparation"""

Encoder = LabelEncoder()
detect_fault['Output'] = Encoder.fit_transform(detect_fault['Output'].values)

# Menampilkan jenis gangguan
print(Encoder.classes_)

# Menampilkan nilai numerik untuk setiap jenis gangguan
for i, class_label in enumerate(Encoder.classes_):
    print(f"{class_label} is encoded as: {i}")
    print(f"Encoded value {i} corresponds to: {Encoder.inverse_transform([i])[0]}")

# Mendefinisikan X merupakan dataset yang digunakan dan menghapus kolom 'output (s)'
X = detect_fault.drop(["Output"],axis =1)

# Mendefiniskan y merupakan dataset yang hanya terdiri dari kolom 'output (s)
y = detect_fault["Output"]

# Melakukan imbalance pada masing - masing class
smote = SMOTE()
x, Y = smote.fit_resample(X,y)

"""### 3.1. Train-test split"""

# Melakukan split dataset
X_train, X_test, y_train, y_test = train_test_split(x, Y, test_size = 0.2, random_state = 42)

# Menampilkan total masing-masing dataset yang telah di split
print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')

"""### 3.2. Standarisasi"""

# Melakukan standarisasi data
numerical_features = ['Ia', 'Ib', 'Ic', "Va", "Vb", "Vc"]
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()

"""## 4.Model Development

Model development adalah tahapan di mana kita menggunakan algoritma machine learning untuk menjawab problem statement
"""

# Mempersiapkan dataframe untuk analisis model
models = pd.DataFrame(index=['train_mse', 'test_mse'],
                      columns=['KNN', 'RandomForest', 'DecisionTree'])

"""### 4.1.Membuat Model dengan Algoritma KNN"""

# Membuat model dengan Algoritma KNN
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)

models.loc['train_mse','knn'] = mean_squared_error(y_pred = knn.predict(X_train), y_true=y_train)

"""### 4.2.Membuat Model dengan Algoritma RandomForest"""

# Membuat model dengan Algoritma RandomForest
RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
RF.fit(X_train, y_train)

models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)

"""### 4.3. Membuat Model dengan Algoritma Decision Tree"""

# Membuat model dengan algoritma Decision Tree
DT = DecisionTreeRegressor(max_depth=16, random_state=55)
DT.fit(X_train, y_train)

models.loc['train_mse','DecisionTree'] = mean_squared_error(y_pred=DT.predict(X_train), y_true=y_train)

"""## 5.Evaluasi Model"""

# Melakukan scaling terhadap fitur numerik pada X_test sehingga memiliki rata-rata=0 dan varians=1
X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])

# Membuat variabel mse yang isinya adalah dataframe nilai mse data train dan test pada masing-masing algoritma
mse = pd.DataFrame(columns=['train', 'test'], index=['KNN','RF', 'DT'])

# Buat dictionary untuk setiap algoritma yang digunakan
model_dict = {'KNN': knn, 'RF': RF, 'DT':DT}

# Hitung Mean Squared Error masing-masing algoritma pada data train dan test
for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))

# Panggil mse
mse

# Melakukan plot metrik dengan menggunakan bar chart
fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)

# Pengujian Model
# Pengujian Model
prediksi = X_test.iloc[2:3].copy()
pred_dict = {'y_true':y_test[2:3]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)

pd.DataFrame(pred_dict)

# Pengujian Model terhadap keseluruhan dataset
# Pengujian Model terhadap keseluruhan dataset
prediksi = X_test.copy()
pred_dict = {'y_true':y_test}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)

pd.DataFrame(pred_dict)