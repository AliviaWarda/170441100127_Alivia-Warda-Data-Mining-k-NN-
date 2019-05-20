# **k-Nearest Neighbor (k-NN) Algoritma**

   ![](assets\images\gambar judul.png)                                                   

## **A. Pengertian k-Nearest Neighbor (k-NN)**

*K-Nearest Neighbor* (k-NN atau KNN) adalah suatu metode untuk melakukan [klasifikasi](https://id.wikipedia.org/wiki/Pengenalan_pola) terhadap objek berdasarkan data pembelajaran yang jaraknya paling dekat dengan objek tersebut. Tujuan nya adalah untuk mengklasifikasikan obyek baru berdasarkan atribut dan training sample. 

Data baru yang diklasifikasi selanjutnya diproyeksikan pada ruang dimensi banyak yang telah memuat titik-titik c data pembelajaran. Proses klasifasikasi dilakukan dengan mencari titik **c** terdekat dari **c-baru** (*nearest neighbor*)*.*Teknik pencarian tetangga terdekat yang umum dilakukan dengan menggunakan formula jarak euclidean*.*

## **B. Kelebihan dan Kekurangan k-Nearest Neighbor (k-NN)**

​            Kelebihan : 

1. Sangat sederhana implementasi

2. Kuat dalam hal ruang pencarian

3. Efektif untuk menghitung data dalam skala kecil

4. Beberapa parameter untuk acuan : jarak metric,k

Kekurangan : 

1. Perlu untuk menentukan nilai k yang optimal sehingga untuk menyatakan jumlah tetangga terdekatnya lebih mudah

2. Biaya komputasi yang cukup tinggi karena perhitungan jarak harus dilakukan pada setiap query instance

## **C. Algoritma k-Nearest Neighbor (k-NN)**

Algoritma metode KNN sangatlah sederhana, bekerja berdasarkan jarak terpendek dari query instance ke training sample untuk menentukan KNN-nya. Training sample diproyeksikan ke ruang berdimensi banyak, dimana masing-masing dimensi merepresentasikan fitur dari data. Ruang ini dibagi menjadi bagian-bagian berdasarkan klasifikasi training sample.

Secara ringkas tahapan langkah kerja menggunakan algoritma knn, sebagai berikut :

Langkah-Langkah algoritma k-Nearest Neighbor(k-NN)

1. Menentukan parameter k (jumlah tetangga paling dekat).

2. Menghitung kuadrat jarak eucliden objek terhadap data training yang diberikan.

3. Mengurutkan hasil no 2 secara *ascending* (berurutan dari nilai tinggi ke rendah)

4. Mengumpulkan kategori Y (Klasifikasi nearest neighbor berdasarkan nilai k)

5. Dengan menggunakan kategori nearest neighbor yang paling mayoritas maka dapat dipredisikan kategori objek.

## **D. Menghitung Jarak (Euclidean Distance)**

Ada beberapa metode yang digunakan untuk menghitung jarak titik baru dengan titik training dalam algoritma KNN, metode -metode yang umum digunakan adalah *Euclidean Distance*, *Manhattan* (untuk waktu yang continu), dan *Hamming Distance* (untuk kategorikal).

**1. Euclidean distance :** Jarak Euclidean dihitung sebagai akar kuadrat dari jumlah perbedaan/selisih kuadrat antara titik baru (x) dan titik yang ada/training (y).

**2. Manhattan :** Jarak antara vektor nyata menggunakan penjumlahan dari perbedaan absolutnya.

​       ![](assets\images\gambar distance.png)

**3. Hamming distance :** digunakan untuk variabel kategorial. Jika nilai (x) dan nilai (y) sama, jarak D akan sama dengan 0. Kalau tidak, D = 1.

![](assets\images\gambar hamming distance 2.png)

​       

**E. Software Requirement**

Python 3.0 atau versi yang lebih baru, disini saya menggunakan python 3.7

1. IDE Pycharm

2. Library Python yang digunakan:

·         Numpy

Numpy merupakan sebuah library pada Python yang berfungsi untuk melakukan operasi vektor dan matriks dengan mengolah array dan array multidimensi. Biasanya NumPy digunakan untuk kebutuhan dalam menganalisis data.

instal numpy:

```
pip install numpy 
```

·         Pandas

pandas adalah sebuah librari berlisensi BSD dan open source yang menyediakan struktur data dan analisis data yang mudah digunakan dan berkinerja tinggi untuk bahasa pemrograman Python.

instal pandas:

```
pip install pandas
```

·         Matplotlib

Matplotlib adalah *library* paling banyak digunakan oleh *data science* untuk menyajikan datanya ke dalam visual yang lebih baik.

instal matplotlib:

```
pip install matplotlib
```

·         Scikit Learn

Machine learning ada yang berbasis statistika ada juga yang tidak. Salah satunya adalah support vector machine dan regresi linier. Mungkin bagi sebagian orang sudah biasa menulis sendiri library untuk implementasi kedua algoritma tadi. Tapi untuk membuatnya dalam waktu singkat tentu butuh waktu yang tidak sedikit pula.

Scikit-Learn memberikan sejumlah fitur untuk keperluan data science seperti:

·         Algoritma Regresi

·         Algoritma Naive Bayes

·         Algoritma Clustering

·         Algoritma Decision Tree

·         Parameter Tuning

·         Data Preprocessing Tool

·         Export / Import Model

·         Machine learning pipeline dan lainnya

instal Scikit Learn :

```
pip install scikit-learn 
```



## **E. Implementasi k-Nearest Neighbor (k-NN)**

Dataset yang akan digunakan untuk mengilunstrasikan algoritma KNN adalah dataset Iris. Ada beberapa tahapan yang harus dilalui sebagai berikut :

**1. Menyiapkan dan mengimpor data**

·         1.1 Mengimpor *libraries*

```
import numpy as np
import pandas as pd
import matplotlib
```

o    1.2 Memuat dataset

Kumpulan data iris mencakup tiga spesies bunga iris dengan masing-masing 50 sampel serta beberapa sifat tentang setiap bunga. Satu spesies bunga terpisah secara linear dari dua lainnya, tetapi dua lainnya tidak terpisah secara linear satu sama lain*

```
#mengimpor dataset
dataset = pd.read_csv('../input/Iris.csv')
```

o    1.3 Meringkas Dataset

```
dataset.head(5)
```

```
dataset.describe()
```

```
#Mari kita lihat jumlah instance (baris) yang dimiliki masing-masing kelas. Kita dapat melihat ini sebagai jumlah absolut.
```

```
dataset.groupby('Species').size()type
```

§  1.4 Membagi data menjadi fitur dan label

Seperti yang dapat kita lihat, dataset berisi enam kolom: Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm dan Spesies. Fitur aktual dijelaskan oleh kolom 1 - 4. Kolom terakhir berisi label sampel / class. Pertama kita perlu membagi data menjadi dua array: X (fitur) dan y (label)*

```
feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']
 X = dataset[feature_columns].values
 y = dataset['Species'].values
 ​
 \# Cara alternatif memilih fitur dan label array:
 \# X = dataset.iloc [:, 1: 5] .values
 \# y = dataset.iloc [:, 5] .values
```

§  1.6 Memisahkan dataset ke dalam set pelatihan dan set tes

Pisahkan dataset menjadi set pelatihan (data training) dan set tes (data testing), ini ditujukan untuk memeriksa nanti apakah classifier berfungsi dengan benar.

```
from sklearn.model_selection import train_test_split
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```

Metode Titik Koordinat 

Menggunakan titik koordinat paralel direpresentasikan sebagai segmen garis yang terhubung. Setiap garis vertikal mewakili satu atribut. Satu set segmen garis yang terhubung mewakili satu titik data. Poin yang cenderung mengelompok akan tampak lebih berdekatan.

```
from pandas.plotting import parallel_coordinates
 plt.figure(figsize=(15,10))
 parallel_coordinates(dataset.drop("Id", axis=1), "Species")
 plt.title('Parallel Coordinates Plot', fontsize=20, fontweight='bold')
 plt.xlabel('Features', fontsize=15)
 plt.ylabel('Features values', fontsize=15)
 plt.legend(loc=1, prop={'size': 15}, frameon=True,shadow=True, facecolor="white", edgecolor="black")
 plt.show()
```

![](assets\images\metode titik koordinat.png)

Metode Pairplot 

Paiplot berguna ketika kita ingin memvisualisasikan distribusi variabel atau hubungan antara beberapa variabel secara terpisah dalam himpunan bagian dari dataset.

```
plt.figure()
 sns.pairplot(dataset.drop("Id", axis=1), hue = "Species", height=3, markers=["o", "s", "D"])
 plt.show()
```

​       ![](assets\images\pairplot.png)

Metode Plot data klasifikasi 

```
plt.figure(figsize=[18,8])
 plt.scatter(data1['Species'], data1['sepal length (cm)'], marker= 'o')
 plt.scatter(data1['Species'], data1['sepal width (cm)'], marker= 'x')
 plt.scatter(data1['Species'], data1['petal width (cm)'], marker= '*')
 plt.scatter(data1['Species'], data1['petal length (cm)'], marker= ',')
 plt.ylabel('Length in cm')
 plt.legend()
 plt.xlabel('Species Name')
 plt.show()
```

​       ![](assets\images\gambar plot 1.png)

```
plt.figure(figsize=[18,8])
 plt.plot(data1['sepal length (cm)'], marker= 'o')
 plt.plot(data1['sepal width (cm)'], marker= 'x')
 plt.plot(data1['petal length (cm)'], marker= '*')
 plt.plot(data1['petal width (cm)'], marker= ',')
 plt.ylabel('Length in cm')
 plt.legend()
 plt.show()
```



Menggunakan cross-validation untuk penyetelan parameter:

```
# membuat daftar dari K untuk KNN
 k_list = list(range(1,50,2))
 \# membuat daftar dari cv scores
 cv_scores = []
 ​
 \# melakukan validasi silang 10 kali lipat
 for k in k_list:
   knn = KNeighborsClassifier(n_neighbors=k)
   scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
   cv_scores.append(scores.mean())     
```

![](assets\images\gambar neighbor k.png)



```
 #menemukan k paling optimal
 best_k = k_list[MSE.index(min(MSE))]
 print("The optimal number of neighbors is %d." % best_k)
```

Ouput dari hasil penerapan :

![](assets\images\output hasil penerapan knn.png)

​       

## Kesimpulan :

Dalam kNN kita perlu menghitung jarak objek baru dengan objek lain (training) untuk menetetangga terdekatnya. Pada umumnya kita menggunakan Euclidean distance.

![](assets\images\kesimpulan knn.png)

​       

Setelah menemukan faktor k yang tepat, kita dapat menggunakan faktor k tersebut sebagai model acuan.

## Link References : 

<https://medium.com/@alfiindah/k-nearest-neighbors-dengan-python-dan-scikit-learn-f5fda40b4e76>

<https://www.kaggle.com/susree64/k-nearest-neighbor-with-iris-data-set>

<https://informatikalogi.com/algoritma-k-nn-k-nearest-neighbor/>

<https://id.wikipedia.org/wiki/KNN>

<https://medium.com/bee-solution-partners/cara-kerja-algoritma-k-nearest-neighbor-k-nn-389297de543e>

<https://www.advernesia.com/blog/data-science/pengertian-dan-cara-kerja-algoritma-k-nearest-neighbours-knn/>

 