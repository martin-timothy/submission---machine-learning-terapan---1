# **Laporan Proyek Machine Learning - Martin Timothy Hutajulu**

## Domain Proyek

Adapun topik dari proyek yang saya kerjakan mengenai Electric Power System (Sistem Tenaga Listrik). Sistem Transmisi merupakan proses penyaluran tenaga listrik dari tempat pembangkit tenaga listrik (Power Plant) hingga Saluran distribusi listrik (substation distribution) sehingga dapat disalurkan sampai pada konsumen pengguna listrik [1](https://ejournal.unsrat.ac.id/index.php/elekdankom/article/download/23646/23298)  

Salah satu gangguan yang sering terjadi adalah gangguan hubung singkat, baik gangguan hubung singkat antar fasa maupun gangguan hubung singkat fasa dengan tanah. Gangguan hubung singkat pada saluran transmisi apabila bersifat permanen pada umumnya dapat mengakibatkan kerusakan mekanis pada peralatan-peralatan listrik yang terhubung dengan sistem yang sedang mengalami gangguan hubung singkat tersebut. Agar tidak berpengaruh terhadap peralatan-peralatan lain, maka secepatnya gangguan hubung singkat ini perlu untuk dideteksi, diklasifikasikan dan ditentukan lokasinya dengan tepat dan jelas secepat mungkin. Dalam sistem daya yang modern, penentuan jenis gangguan hubung singkat yang terjadi dengan cepat tentu akan sangat membantu dalam penanganan gangguan. [2](https://journal.umg.ac.id/index.php/e-link/article/view/582) 

Gangguan hubung singkat dapat juga terjadi akibat adanya isolasi yang tembus atau rusak karena tidak tahan terhadap tegangan lebih, baik yang berasal dari dalam maupun yang berasal dari luar (akibat sambaran petir). Gangguan yang mengakibatkan hubung singkat dapat menimbulkan arus yang jauh lebih besar dari pada arus normal. Bila gangguan hubung singkat dibiarkan berlangsung dengan lama pada suatu sistem daya, banyak pengaruh-pengaruh yang tidak diinginkan yang dapat terjadi. [3](https://ojs.unimal.ac.id/energi-elektrik/article/download/2408/pdf_1).

## Business Understanding
### Problem statements
Adapun permasalahan yang akan diselesaikan pada proyek ini yakni :
1. Bagaimana cara mengklasifikasikan jenis gangguan hubung singkat pada saluran transmisi berdasarkan nilai arus dan tegangan pada setiap fase (A, B, C) menggunakan algoritma machine learning?
2. Algoritma machine learning yang cocok untuk memprediksi jenis gangguan hubung singkat apa yang terjadi pada saluran transmisi? 
3. Berdasarkan pernyataan nomor 2 sertakan alasan algoritma machine learning yang dpilih cocok untuk digunakan

### Goals
Untuk menyelesaikan problem statements yang telah dibuat, maka akan diselesaikan sebagai berikut
1.  - Pengumpulan Data, Kumpulkan data nilai arus dan tegangan pada setiap fase (A, B, C) dari saluran transmisi yang ingin diprediksi jenis gangguannya.
    - Preprocessing Data, Lakukan preprocessing data untuk memastikan bahwa data yang digunakan sudah dalam bentuk yang siap digunakan oleh algoritma machine learning. Contohnya, normalisasi data, penghapusan data yang tidak lengkap, dan lain-lain.
    - Pemilihan Algoritma, Pilih algoritma machine learning yang sesuai untuk mengklasifikasikan jenis gangguan hubung singkat.
    - Pelatihan Model, Pelatihan model machine learning menggunakan data yang telah dipreprocessing.
    - Evaluasi Model, Evaluasi model machine learning menggunakan data pengujian.
    - Implementasi Model, Implementasikan model machine learning yang telah dipelatihan dan dievaluasi untuk mengklasifikasikan jenis gangguan hubung singkat pada saluran transmisi berdasarkan nilai arus dan tegangan pada setiap fase (A, B, C).
2. Dari sekian banyaknya algoritma machine learning yang dapat menyelesaikan masalah tersebut, Penulis dalam menyelesaikan proyek ini akan menggunakan machine learning KNN, RandomForest, dan DecisionTree.
3. Melakukan Evaluasi Model dengan MSE (Mean Squarer Error)

### Solution Statements
Solusi yang dapat dilakukan untuk memenuhi goals proyek ini diantaranya adalah sebagai berikut :
1. - Kumpulkan data historis tentang nilai arus dan tegangan setiap fase (A, B, C) dari saluran transmisi, beserta jenis gangguan yang terkait. Data ini akan digunakan untuk melatih dan menguji model pembelajaran mesin.
   - Memproses data yang dikumpulkan terlebih dahulu untuk memastikannya dalam format yang sesuai untuk algoritme pembelajaran mesin. Ini termasuk Menormalkan data untuk mencegah dominasi fitur,  Menangani nilai yang hilang dan outlier, Mengubah data ke dalam format yang sesuai untuk algoritme yang dipilih.
   - Melatih model pembelajaran mesin yang dipilih menggunakan data yang telah diproses sebelumnya. Hal ini melibatkan memasukkan data ke dalam algoritma dan menyesuaikan parameter model untuk meminimalkan kesalahan antara jenis kesalahan yang diprediksi dan yang sebenarnya.

2. - k-Nearest Neighbor (kNN) merupakan salah satu algoritme klasifikasi dalam data mining yang memanfaatkan data terdekat untuk melakukan prediksi pada data baru yang belum dikenal (data uji). Algoritme ini bekerja dengan cara mencari sejumlah tetangga terdekat dari data uji dan menentukan kelas data uji tersebut berdasarkan mayoritas kelas dari tetangga terdekat (data latih) yang ditemukan.
   - ![image](https://miro.medium.com/v2/resize:fit:640/format:webp/0*5F4J_0lY9qQY7fJ0)


   - Decision Tree ![image](https://blog.algorit.ma/content/images/size/w1000/2022/07/Decision-Tree-Diagram-Example-MindManager-Blog.png)
   - Decision tree yang digunakan untuk memecahkan masalah regresi. Dalam regression tree, variabel target atau dependen merupakan variabel kontinu. Setiap cabang pada pohon decision tree merepresentasikan suatu keputusan yang dapat menghasilkan prediksi nilai kontinu pada data yang diberikan.



   - Random Forest ![image](https://sis.binus.ac.id/wp-content/uploads/2024/04/Random-Forest.png).
   - Random Forest merupakan gabungan dari beberapa decision tree.memiliki dua fase yaitu Pertama, menggabungkan sejumlah N decision tree untuk membuat random forest. Kedua, membuat prediksi untuk setiap tree yang dibuat pada fase pertama. Random Forest dapat bekerja pada kedua tugas sekaligus Classification dan Regression, mampu menangani data set yang besar, dan meningkatkan akurasi model dan menangani masalah overfitting.
4. Dalam mendeteksi jenis gangguan pada transmission line maka digunakan metrik evaluasi untuk menyakinkan bahwa algortima yang digunakan cocok dengan menggunakan MSE (Mean Squared Error).

## Data Understanding
Adapun dataset yang digunakan pada proyek ini yakni [Electrical fault detection and classification](https://www.kaggle.com/datasets/esathyaprakash/electrical-fault-detection-and-classification). Pada dataset tersebut terdiri dari dua dataset yakni **classData.csv** dan **detect_dataset.csv**, pada proyek ini hanya menggunakan dataset **classData.csv** dengan alasan bahwa dataset tersebut disertai jenis jenis gangguan hubung singkat.

### Variabel - variabel pada Electrical fault detection and classification dataset adalah sebagai berikut
1. G : Bagian dari garis patahan, jika tidak 1, maka 0.
2. C : Bagian dari garis patahan, jika tidak 1, maka 0.
3. B : Bagian dari garis patahan, jika tidak 1, maka 0.
4. A : Bagian dari garis patahan, jika tidak 1, maka 0.
5. Ia : Arus Listrik pada Phase A
6. Ib : Arus Listrik pada Phase B
7. Ic : Arus Listrik pada Phase C
8. Va : Tegangan Listrik pada Phase A
9. Vb : Tegangan Listrik pada Phase B
10. Vc : Tegangan Listrik pada Phase C


Inputs - [Ia,Ib,Ic,Va,Vb,Vc]


Outputs - [G C B A]


[0 0 0 0] - No Fault


[1 0 0 1] - LG fault (Between Phase A and Gnd)


[0 0 1 1] - LL fault (Between Phase A and Phase B)


[1 0 1 1] - LLG Fault (Between Phases A,B and ground)


[0 1 1 1] - LLL Fault(Between all three phases)


[1 1 1 1] - LLLG fault( Three phase symmetrical fault)


## Exploratory Data Analysis
### Tipe data pada setiap kolom pada Electrical fault detection and classification dataset
![Screenshot 2024-08-28 145646](https://github.com/user-attachments/assets/b219e9c9-ab23-4806-92a1-fd917f1c7494)

didapatkan tipe dataset bertipe int64 dan float64, dimana dataset memiliki 7861 baris dan 10 kolom

### Pengecekan outliers pada masing-masing kolom
Setelah dilakukan pengecekan outliers didapatkan kolom Ia, Ib, Ic, Va memiliki outliers
![Screenshot 2024-08-28 145909](https://github.com/user-attachments/assets/2b41770f-f953-4232-8ac6-d28be95e2821)
![Screenshot 2024-08-28 150101](https://github.com/user-attachments/assets/209c4ff5-c0e6-4a23-957f-ebbfd1f7cc40)
![Screenshot 2024-08-28 150152](https://github.com/user-attachments/assets/3f9330a8-c41d-4610-bf33-233994603ec8)
![Screenshot 2024-08-28 150238](https://github.com/user-attachments/assets/5e7bc1cd-1bfd-4b59-892a-9d8e1ffc780a)

### Penanganan Missing Value dan Outliers
Pada dataset yang dipakai pada proyek ini tidak didapatkan missing value dan data duplikat.
Mengenai Outliers dilakukan pembersihan dengan cara menggunakan IQR method, sehingga setelah dibersihkan total baris dan kolom pada dataset yakni 2669 baris dan 10 kolom.

### Penggabungkan Kolom G, B, C, A
![Screenshot 2024-08-28 150326](https://github.com/user-attachments/assets/fe0c2232-d9c8-464f-9ff5-f0d690a8d757)


untuk memudahkan dalam memprediksi gangguan maka dilakukan penggabungkan kolom dan kemudian menentukan nilai dan jenis gangguan berdasarkan keterangan dataset diawal tadi.

### Unvariate Analysis
![Screenshot 2024-08-28 150446](https://github.com/user-attachments/assets/b2aa7c3d-f3de-4154-be62-621d938c6181)

berikut merupakan gangguan yang sering terjadi pada dataset yang digunakan, dapat diketahui bahwa tidak adanya gangguan menjadi terbanyak pada dataset yang digunakan.

### Multivariate Analysis
![Screenshot 2024-08-28 151755](https://github.com/user-attachments/assets/3cefb3ab-e436-4dc6-85cd-e64285fe4eb8)


berikut merupakan korelasi antar setiap kolom dimana 
Vb berkorelasi negatif tinggi dengan Vc, dan Ic yakni -0.78 dan -0.73
Va berkorelasi positif tinggi dengan Ic yakni 0.55, disusul dengan Vc berkorelasi positif dengan Ic yakni 0.43

## Data Preparation
Sebelum melakukan train-test split kita lakukan pengubahan menjadi numerik pada kolom output (dimana merupakan hasil penggabungan kolom G,B,C,A) dengan menggunakan LabelEncoder dengan tujuan untuk model machine learning dapat bekerja pada saat melakukan prediksi
sehingga didapatkan hasilnya :
1. LG fault (Between Phase A and Gnd) di encoded menjadi 0
2. LLL Fault(Between all three phases) di encoded menjadi 1
3. LLLG fault( Three phase symmetrical fault) di encoded menjadi 2
4. No Fault di encoded menjadi 3
5. Unknown Fault di endoded menjadi : 4
Setelah dilakukan encoded kemudian dilakukan pemisahan antara kolom yang bersifat input seperti kolom Ia, Ib, Ic, Va, Vb, Vc dan kolom yang bersifat output seperti kolom output.
Untuk mencegah terjadinya imbalance class pada dataset maka digunakan SMOTE supaya menghindarkan class yang dominan sehingga dapat membantu model menghasilkan kerja yang baik.


### Train-test split dan Standarisasi
Dilakukan train test split dengan rasio 80 : 20 serta standarisasi data pada kolom Ia, Ib, Ic, Va, Vb, Vc untuk menghasilkan data yang seragam dan menghasilkan model dapat bekerja dengan lebih baik

## Modelling
Adapun Modelling yang digunakan 
1. KNN
   - Kelebihan, algoritmanya yang sederhana dan mudah diimplementasikan. Selanjutnya, tidak perlu membangun model, membuat beberapa parameter, atau membuat asumsi tambahan. Terakhir, algoritma K-nearest neighbor ini sangat serbaguna. Anda bisa menggunakannya untuk membuat klasifikasi, regresi, dan pencarian data.
   - Kekurangan, Algoritmanya bisa menjadi lebih lambat secara signifikan karena jumlah contoh atau prediksi variabel independennya meningkat. Selain itu, K-nearest neighbor juga selalu memerlukan penentuan nilai K yang mungkin kompleks untuk beberapa kasus. Biaya komputasinya juga tinggi karena harus menghitung jarak antara titik data dengan semua sampel yang tersebar di sekitarnya (neighbor).


2. DecisionTree 
   - Kelebihan, Saat menggunakan decision tree sebagai alat pengambilan keputusan bagi perusahaan maupun organisasi, tentu ada kelebihan yang membuat pohon keputusan ini dipakai. Kelebihan tersebut bisa dilihat dari kemudahan pembuatannya. Selain itu, jika ada opsi yang lebih baik, ini juga dapat dengan mudah dimasukkan dalam rangkaian pohon keputusan. Di sisi lain, penggunaan decision tree juga bisa dipadukan dengan beberapa metode pengambilan keputusan lainnya agar dapat menghasilkan keputusan final yang matang.
   - Kekurangan, dalam kasus-kasus kompleks, tentu decision tree yang dibangun akan lebih banyak dan dalam, sehingga membuat gambar pohonnya terlalu membingungkan. Saat mengalami hal-hal seperti ini, perusahaan atau organisasi bisa menggunakan influence diagram yang fokus pada pengambilan keputusan, mulai dari tujuan utama hingga masukan yang diterima.
  

3. RandomForest 
   - Kelebihan, Keuntungan  ketika menggunakan metode random forest adalah keserbagunaannya. Hal ini dapat terlihat dari klasifikasi yang diciptakan. Anda bisa lebih mudah melihat keperluan relatif pada fitur input. Kumpulan pohon keputusan yang hadir secara acak ini bisa menghasilkan prediksi yang tepat dan akurat.
   - Kekurangan, karena random forest mengutamakan pohon yang acak, maka algoritma menjadi lambat dan terkadang tidak efektif. Karena kinerja yang lambat ini, maka hasil prediksi tidak bisa terjadi secara real time. Padahal, ketika diaplikasikan pada dunia nyata, run time sebuah machine learning sangatlah penting.

Model Machine Learning yang telah dikemukakan diatas akan digunakan pada proyek ini




   
