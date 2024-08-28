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



## Data Preparation

   
