# Laporan Proyek Machine Learning - Anak Agung Sinta Trisnajayanti

## Domain Proyek
Kesehatan adalah hal yang sangat penting untuk diperhatikan bagi semua manusia karena jika tubuh sehat maka kita dapat melakukan aktivitas sehari- hari dengan baik. Namun, tidak ada yang dapat memastikan bahwa seseorang akan sehat selamanya. Resiko sakit bisa terjadi kapanpun dan kepada siapapun. Jika kita sakit salah satu yang menjadi masalah utama yaitu mengenai biaya kesehatan yang tidak murah dan kurangnya persiapan dana mengenai hal tersebut. Oleh karena itu, sangatlah diperlukan suatu persiapan untuk mengatasi risiko sakit tersebut salah satunya dengan berpartisipasi dalam asuransi kesehatan. Asuransi kesehatan adalah asuransi yang memberikan jaminan kepada tertanggung untuk mengganti setiap biaya pengobatan meliputi biaya perawatan rumah sakit, biaya pembedahan dan biaya obat- obatan. Pada website [Qoala](https://www.qoala.app/id/blog/asuransi/kesehatan/alasan-pentingnya-asuransi-kesehatan/) memberikan alasan betapa pentingnya asuransi kesehatan untuk dimiliki. Pada proyek ini saya memilih topik untuk memprediksi premi asuransi biaya pengobatan di masa depan. Premi ini ialah iuran biaya yang harus dibayarkan oleh nasabah selama jangka waktu yang sudah disepakati. Data dalam proyek ini berisi mengenai data nasabah seperti age, sex, bmi, children, smoker, region, expenses. Untuk menjawab masalah ini, predictive analytics diharapkan dapat memprediksi masalah tersebut dan mendapatkan solusi yang terbaik dengan menggunakan model machine learning. 

## Business Understanding
Asuransi kesehatan adalah asuransi yang memberikan jaminan kepada tertanggung untuk mengganti setiap biaya pengobatan meliputi biaya perawatan rumah sakit, biaya pembedahan dan biaya obat- obatan. Pada proyek ini saya memilih topik untuk memprediksi premi asuransi biaya pengobatan di masa depan. Premi ini ialah iuran biaya yang harus dibayarkan oleh nasabah selama jangka waktu yang sudah disepakati. Data dalam proyek ini berisi mengenai data nasabah seperti age, sex, bmi, children, smoker, region, expenses. Untuk menjawab masalah ini, predictive analytics diharapkan dapat memprediksi masalah tersebut dan mendapatkan solusi yang terbaik dengan menggunakan model machine learning. 
### Problem Statement
Berikut adalah problem statement dari proyek ini:
* Apakah setiap fitur dalam dataset ini memiliki pengaruh terhadap prediksi biaya asuransi yang harus dibayar nasabah?</br>
* Model Machine Learning manakah yang dapat menyelesaikan permasalahan dan menyajikan model terbaik sebagai solusi?</br>
### Goals
Berikut adalah goals yang ingin dicapai dalam proyek ini:
*	Mengetahui fitur apa saja yang mempengaruhi biaya asuransi seorang nasabah
*	Mengetahui model terbaik dalam Machine Learning untuk memprediksi biaya asuransi yang harus dibayar seorang nasabah
### Solution Statements
Untuk mencapai tujuan memprediksi biaya asuransi kesehatan ini saya menggunakan tiga model Machine Learning. Dimana ketiga model ini cocok digunakan untuk data regresi karena output yang diprediksi adalah sebuah angka. Berikut penjelasan secara singkat mengenai tiga model yang saya gunakan: 
*	**SVR (Support Vector Regression)**
<br>Algoritma SVR adalah teori yang diadaptasi dari teori machine learning yang sudah digunakan untuk memecahkan masalah klasifikasi, yaitu Support Vector Machine (SVM). SVR ini adalah penerapan algoritma SVM dalam kasus regresi. Konsep algoritma SVR dapat menghasilkan nilai peramalam yang bagus karena SVR mempunyai kemampuan menyelesaiakan masalah overfitting ([Furi, Jordi, & Saepudin, 2015](https://journal.universitasbumigora.ac.id/index.php/matrik/article/download/511/390/ )). Metode SVR masih memiliki kekurangan yaitu pada penentuan nilai parameter yang tepat. Maka  diperlukan  algoritma  optimasi  untuk  membantu  menentukan  nilai  parameter  SVR  yang  tepat.</br>
*	**Decision Tree Regression**
<br>[Decision Tree](https://www.megabagus.id/machine-learning-decision-tree-regression/) adalah teknik pengambilan keputusan dengan analogi sebuah pohon memiliki banyak cabang/ akar. Di mana satu cabang akan bercabang lagi, kemudian bercabang lagi, dan seterusnya. Dalam konteks regresi, maka decision tree adalah regresi yang bersifat non-linear dan non-kontinu (diskret). Pada algoritma ini juga memiliki kelebihan dan kekurangannya. Kelebihan nya yaitu memiliki akurasi yang baik, dapat menemukan kombinasi data yang tidak terduga, hasil keputusan dapat dibuat lebih sederhana dan spesifik. Kemudian kekurangan algoritma ini adalah waktu keputusan yang lebih lama dan memori yang dibutuhkan besar, akumulasi jumlah kesalahan dari setiap level dalam pohon keputusan besar, dan Kesulitan dalam merancang pohon keputusan yang optimal. </br>
*	**Random Forest Regression**
<br>[Regresi Random Forest](http://etd.repository.ugm.ac.id/penelitian/detail/97918) merupakan gabungan dari banyak CART yang ditumbuhkan sehingga akurasi yang dihasilkan akan lebih akurat dari pohon tunggal. Adapun kelebihan dan kekurangan pada algoritma ini yaitu kelebihannya dapat mengatasi noise dan missing value serta dapat mengatasi data dalam jumlah yang besar. Kemudian kekurangan pada algoritma Random Forest yaitu interpretasi yang sulit dan membutuhkan tuning model yang tepat untuk data. Kelebihannya yaitu dapat mengatasi noise dan missing value serta dapat mengatasi data dalam jumlah yang besar. Dan kekurangan pada algoritma Random Forest yaitu interpretasi yang sulit dan membutuhkan tuning model yang tepat untuk data </br>
## Data Understanding
Dataset yang saya gunakan berasal dari kaggle yang merupakan salah satu platform di bidang Data Science. Pada proyek ini saya menggunakan dataset [berikut](https://www.kaggle.com/noordeen/insurance-premium-prediction) dimana sesuai dengan topik yang saya ambil yaitu mengenai data nasabah asuransi. Dataset ini memiliki 1338 data yang memiliki fitur numerik dan kategorikal sebagai berikut: 
* age: Usia dari Nasabah (fitur numerik) 
* sex: Jenis kelamin dari nasabah (fitur kategorikal)
 	* female: Perempuan
  * male: Laki-laki
* bmi: Indeks massa tubuh dari nasabah. Memberikan pemahaman tentang tubuh, bobot yang relatif tinggi atau rendah (fitur numerik)
* children: Jumlah anak yang ditanggung oleh asuransi kesehatan atau jumlah tanggungan (fitur numerik)
* smoker: Apakah nasabah perokok atau tidak (fitur kategorikal)
 	* yes: Perokok
 	* no: Tidak perokok
* region: Daerah perumahan dari nasabah (fitur kategorikal)
 	* southeast: Tenggara
 	* southwest: Barat daya
 	* northwest: Barat laut
 	* northeast: Timur laut 
* expenses: Biaya yang dibayarkan nasabah ke perusahaan asuransi (fitur numerik) 


Untuk memahami sebuah data dengan memiliki jumlah yang banyak akan lebih efisien jika kita menggunakan yang disebut dengan visualisasi data. Dalam proyek ini, menggunakan beberapa visualisasi yang ada. 

Yang pertama yaitu melihat apakah terdapat outlier pada data. Outlier adalah data yang menyimpang terlalu jauh dari data yang lainnya dalam suatu rangkaian data. Dimana disini visualisasi data outlier dilakukan pada fitur numerik seperti age, children, bmi, expenses dengan menggunakan box
pada pandas. 

![messageImage_1633942610750](https://user-images.githubusercontent.com/89082302/136762431-eeeed6b0-143c-45a1-93b7-3a5b4530e1ac.jpg)

Kemudian memisahkan fitur numerik dan kategorikal dengan analisis Univariate dan Multivariate. Yang pertama saya akan membahas mengenai fitur kategorikal dari analisis Univariate dengan menggunakan countplot. 

![messageImage_1633942706804](https://user-images.githubusercontent.com/89082302/136762670-c0102860-7f51-4967-938b-5fd440a313ba.jpg)

Dilihat dari diagram diatas bahwa pada fitur kategorikal jumlah male dan female pada sampel hampir sama.  Kemudian  80% data pada sampel menunjukan bahwa nasabah tidak perokok dan 20% sebagai perokok. Lalu pada region dapat dilihat terbanyak nasabah daerah perumahan di southeast dan merata pada  southwest, northwest, northeast. 

Selanjutnya saya akan membahas mengenai fitur numerik dari analisis Univariate. Di bawah ini adalah visualisasi data dengan menggunakan hist. Pada fitur age banyak nasabah memiliki umur sekitar 20 tahun dan kebanyakan nasabah tidak memiliki anak dan hanya sebagian kecil yang memiliki lebih dari tiga anak. Kemudian pada target yaitu expenses gambar menunjukkan biaya asuransi pengobatan rendah untuk sebagian nasabah dan distribusinya bersifat logaritmik. Dari hasil ini, bisa disimpulkan bahwa banyak nasabah yang membayar asuransi sekitar 10.000$. 

![messageImage_1633942762323](https://user-images.githubusercontent.com/89082302/136762763-5cd45db2-aa64-43ac-bec8-cd9c4678d6af.jpg)

![download (2)](https://user-images.githubusercontent.com/89082302/136763805-0b8dde8c-a09d-4c80-b840-98d0c8eaf1a4.png)

Selanjutnya pada Multivariate fitur numerik visualisasi datanya menggunakan pairplots dan heatmap. Kemudian menggunakan parameter hue yaitu (smoker, dan sex) yang berfungsi mengelompokkan variabel yang akan menghasilkan data point dengan warna berbeda sesuai kategorinya.

![download (1)](https://user-images.githubusercontent.com/89082302/136763421-dc02ca5f-b139-4395-89f8-06e37740d373.png)
![download](https://user-images.githubusercontent.com/89082302/136763303-9f11ce40-d141-4637-8aeb-383123f707ca.png)

Pada pairplots di atas dapat dilihat bahwa pada diagonal pairplots  smoker dapat disimpulkan bahwa: 
 *	Perokok relatif muda (di bawah 30 tahun).
 *	Nasabah memiliki indeks bmi yang lebih rendah dibandingkan dengan bukan perokok.
 *	Nasabah memiliki lebih sedikit anak daripada bukan perokok . Pernyataan tersebut masuk akal juga karena umurnya masih muda 
 *	Yang paling penting, biaya asuransi pengobatan nasabah secara signifikan lebih tinggi dibandingkan dengan non perokok, yang berarti bahwa merokok mungkin memiliki kekuatan prediksi.
 * Trdapat hubungan lain dalam plot expenses-age dan expenses-bmi
Lalu untuk melihat hubungan lebih jelas mengenai fitur expenses-age dan expenses-bmi saya menggunakan visualisasi data yaitu lmplot. 

![download (3)](https://user-images.githubusercontent.com/89082302/136763901-a9c837b8-1f57-4480-82ca-e949f44b4167.png)
![download (4)](https://user-images.githubusercontent.com/89082302/136764021-0d1237c9-aa3f-4522-b1d9-28710fe894fb.png)

Dapat dilihat pada gambar di atas antara orang-orang dengan usia yang sama atau perokok bmi yang sama akan menghadapi biaya pengobatan yang jauh lebih tinggi dari pada tidak perokok.
## Data Preparation
Pada data preparation ini saya menggunakan One-Hot_Encoding yaitu salah satu metode encoding yang akan merepresentasikan data bertipe kategori menjadi biner yang bernilai integer 0 dan 1 . Dimana seperti yang kita ketahui jika komputer tidak dapat membaca data yang bertipe kategori sehingga kita harus mengubah data tersebut menjadi berbentuk bilangan dan akan mempermudah nantinya dalam membuat model. Berikut hasil implementasi One- Hot-Encoding pada proyek saya.
![messageImage_1633943353191](https://user-images.githubusercontent.com/89082302/136764197-ac261291-f58c-4fc1-8695-cef5e3dbe377.jpg)

Kemudian terdapat metode train-test-split dimana berfungsi untuk membagi data menjadi data latih dan data uji dijalankan sebelum membuat model. Dengan melakukan hal ini dapat melatih data untuk mencari korelasinya sendiri atau belajar pola dari data yang diberikan kemudian dilihat keakuratannya atau performa dari model nantinya. 
Selanjutnya menggunakan metode standarisasi dimana metode ini akan membuat mean data menjadi 0 dan standar deviasi menjadi 1. Metode ini juga membantu proses pelatihan model agar prediksi menjadi lebih akurat. 
## Modeling
Pada modeling proyek yang saya buat ini mencoba untuk menyelesaikan masalah menggunakan tiga solusi yaitu, SVR (Support Vector Regression), Decision Tree Regression, Random Forest Regression. Kemudian pada tahap ini juga saya melakukan improvement terhadap model seperti hyperparameter tuning. Dimana hyperparameter tuning ini adalah proses untuk menemukan nilai parameter yang dapat menghasilkan model dengan performa yang lebih baik dalam pelatihan model. Pada SVR menggunakan parameter gamma="auto", kernel="linear", C=1000. Kemudian pada Decision Tree Regression menggunakan parameter max_depth=5, random_state=13. Pada solusi terakhir menggunakan parameter n_estimators=400, max_depth=5, random_state=13. 
Untuk membandingkan ketiga model ini akan dilakukan penghitungan nilai dari Training Accuracy, Testing Accuracy, RMSE Training Data, RMSE Testing Data, dan Accuracy dari prediksi. Setelah dilakukan pelatihan maka dapat dilihat bahwa jika menggunakan model Random Forest Regression akan menghasilkan accuracy dari prediksi yang tinggi yaitu 81% dan RMSE yang terrendah. Hasil dari prediksi model terhadap data uji dapat dilihat pada gambar berikut:

![messageImage_1633943558953](https://user-images.githubusercontent.com/89082302/137136139-c5391071-ca9e-4838-a1fb-d50f94cd8bb3.jpg)

## Evaluation
Pada tahap evaluation ini saya akan menjelaskan mengenai metrik yang digunakan dalam prediksi proyek saya yaitu metriks r2_score dan metriks RMSE (Root Mean Squared Error) . r2_score disebut juga dengan koefisien determinasi yaitu sebuah nilai yang menyatakan seberapa sesuai hasil prediksi yang digunakan untuk mengevaluasi kinerja model regresi linier . Semakin besar r2_score, maka hasil prediksi semakin dekat dengan data yang sebenarnya. Semakin besar r2_score maka model semakin bagus. Nilai maksimum untuk r2_score yaitu 100% dan tidak ada nilai negatif. Berikut adalah rumus dari R²: 

![messageImage_1634131833276](https://user-images.githubusercontent.com/89082302/137142505-9b7186f9-3fd7-45f0-a4cd-a2ddfe1f574d.jpg)

Keterangan: 
<br>SS res = jumlah kuadrat dari kesalahan residual</br>
<br>SS tot = jumlah total kesalahan</br>

Kemudian disini juga menggunakan metriks RMSE untuk menghitung seberapa berbedanya seperangkat nilai. Semakin kecil nilai RMSE, semakin dekat nilai yang diprediksi dan diamati. Untuk menghitung nilai dari RMSE menggunakan rumus berikut:
![messageImage_1633943461852](https://user-images.githubusercontent.com/89082302/136764545-5708bca7-1bc3-485b-9e26-8ebd2c205eef.jpg)

Keterangan: 
<br>RMSE = nilai root mean square error </br>
<br>y  = nilai hasil observasi</br>
<br>ŷ  = nilai hasil prediksi</br>
<br>i  = urutan data pada database </br>
<br>n  = jumlah data </br>

![messageImage_1633943558953](https://user-images.githubusercontent.com/89082302/136764689-24725b4d-64ee-4772-a7e0-4b618eee31c8.jpg)

Hasil dari evaluation model pada proyek ini mengenai prediksi biaya asuransi dapat dilihat pada gambar di atas ini. Dimana accuracy prediksi tertinggi jika menggunakan model Random Forest Regression dan juga nilai RMSE model paling rendah. Maka untuk prediksi yang lebih akurat menggunakan model Random Forest Regression. 
     
 
## Kesimpulan
Kesimpulan yang didapat dalam memprediksi biaya asuransi pada proyek ini adalah sebagai berikut: 
*	Perokok atau tidak perokok mempengaruhi biaya asuransi seorang nasabah
*	Model yang lebih akurat dalam memprediksi biaya asuransi pada proyek ini adalah menggunakan model Random Forest Regression dengan memiliki accuracy prediksi tertinggi yaitu 81% dan nilai RMSE terendah. 
