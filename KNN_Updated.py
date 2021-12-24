from random import seed
from random import randrange
from math import sqrt
from csv import reader

# Memasukkan file csv dari dataset
def load_csv(file):
	dataset = list()
	with open(file, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Mengubah datatype kolom dari string menjadi float
def string_ke_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Normalisasi data
def normalisasi_dataset(dataset, minmaks):
	for row in dataset:
		for i in range(len(row)):
			row[i] = round((row[i] - minmaks[i][0]) / (minmaks[i][1] - minmaks[i][0]), 2)
	return dataset
    
# Menentukan nilai maks dan min dari setiap kolom
def dataset_minmaks(dataset):
	minmaks = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmaks.append([value_min, value_max])
	return minmaks

# Bagi dataset menjadi n bagian
def bagi_data(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for _ in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Hitung akurasi
def hitung_akurasi(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluasi dengan cross validation split
def evaluasi(dataset, algorithm, n_folds, *args):
	folds = bagi_data(dataset, n_folds)
	skor = list()
	count = int(0)
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		count += 1
		print("Train dataset ke-{}".format(count))
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		akurasi = hitung_akurasi(actual, predicted)
		skor.append(akurasi)
	return skor

# Jarak Euclidean
def jarak_euclidian(row1, row2):
	jarak = 0.0
	for i in range(len(row1)-1):
		jarak += (row1[i] - row2[i])**2
	return sqrt(jarak)

# Mencari tetangga
def cari_tetangga(train, test_row, jumlah_tetangga):
	alldistance = list()
	for train_row in train:
		dist = jarak_euclidian(test_row, train_row)
		alldistance.append((train_row, dist))
	alldistance.sort(key=lambda tup: tup[1])
	tetangga = list()
	for i in range(jumlah_tetangga):
		tetangga.append(alldistance[i][0])
	print("Tetangga dari : {}".format(test_row))
	for i in range (jumlah_tetangga):
		print("{} Jarak Euclidian: {} \n".format((tetangga[i]), (alldistance[i][1])))
	return tetangga

# Buat prediksi
def prediksi_kelas(train, test_row, jumlah_tetangga):
	tetangga = cari_tetangga(train, test_row, jumlah_tetangga)
	output_values = [row[-1] for row in tetangga]
	prediction = max(set(output_values), key=output_values.count)
	return prediction

# Algoritma kNN
def KNN(train, test, jumlah_tetangga):
	predictions = list()
	for row in test:
		output = prediksi_kelas(train, row, jumlah_tetangga)
		predictions.append(output)
	return(predictions)

# Test kNN di dataset diabetes
seed(1)
file = 'pima-indians-diabetes.data.csv'
dataset = load_csv(file)
for i in range(len(dataset[0])):
	string_ke_float(dataset, i)
minmaks = dataset_minmaks(dataset)
dataset_normal = normalisasi_dataset(dataset,minmaks)
n_folds = 5
jumlah_tetangga = 21
skor = evaluasi(dataset_normal, KNN, n_folds, jumlah_tetangga)
print('Akurasi: %s' % skor)
print('Rata-rata Akurasi: %.3f%%' % (sum(skor)/float(len(skor))))
