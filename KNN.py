from random import seed
from random import randrange
from csv import reader
from math import sqrt

# Load CSV file
def load_csv(file):
	dataset = list()
	with open(file, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def string_ke_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# minmax
def dataset_minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax

# Normalisasi data
def normalisasi_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

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
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = hitung_akurasi(actual, predicted)
		skor.append(accuracy)
	return skor

# Jarak Euclidean
def jarak_euclidian(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

# Mencari tetangga
def cari_tetangga(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = jarak_euclidian(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors

# Buat prediksi
def prediksi_kelas(train, test_row, num_neighbors):
	neighbors = cari_tetangga(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction

# Algoritma kNN
def KNN(train, test, num_neighbors):
	predictions = list()
	for row in test:
		output = prediksi_kelas(train, row, num_neighbors)
		predictions.append(output)
	return(predictions)

# Test kNN di dataset diabetes
seed(1)
file = 'diabetes.csv'
dataset = load_csv(file)
for i in range(len(dataset[0])-1):
	string_ke_float(dataset, i)
n_folds = 5
num_neighbors = 21
skor = evaluasi(dataset, KNN, n_folds, num_neighbors)
print('Akurasi: %s' % skor)
print('Rata-rata Akurasi: %.3f%%' % (sum(skor)/float(len(skor))))