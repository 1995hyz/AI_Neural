import csv
import random

file_path = "winequality-white.csv"
train_file_path = "wine_quality.train"
test_file_path = "wine_quality.test"
weight_file_path = "wine_quality_weight.init"
train_data_num = 3500
output_layer_num = 11
hidden_layer_num = 10
input_layer_num = 11

column_max = [14.2, 1.1, 1.66, 65.8, 0.346, 289, 440, 1.03898, 3.82, 1.08, 14.2]
bit_mapping = {
    0: ['0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1'],
    1: ['1', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1'],
    2: ['1', '1', '0', '1', '1', '1', '1', '1', '1', '1', '1'],
    3: ['1', '1', '1', '0', '1', '1', '1', '1', '1', '1', '1'],
    4: ['1', '1', '1', '1', '0', '1', '1', '1', '1', '1', '1'],
    5: ['1', '1', '1', '1', '1', '0', '1', '1', '1', '1', '1'],
    6: ['1', '1', '1', '1', '1', '1', '0', '1', '1', '1', '1'],
    7: ['1', '1', '1', '1', '1', '1', '1', '0', '1', '1', '1'],
    8: ['1', '1', '1', '1', '1', '1', '1', '1', '0', '1', '1'],
    9: ['1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '1'],
    10: ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0']
}

train_samples = []
test_samples = []
with open(file_path) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=",")
    counter = 0
    for row in readCSV:
        if counter < train_data_num:
            counter += 1
            processed_row = []
            for i in range(len(column_max)):
                processed_row.append('{0:.3f}'.format(float(row[i]) / column_max[i]))
            processed_row += bit_mapping[int(row[11])]
            train_samples.append(processed_row)
        else:
            processed_row = []
            for i in range(len(column_max)):
                processed_row.append('{0:.3f}'.format(float(row[i]) / column_max[i]))
            processed_row += bit_mapping[int(row[11])]
            test_samples.append(processed_row)

with open(train_file_path, 'w+') as f:
    f.write(str(len(train_samples)) + " " + str(11) + " " + str(11) + "\n")
    for row in train_samples:
        f.write(" ".join(row))
        f.write("\n")

with open(test_file_path, 'w+') as f:
    f.write(str(len(test_samples)) + " " + str(11) + " " + str(11) + "\n")
    for row in train_samples:
        f.write(" ".join(row))
        f.write("\n")

weights = []
for i in range(hidden_layer_num):
    row = []
    for j in range(input_layer_num+1):
        row.append('{0:.3f}'.format(float(random.uniform(0, 1))))
    weights.append(row)
for i in range(output_layer_num):
    row = []
    for j in range(hidden_layer_num+1):
        row.append('{0:.3f}'.format(float(random.uniform(0, 1))))
    weights.append(row)

with open(weight_file_path, 'w+') as f:
    f.write(str(input_layer_num) + " " + str(hidden_layer_num) + " " + str(output_layer_num) + "\n")
    for row in weights:
        f.write(" ".join(row))
        f.write("\n")
