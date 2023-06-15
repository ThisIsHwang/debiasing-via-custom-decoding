import csv

input_file = '/home/doubleyyh/bias_mitigation/dataset/train.tsv'
output_file = 'preprocessed_train.tsv'

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    tsv_reader = csv.reader(infile, delimiter='\t')
    tsv_writer = csv.writer(outfile, delimiter='\t')

    for row in tsv_reader:
        if row[3] == 'debias':
            tsv_writer.writerow(row[:2])
