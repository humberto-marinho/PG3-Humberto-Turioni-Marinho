import csv

def read_csv_to_array(filename,encoding='utf-8'):
    data_array = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data_array.append(row[0])
    return data_array

def read_txt_to_array(filename):
    data_array = []
    with open(filename, 'r') as file:
        for line in file:
            data_array.append(line.strip())
    return data_array

def write_to_csv(filename, data): 
    with open(filename, 'a', newline='') as file:
        csv_writer = csv.writer(file, delimiter=';')
        csv_writer.writerow(data.split(';'))