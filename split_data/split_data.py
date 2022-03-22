import csv
from posixpath import split

input = './input/datasource_5.csv'
output1 = './training/datasource_5.csv'
output2 = './testing/datasource_5.csv'

file = open(input)
type(file)
csvreader = csv.reader(file)
next(csvreader)

rows = []
for row in csvreader:
    rows.append(row)

def myFunc(v):
    tmp = v[2].split('_')
    return tmp[1] + '_' + tmp[0]

rows.sort(key=myFunc)

training = []
testing = []

for row in rows:
    # datasource_1
    # if row[2].split('_')[1] == '2015':
    # datasource_3
    # if row[2] == '4_2014':
    # datasource_4
    # if row[2].split('_')[1] == '2014':
    # datasource_5
    if row[2] == '4_2017':
        testing.append(row)
    else:
        training.append(row)

header = ['customer_id', 'account_id', 'date_key', 'total_amount', 'num_trans', 'first_transaction', 'last_transaction']
with open(output1, 'w', newline="") as result:
    csvwriter = csv.writer(result)
    csvwriter.writerow(header)
    csvwriter.writerows(training)

with open(output2, 'w', newline="") as result:
    csvwriter = csv.writer(result)
    csvwriter.writerow(header)
    csvwriter.writerows(testing)

file.close()