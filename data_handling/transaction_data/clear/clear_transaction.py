import csv
from datetime import datetime

def formatDate(date: datetime):
    return date.strftime('%d/%m/%Y')

input = './rawdata/datasource_3.csv'
output = './output/transactions/datasource_3.csv'

# open file
file = open(input)
type(file)

# create reader object
csvreader = csv.reader(file)
# next(): return the first value, similar to shift() in js
next(csvreader)

rows = []

# datasource 1
""" for row in csvreader:
    customer_id = row[0]
    transaction_date = formatDate(datetime.strptime(row[1], '%d-%b-%y'))
    transaction_amount = '{:.2f}'.format(float(row[2]))
    rows.append([customer_id, transaction_date, transaction_amount]) """

# datasource 3
for row in csvreader:
    customer_id = row[1]
    transaction_date = row[2]
    transaction_amount = '{:.2f}'.format(float(row[15].replace(',', '')))
    rows.append([customer_id, transaction_date, transaction_amount])

# datasource 4
""" for row in csvreader:
    customer_id = row[1]
    if '-' in row[2]:
        transaction_date = formatDate(datetime.strptime(row[2], '%d-%m-%Y'))
    else:
        transaction_date = formatDate(datetime.strptime(row[2], '%d/%m/%Y'))
    if float(row[8]) > 0: 
        transaction_amount = '{:.2f}'.format(float(row[8]))
    else:
        continue
    rows.append([customer_id, transaction_date, transaction_amount]) """

# datasource 5
""" for row in csvreader:
    customer_id = row[2]
    transaction_date = row[3]
    transaction_amount = '{:.2f}'.format(float(row[11].replace('US$', '')))
    rows.append([customer_id, transaction_date, transaction_amount]) """

header = ['customer_id', 'transaction_date', 'transaction_amount']
# create new file and save normalized data
with open(output, 'w', newline="") as result:
    csvwriter = csv.writer(result)
    csvwriter.writerow(header)
    csvwriter.writerows(rows)

file.close()