import csv
from datetime import datetime

startDate = datetime(1990, 1, 1)

def getDate(date):
    return datetime.strptime(date , "%d/%m/%Y")

def getDateKey(date: datetime, year):
    # quarter 1: 1 Jan -> 31 Mar
    startQ1 = getDate('01/01' + '/' + str(year))
    endQ1 = getDate('31/03' + '/' + str(year))
    # quarter 2: 1 Apr -> 30 June
    startQ2 = getDate('01/04' + '/' + str(year))
    endQ2 = getDate('30/06' + '/' + str(year))
    # quarter 3: 1 July -> 30 Sep
    startQ3 = getDate('01/07' + '/' + str(year))
    endQ3 = getDate('30/09' + '/' + str(year))
    # quarter 4: 1 Oct -> 31 Dec

    result = ''
    if date > startQ1 and date < endQ1:
        result += '1'
    elif date > startQ2 and date < endQ2:
        result += '2'
    elif date > startQ3 and date < endQ3:
        result += '3'
    else:
        result += '4'
    return result + '_' + str(date.year)

input = 'transactions/datasource_5.csv'
output = 'output/fact_quarter_transactions_datasource_5.csv'

file = open(input)
type(file)

csvreader = csv.reader(file)
next(csvreader)

# {{customer_id, account_id, date_key}.toString(): {customer_id, account_id, date_key, total_amount, num_trans, first_transaction, last_transaction}}
map = dict()
for row in csvreader:
    transaction_date = datetime.strptime(row[1], "%d/%m/%Y")
    first_transaction = transaction_date - startDate
    last_transaction = transaction_date - startDate

    date_key = getDateKey(transaction_date, transaction_date.year)
    mapKey = str({'customer_id': row[0], 'account_id': 'training', 'date_key': date_key})

    customer = map.get(mapKey)
    if customer:
        customer['total_amount'] += float(row[2])
        customer['num_trans'] += 1
        if customer['last_transaction'] < last_transaction:
            customer['last_transaction'] = last_transaction
        if customer['first_transaction'] > first_transaction:
            customer['first_transaction'] = first_transaction
    else:
        customer = {
            'customer_id': row[0],
            'account_id': 'training',
            'date_key': date_key,
            'total_amount': float(row[2]),
            'num_trans': 1,
            'first_transaction': first_transaction,
            'last_transaction': last_transaction,
        }
    map[mapKey] = customer

header = ['customer_id', 'account_id', 'date_key', 'total_amount', 'num_trans', 'first_transaction', 'last_transaction']
with open(output, 'w', newline="") as result:
    csvwriter = csv.DictWriter(result, fieldnames=header)
    csvwriter.writeheader()
    for item in map:
        map[item]['total_amount'] = '{:.2f}'.format(map[item]['total_amount'])
        map[item]['first_transaction'] = map[item]['first_transaction'].days
        map[item]['last_transaction'] = map[item]['last_transaction'].days
        csvwriter.writerow(map[item])

file.close()