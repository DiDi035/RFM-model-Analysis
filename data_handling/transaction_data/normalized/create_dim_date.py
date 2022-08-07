import csv

# quarter 1: 1 Jan -> 31 Mar
# quarter 2: 1 Apr -> 30 June
# quarter 3: 1 July -> 30 Sep
# quarter 4: 1 Oct -> 31 Dec
def getStartDay(quarter, year):
    switcher = {
        '1': '01/01',
        '2': '01/04',
        '3': '01/07',
        '4': '01/10'
    }
    return switcher[quarter] + '/' + year

def getEndDay(quarter, year):
    switcher = {
        '1': '31/03',
        '2': '30/06',
        '3': '30/09',
        '4': '31/12'
    }
    return switcher[quarter] + '/' + year

input = 'fact_quarter_transactions.csv'
output = 'dim_date.csv'

file = open(input)
type(file)

csvreader = csv.reader(file)
next(csvreader)

# {date_key: {date_key, quarter_start_day, quarter_end_day, quarter, year}}
map = dict()
for row in csvreader:
    item = map.get(row[2])
    arr = row[2].split('_')
    if item:
        continue
    else:
        item = {
           'date_key': row[2],
           'quarter_start_day': getStartDay(arr[0], arr[1]),
           'quarter_end_day': getEndDay(arr[0], arr[1]),
           'quarter': arr[0],
           'year': arr[1],
        }
    map[row[2]] = item

header = ['date_key', 'quarter_start_day', 'quarter_end_day', 'quarter', 'year']
with open(output, 'w', newline="") as result:
    csvwriter = csv.DictWriter(result, fieldnames=header)
    csvwriter.writeheader()
    for item in map:
        csvwriter.writerow(map[item])

file.close()
