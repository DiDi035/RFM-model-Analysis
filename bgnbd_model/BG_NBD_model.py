import csv
from datetime import datetime
from traceback import print_tb
from lifetimes import ModifiedBetaGeoFitter

def normalizedData(input_file, end_date):
    file = open(input_file)
    type(file)
    csvreader = csv.reader(file)
    next(csvreader)
    # customer_id: {id, x, t_x, T}
    map = dict()
    for row in csvreader:
        mapKey = row[0]
        x = int(row[4])
        first_purchase = int(row[5])
        last_purchase = int(row[6])
        customer = map.get(mapKey)
        if customer:
            customer['x'] += x
            if customer['first_purchase'] > first_purchase:
                customer['first_purchase'] = first_purchase
            if customer['last_purchase'] < last_purchase:
                customer['last_purchase'] = last_purchase
        else:
            customer = {
                'id': mapKey,
                'x': x - 1,
                'first_purchase': first_purchase,
                'last_purchase': last_purchase,
            }
        map[mapKey] = customer
    file.close()
    for item in map.keys():
        map[item]['T'] = end_date - map[item]['first_purchase']
        map[item]['t_x'] = map[item]['last_purchase'] - map[item]['first_purchase']
    return map

def bgnbdModel(data, predict_time):
    arr_id = []
    arr_x = []
    arr_tx = []
    arr_T = []
    for item in data.values():
        arr_id.append(item['id'])
        arr_x.append(item['x'])
        arr_tx.append(item['t_x'])
        arr_T.append(item['T'])
    bgf = ModifiedBetaGeoFitter(penalizer_coef=0.000001)
    bgf.fit(arr_x, arr_tx, arr_T)
    predict = bgf.predict(predict_time, arr_x, arr_tx, arr_T)
    return predict

def savePredict(data, predict, output_file):
    header = ['id', 'x', 'tx', 'T', 'predict']
    with open(output_file, 'w', newline="") as result:
        csvwriter = csv.writer(result)
        csvwriter.writerow(header)
        for index, item in enumerate(data.values()):
            csvwriter.writerow([item['id'], item['x'], item['t_x'], item['T'], predict[index]])

# main
# quarter 1: 1 Jan -> 31 Mar
# quarter 2: 1 Apr -> 30 June
# quarter 3: 1 July -> 30 Sep
# quarter 4: 1 Oct -> 31 Dec
defaultStartDate = datetime(1990, 1, 1)

# datasource_1
data = normalizedData('./training/datasource_1.csv', (datetime(2014, 12, 31) - defaultStartDate).days)
predict = bgnbdModel(data, 365)
savePredict(data, predict, './output/datasource_1.csv')

# datasource_3
data = normalizedData('./training/datasource_3.csv', (datetime(2014, 3, 31) - defaultStartDate).days)
predict = bgnbdModel(data, 92)
savePredict(data, predict, './output/datasource_3.csv')

# datasource_4
data = normalizedData('./training/datasource_4.csv', (datetime(2013, 12, 31) - defaultStartDate).days)
predict = bgnbdModel(data, 365)
savePredict(data, predict, './output/datasource_4.csv')

# datasource_5
data = normalizedData('./training/datasource_5.csv', (datetime(2017, 9, 30) - defaultStartDate).days)
predict = bgnbdModel(data, 92)
savePredict(data, predict, './output/datasource_5.csv')
