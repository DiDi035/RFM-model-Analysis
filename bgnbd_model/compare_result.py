from cProfile import label
import csv
import matplotlib.pyplot as plt
import operator
import numpy as np

def normalizedData(filename):
    file = open(filename)
    type(file)
    csvreader = csv.reader(file)
    next(csvreader)
    # customer_id: {id, x, t_x, T}
    map = dict()
    for row in csvreader:
        mapKey = row[0]
        x = float(row[4])
        customer = map.get(mapKey)
        if customer:
            customer += x
        else:
            customer = x
        map[mapKey] = customer
    file.close()
    return map

def drawGraph(real_filename, predict_filename, axis, title):
    real = normalizedData(real_filename)
    predict = normalizedData(predict_filename)
    differences = dict()
    for key in predict.keys():
        diff = 0
        if real.get(key):
            diff = round(abs(predict[key] - real[key]))
        else:
            diff = round(predict[key])
        if differences.get(diff):
            differences[diff] += 1
        else:
            differences[diff] = 1
    axis.bar(differences.keys(), differences.values())
    axis.set_title(title)
    
# main
# datasource_1
figure, axis = plt.subplots(2, 2)
drawGraph('./testing/datasource_1.csv', './output/datasource_1.csv', axis[0, 0], 'datasouce 1')
drawGraph('./testing/datasource_3.csv', './output/datasource_3.csv', axis[0, 1], 'datasouce 3')
drawGraph('./testing/datasource_4.csv', './output/datasource_4.csv', axis[1, 0], 'datasouce 4')
drawGraph('./testing/datasource_5.csv', './output/datasource_5.csv', axis[1, 1], 'datasouce 5')
plt.show()