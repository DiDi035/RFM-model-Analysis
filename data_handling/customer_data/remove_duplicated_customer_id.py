import csv

input = './output/datasource_5.csv'
output = './cleanned/datasource_5.csv'

file = open(input)

csvreader = csv.reader(file)

next(csvreader)


headers = ['id','dob','gender','country','city','job_title','job_industry','wealth_segment']

with open(output, 'w', newline="") as result:
    hash_map = dict()
    csvwriter = csv.writer(result)
    csvwriter.writerow(headers)
    for row in list(csvreader):
        if row[0] not in hash_map:
            csvwriter.writerow(row)
            hash_map[row[0]] = 1

file.close()
