# female: 0
# male: 1

import csv
from datetime import datetime

input = './rawdata/datasource_4/Customer.csv'
output = './output/customers/datasource_4.csv'

file = open(input)
type(file)

csvreader = csv.reader(file)
next(csvreader)

rows = []

# datasource 3
""" for row in csvreader:
    id = row[1]
    age = ''
    if row[4] == 'Female':
        gender = 0
    elif row[4] == 'Male':
        gender = 1
    else:
        gender = ''
    country = row[6]
    city = row[8]
    job_title = ''
    job_industry = ''
    wealth_segment = ''
    rows.append([id, age, gender, country, city, job_title, job_industry, wealth_segment]) """

# datasource 4
for row in csvreader:
    id = row[0]
    age = int(datetime.strptime(row[1], '%d-%m-%Y').timestamp())
    if row[2] == 'F':
        gender = 0
    elif row[2] == 'M':
        gender = 1
    else:
        gender = ''
    country = ''
    city = ''
    job_title = ''
    job_industry = ''
    wealth_segment = ''
    rows.append([id, age, gender, country, city, job_title, job_industry, wealth_segment])

# datasource 5
""" for row in csvreader:
    id = row[0]
    age = ''
    if row[2] == 'Female':
        gender = 0
    elif row[2] == 'Male':
        gender = 1
    else:
        gender = ''
    country = 'Australia'
    city = ''
    job_title = row[6]
    job_industry = row[7]
    wealth_segment = row[8]
    rows.append([id, age, gender, country, city, job_title, job_industry, wealth_segment]) """

header = ['id', 'age', 'gender', 'country', 'city', 'job_title', 'job_industry', 'wealth_segment']
with open(output, 'w', newline="") as result:
    csvwriter = csv.writer(result)
    csvwriter.writerow(header)
    csvwriter.writerows(rows)

file.close()