# female: 0
# male: 1

import csv
from datetime import datetime

# datasource 3
""" input = './rawdata/datasource_3.csv'
output = './output/datasource_3.csv' """

# datasource 4
""" input = './rawdata/datasource_4/Customer.csv'
output = './output/datasource_4.csv' """

# datasource 5
input = './rawdata/datasource_5/CustomerDemographic.csv'
output = './output/datasource_5.csv'

file = open(input)
type(file)

csvreader = csv.reader(file)
next(csvreader)

rows = []

# datasource 3
""" for row in csvreader:
    id = row[1]
    dob = ''
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
    rows.append([id, dob, gender, country, city, job_title, job_industry, wealth_segment]) """

# datasource 4
""" for row in csvreader:
    id = row[0]
    dob = row[1].replace("-", "/")
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
    rows.append([id, dob, gender, country, city, job_title, job_industry, wealth_segment]) """

# datasource 5
for row in csvreader:
    id = row[0]
    dobStr = row[4].split('-')
    dobStr.reverse()
    dob = '/'.join(dobStr)
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
    rows.append([id, dob, gender, country, city, job_title, job_industry, wealth_segment])

header = ['id', 'dob', 'gender', 'country', 'city', 'job_title', 'job_industry', 'wealth_segment']
with open(output, 'w', newline="") as result:
    csvwriter = csv.writer(result)
    csvwriter.writerow(header)
    csvwriter.writerows(rows)

file.close()
