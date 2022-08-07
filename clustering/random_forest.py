import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def pre_processing(raw_data, RFM_km, highest_clv_cluster_idx):
    for (column_name, column_data) in raw_data.iteritems():
        if column_data.isnull().all():
            raw_data.drop(column_name, inplace=True, axis=1)

    classify_labels = np.zeros(len(raw_data))
    raw_data['label'] = classify_labels
    for _idx, row in RFM_km.iterrows():
        if row.cluster_id == highest_clv_cluster_idx:
            raw_data.loc[raw_data['customer_id']
                         == row.customer_id, 'label'] = 1

    return raw_data


def call(RFM_km_with_labels):
    if 'dob' in RFM_km_with_labels.columns:
        RFM_km_with_labels.dob = LabelEncoder() \
            .fit_transform(RFM_km_with_labels.dob)
    if 'gender' in RFM_km_with_labels.columns:
        RFM_km_with_labels.gender = LabelEncoder() \
            .fit_transform(RFM_km_with_labels.gender)
    if 'country' in RFM_km_with_labels.columns:
        RFM_km_with_labels.country = LabelEncoder() \
            .fit_transform(RFM_km_with_labels.country)
    if 'city' in RFM_km_with_labels.columns:
        RFM_km_with_labels.city = LabelEncoder() \
            .fit_transform(RFM_km_with_labels.city)
    if 'job_title' in RFM_km_with_labels.columns:
        RFM_km_with_labels.job_title = LabelEncoder() \
            .fit_transform(RFM_km_with_labels.job_title)
    if 'job_industry' in RFM_km_with_labels.columns:
        RFM_km_with_labels.job_industry = LabelEncoder() \
            .fit_transform(RFM_km_with_labels.job_industry)
    if 'wealth_segment' in RFM_km_with_labels.columns:
        RFM_km_with_labels.wealth_segment = LabelEncoder() \
            .fit_transform(RFM_km_with_labels.wealth_segment)

    inputs = RFM_km_with_labels.drop('label', axis='columns')
    target = RFM_km_with_labels.label
    scaler = StandardScaler()
    inputs = scaler.fit_transform(inputs)
    classifier = RandomForestClassifier()
    classifier.fit(inputs, target)
    dump = pickle.dumps(classifier)
    pickle_model = pickle.loads(dump)
    return [pickle_model.predict(inputs), target]
