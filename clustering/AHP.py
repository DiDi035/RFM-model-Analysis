import ahpy

RELATIVE_IMPORTANCE = {
    'Equal importance': 1,
    'Moderate importance': 3,
    'Strong importance': 5,
    'Very strong importance': 7,
    'Extreme importance': 9,
    'Intermideate values': 1,
}

MY_RFM_COMPARISIONS = {
    ('recency', 'monetary'): 5, ('recency', 'frequency'): 3,
    ('frequency', 'monetary'): 3
}

# https://reader.elsevier.com/reader/sd/pii/S1877050910003868?token=7073ACE276A505B97CCE65D55B0B045519ED0355E31860F5619B4B08E18DA189DAF4051E2ECC340B9FB8ECBCF7C1FFE6&originRegion=eu-west-1&originCreation=20220319042401
# {'frequency': 0.637, 'monetary': 0.258, 'recency': 0.105}
RFM_COMPARISIONS_1 = {
    ('frequency', 'recency'): 5, ('frequency', 'monetary'): 3,
    ('monetary', 'recency'): 3
}

CHOSEN_COMPARISONS = RFM_COMPARISIONS_1

RFM = ahpy.Compare(
    name='RFM model', comparisons=CHOSEN_COMPARISONS, precision=3, random_index='saaty')

print(RFM.target_weights)
