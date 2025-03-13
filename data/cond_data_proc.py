import pandas as pd

data = pd.read_csv('data/CALISol-23 Dataset.csv')
# data with T between 298 and 304
data = data[(data['T'] > 298) & (data['T'] < 304)]
# count the number of data points for each solvent ratio type
# print(data['solvent ratio type'].value_counts())
data = data[data['solvent ratio type'] == 'w']
# drop columns with 0 for all rows
data = data.loc[:, (data != 0).any(axis=0)]
# drop doi, T, c units, solvent ratio type
data = data.drop(columns=['doi', 'T', 'c units', 'solvent ratio type'])

props = pd.read_csv('elements.csv')

# add columns 'solvent 1', 'solvent 2' and 'solvent 3' to data
solvents = data.iloc[:, 3:]
data = pd.concat([data, pd.DataFrame(columns=['solvent 1', 'solvent 2', 'solvent 3', 'solvent 4'])])
# find which columns after 'salt' are not 0 for each row and assign them to 'solvent 1', 'solvent 2' and 'solvent 3'
for i, row in solvents.iterrows():
    s = row[row != 0].index.tolist()
    # check how many solvents are there, and assign based on that
    if len(s) == 1:
        data.at[i, 'solvent 1'] = s[0]
    elif len(s) == 2:
        data.at[i, 'solvent 1'] = s[0]
        data.at[i, 'solvent 2'] = s[1]
    elif len(s) == 3:
        data.at[i, 'solvent 1'] = s[0]
        data.at[i, 'solvent 2'] = s[1]
        data.at[i, 'solvent 3'] = s[2]
    elif len(s) == 4:
        data.at[i, 'solvent 1'] = s[0]
        data.at[i, 'solvent 2'] = s[1]
        data.at[i, 'solvent 3'] = s[2]
        data.at[i, 'solvent 4'] = s[3]

data = pd.concat([data, pd.DataFrame(columns=['s1 ratio', 's2 ratio', 's3 ratio', 's4 ratio'])])
# find the ratio of each solvent in the mixture, check for nan values
for i, row in data.iterrows():
    if not pd.isna(row['solvent 1']):
        data.at[i, 's1 ratio'] = row[row['solvent 1']]
    if not pd.isna(row['solvent 2']):
        data.at[i, 's2 ratio'] = row[row['solvent 2']]
    if not pd.isna(row['solvent 3']):
        data.at[i, 's3 ratio'] = row[row['solvent 3']]
    if not pd.isna(row['solvent 4']):
        data.at[i, 's4 ratio'] = row[row['solvent 4']]
        
data = data.drop(columns=data.columns[3:13])

data = pd.concat([data, pd.DataFrame(columns=['s1 M', 's2 M', 's3 M', 's4 M'])])
for i, row in data.iterrows():
    for j in range(1, 5):
        # if not N/A look up name of solvent in prop and the molecular weight in column 'mol/L'
        if not pd.isna(row[f'solvent {j}']):
            data.at[i, f's{j} M'] = props[props['Species'] == row[f'solvent {j}']]['mol/L'].values[0] * row[f's{j} ratio']

# replace nan in props with 0
props = props.fillna(0)

# add columns 'n Oxygen', 'n Carbon', 'n Fluorine', 'n Hydrogen', 'n Sulfur', 'n Phosphorus' to data
data = pd.concat([data, pd.DataFrame(columns=['n O', 'n C', 'n F', 'n H', 'n S', 'n P', 'n N'])])
# find the number of atoms of each type in the molecule from prop
for i, row in data.iterrows():
    for k in ['O', 'C', 'F', 'H', 'S', 'P', 'N']:
        total = 0
        for j in range(1, 5):
            if not pd.isna(row[f's{j} M']):
                total += props[props['Species'] == row[f'solvent {j}']][f'{k}'].values[0] * row[f's{j} M']
        total += props[props['Species'] == row['salt']][f'{k}'].values[0] * row['c']
        data.at[i, f'n {k}'] = total
        
data = pd.concat([data, pd.DataFrame(columns=['ns O', 'ns C', 'ns F', 'ns H', 'ns S', 'ns P', 'ns N'])])
# find the number of atoms of each type in the molecule from prop
for i, row in data.iterrows():
    for k in ['O', 'C', 'F', 'H', 'S', 'P', 'N']:
        total = 0
        for j in range(1, 5):
            if not pd.isna(row[f's{j} M']):
                total += props[props['Species'] == row[f'solvent {j}']][f'{k}'].values[0] * row[f's{j} M']
        data.at[i, f'ns {k}'] = total

data = pd.concat([data, pd.DataFrame(columns=['na O', 'na C', 'na F', 'na H', 'na S', 'na P', 'na N'])])
# find the number of atoms of each type in the molecule from prop
for i, row in data.iterrows():
    for k in ['O', 'C', 'F', 'H', 'S', 'P', 'N']:
        total = 0
        total += props[props['Species'] == row['salt']][f'{k}'].values[0] * row['c']
        data.at[i, f'na {k}'] = total

data = pd.concat([data, pd.DataFrame(columns=['O', 'sO', 'aO', 'C', 'sC', 'aC', 'F', 'sF', 'aF', 'FO', 'FC', 'OC', 'InOr'])])
for i, row in data.iterrows():
    n_total = row['n O'] + row['n C'] + row['n F'] + row['n H'] + row['n S'] + row['n P'] + row['n N']
    data.at[i, 'O'] = row['n O']/n_total
    data.at[i, 'C'] = row['n C']/n_total
    data.at[i, 'F'] = row['n F']/n_total
    data.at[i, 'sO'] = row['ns O']/n_total
    data.at[i, 'sC'] = row['ns C']/n_total
    data.at[i, 'sF'] = row['ns F']/n_total
    data.at[i, 'aO'] = row['na O']/n_total
    data.at[i, 'aC'] = row['na C']/n_total
    data.at[i, 'aF'] = row['na F']/n_total
    data.at[i, 'FO'] = row['n F']/(row['n O'] + row['n F'])
    data.at[i, 'FC'] = row['n F']/(row['n C'] + row['n F'])
    data.at[i, 'OC'] = row['n O']/(row['n C'] + row['n O'])
    data.at[i, 'InOr'] = row['n F']/(n_total - row['n F'])

ce_X = data.loc[:, 'O':'InOr']
ce_y = data.loc[:, 'k']
# data_fea.to_csv('data_fea.csv')
# data_k.to_csv('data_k.csv')