import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import random
import sys


sys.path.insert(1, '../PolitiTrend/')

import data_cleaning
# Load data
# df = pd.read_csv('p_and_c.csv')
df = pd.read_csv('r_and_l.csv')

df = data_cleaning.remove_stop_words(df,"text","cleaned_text")
df = data_cleaning.remove_punctuations(df,"cleaned_text","cleaned_text")
df = data_cleaning.lemm_text(df,"cleaned_text","cleaned_text")

df['cmp_code'] = df['cmp_code'].replace({'laissez-faire': 0, 'regulationism': 1})
#df['cmp_code'] = df['cmp_code'].replace({'conservatism': 0, 'progressivism': 1})
df = df.rename(columns={'cmp_code': 'label'})

train_ratio = 0.64
validation_ratio = 0.16
test_ratio = 0.2




test_data, remaining_data = train_test_split(df, train_size=0.2, shuffle=True, stratify=df['label'],random_state=42)
train_data, validation_data = train_test_split(remaining_data, train_size=0.8,shuffle=True, stratify=remaining_data['label'],random_state=42)

# Save each split into separate TSV files
train_data.to_csv('/Users/lifeifan/Desktop/ece1786/project/model/project_CNN/data/train.tsv', sep='\t', index=False)
validation_data.to_csv('/Users/lifeifan/Desktop/ece1786/project/model/project_CNN/data/validation.tsv', sep='\t', index=False)
test_data.to_csv('/Users/lifeifan/Desktop/ece1786/project/model/project_CNN/data/test.tsv', sep='\t', index=False)

df_label0 = df[df['label'] == 0]
df_label1 = df[df['label'] == 1]

overfit_data = pd.concat([df_label0.sample(25, random_state=1), df_label1.sample(25, random_state=1)])

overfit_data.to_csv('/Users/lifeifan/Desktop/ece1786/project/model/project_CNN/data/overfit.tsv', sep="\t")
print("the size of train_data", len(train_data))
print("the size of validation_data", len(validation_data))
print("the size of test_data", len(test_data))
print("the size of overfit_data", len(overfit_data))

def check_class_distribution(data):
    if data['label'].value_counts()[0] == data['label'].value_counts()[1]:
        print("This dataset is balance!")
    else:
        print("This dataset is not balance!")

class_distribution_consistency_train_validation = check_class_distribution(train_data)
class_distribution_consistency_train_test = check_class_distribution(validation_data)
class_distribution_consistency_validation_test = check_class_distribution(test_data)
class_distribution_consistency_validation_test = check_class_distribution(overfit_data)

def check_overlapping_samples(dataset1, dataset2):
    if dataset1.merge(dataset2, on=['text', 'label'], how='inner').empty:
        print("There is no overlap")

overlap_train_validation = check_overlapping_samples(train_data, validation_data)
overlap_train_test = check_overlapping_samples(train_data, test_data)
overlap_validation_test = check_overlapping_samples(train_data, validation_data)
