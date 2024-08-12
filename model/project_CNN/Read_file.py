import pandas as pd
import torchtext
glove = torchtext.vocab.GloVe(name="6B",dim=100) 
overfitdata = pd.read_csv('./model/project_CNN/data/overfit.tsv', sep='\t')
test_dataset = pd.read_csv('./model/project_CNN/data/test.tsv', sep='\t')
train_dataset = pd.read_csv('./model/project_CNN/data/train.tsv', sep='\t')
val_dataset = pd.read_csv('./model/project_CNN/data/validation.tsv', sep='\t')