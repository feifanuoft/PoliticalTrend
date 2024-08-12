import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-batch_size", type=int, default=16)  #batch_size
parser.add_argument("-of", type=bool, default=False
                    )   # use overfit dataset or not, false = "not"
parser.add_argument("-bias", type=bool, default=False) #bias
# k1,k2,n1,n2
parser.add_argument("-k1", type=int, default=2) 
parser.add_argument("-n1", type=int, default=20)
parser.add_argument("-k2", type=int, default=4)
parser.add_argument("-n2", type=int, default=30)
parser.add_argument("-lr", type=int, default=0.0005) # learning rate
parser.add_argument("-epochs", type=int, default=20) # epoch
parser.add_argument("-freeze_embedding", type=bool, default=True) #freeze_embedding
args = parser.parse_args()
