import torch
import torchtext
import CNN_model
import args_parameter

args = args_parameter.args
glove = torchtext.vocab.GloVe(name="6B", dim=100)


def print_closest_cosine_words(vec, n):
    dists = torch.cosine_similarity(vec.unsqueeze(0),glove.vectors, dim=1)
    lst = sorted(enumerate(dists.numpy()), key=lambda x: x[1], reverse = True)
    for idx, difference in lst[1:n+1]:
        print(glove.itos[idx], "\t%5.2f" % difference)

if __name__ == "__main__":
    model = CNN_model.CNNTextClassifier(glove,args)
    model.load_state_dict(torch.load('/cnn.pt'))
    k = model.state_dict()

    print("layer1")

    weights = model.cnn_layer1.weight
    weights = weights.squeeze()
    print(weights.size()) # [20,2,100]
    reshaped_weights = weights.reshape(40, 100)
    print(reshaped_weights.size())
    for word in reshaped_weights:
        print_closest_cosine_words(word.detach(),5)
        print('-------------------------------------------')

    print("layer2")

    weights2 = model.cnn_layer2.weight
    weights2 = weights2.squeeze()
    print(weights2.size()) # [30,4,100]
    reshaped_weights2 = weights2.reshape(120, 100)
    print(reshaped_weights2.size())
    for word in reshaped_weights2:
        print_closest_cosine_words(word.detach(),5)
        print('-------------------------------------------')
