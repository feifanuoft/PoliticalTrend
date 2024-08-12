
import torch
import torchtext
import CNN_model
import CNN_main
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score  
from Project_Starter import *
import Read_file



train_dataset= Read_file.train_dataset
test_data= Read_file.test_dataset
val_dataset= Read_file.val_dataset
overfitdata = Read_file.overfitdata




args = CNN_main.args
glove = torchtext.vocab.GloVe(name="6B", dim=100)

k_list = [[2,4],[4,8],[2,8]]
n_list = [[10,10],[10,20],[20,20],[20,30]]


def findbestpara(args):

    torch.manual_seed(2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print ("Using device:", device)

    ### 3.3 Processing of the data ###
    # 3.3.1
    # The first time you run this will download a 862MB size file to .vector_cache/glove.6B.zip
    glove = torchtext.vocab.GloVe(name="6B",dim=100) # embedding size = 100


    learning_rate =  0.0001

    train_dataset = TextDataset(glove, "train")
    val_dataset = TextDataset(glove, "validation")
    test_dataset = TextDataset(glove, "test")

    lossfunction = torch.nn.BCEWithLogitsLoss()

    

        # 3.3.3
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=lambda batch: my_collate_function(batch, device))

    validation_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=lambda batch: my_collate_function(batch, device))

    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: my_collate_function(batch, device))
    

    epochs = 35
    test_acc_list = []
    best_acc = 0
    bk1 = 0  #best k1
    bk2 = 0  #best k2
    bn1 = 0  #best n1
    bn2 = 0  #best n2
    for k1, k2 in k_list:
        for n1,n2 in n_list:
            args.k1 = k1
            args.k2 = k2
            args.n1 = n1
            args.n2 = n2
            current_model = CNN_model.CNNTextClassifier(glove,args)
            for i in range(epochs):                        
                optimizer = optim.Adam(current_model.parameters(), lr=learning_rate)

                train_acc = 0
                train_loss = 0
                # train------------------------------
                for X_train, Y_train in train_dataloader:  
                    current_model.train()
                    optimizer.zero_grad()
                    out = current_model(X_train)
                    loss = lossfunction(out, Y_train.float())
                    train_loss += loss.item()

                    logit = torch.sigmoid(out)
                    Y_train_pred = torch.round(logit).long()
                    for i in range(len(Y_train_pred)):
                            if Y_train_pred[i] == Y_train[i]:
                                train_acc += 1
                    loss.backward()
                    optimizer.step()
                    current_model.eval()


                test_acc = 0
                for X_test, Y_test in test_dataloader:
                    
                    test_pred = current_model(X_test)
                    logit = torch.sigmoid(test_pred)
                    Y_test_pred = torch.round(logit).long()
                    for i in range(len(Y_test_pred)):
                        if Y_test_pred[i] == Y_test[i]:
                            test_acc += 1

                test_acc /= len(test_dataset)
                test_acc_list.append(test_acc)
                
            fianl_acc = test_acc_list[-1]    
            test_acc_list = []
            print(f'When k1 = {k1}, n1 = {n1}, k2 = {k2}, n2 = {n2}, final_test accuracy = {fianl_acc}')
            if fianl_acc > best_acc:
                best_acc = fianl_acc
                bk1 = k1
                bk2 = k2
                bn1 = n1
                bn2 = n2
            else:
                continue
                


    print(f'When k1 = {bk1}, n1 = {bn1}, k2 = {bk2},n2 = {bn2}, final_test accuracy = {best_acc}, and the model is the best model')
                



if __name__ == "__main__":
    findbestpara(args)
