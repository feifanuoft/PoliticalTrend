import torch
import torchtext
import torch.nn as nn
import Project_Starter
from Project_Starter import *
import CNN_model 
import Read_file



train_dataset= Read_file.train_dataset
val_dataset= Read_file.val_dataset
overfitdata = Read_file.overfitdata
test_dataset= Read_file.test_dataset

def train(args):
    #   fix seed
    torch.manual_seed(2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print ("Using device:", device)
    
    ### 3.3 Processing of the data ###
    # 3.3.1
    # The first time you run this will download a 862MB size file to .vector_cache/glove.6B.zip
    glove = torchtext.vocab.GloVe(name="6B",dim=100) # embedding size = 100
                                   
    # 3.3.2
    learning_rate =  args.lr

    train_dataset = TextDataset(glove, "train")
    val_dataset = TextDataset(glove, "validation")
    test_dataset = TextDataset(glove, "test")
    overfit_dataset = TextDataset(glove,"overfit")
    
        
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
    
    overfit_dataloader = torch.utils.data.DataLoader(
        dataset=overfit_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: my_collate_function(batch, device))
    
    # model = CNN_model.CNNTextClassifier(glove,args.k1,args.n1,args.k2,args.n2,args.freeze_embedding,args.bias)
    model = CNN_model.CNNTextClassifier(glove,args)
    lossfunction = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    
    train_loss_list = []
    train_acc_list = []

    val_loss_list = []
    val_acc_list = []

    test_acc_list = []

    overfit_loss_list = []
    overfit_acc_list = []


    for epoch in range(args.epochs):
        if args.of == False:
            train_acc = 0
            train_loss = 0
            # train------------------------------
            for X_train, Y_train in train_dataloader:  
                model.train()
                optimizer.zero_grad()
                out = model(X_train)
                loss = lossfunction(out, Y_train.float())
                train_loss += loss.item()

                logit = torch.sigmoid(out)
                Y_train_pred = torch.round(logit).long()
                for i in range(len(Y_train_pred)):
                        if Y_train_pred[i] == Y_train[i]:
                            train_acc += 1
                loss.backward()
                optimizer.step()
                model.eval()

            
            train_loss /= len(train_dataloader)
            train_acc = train_acc/len(train_dataset)
            train_loss_list.append(train_loss)  
            train_acc_list.append(train_acc)
            print((f'Epoch {epoch + 1}/{args.epochs}, trainLoss: {train_loss:.4f}, train_acc: {train_acc:.4f}'))
            # print("The training accuracy is",train_acc)

            # validation --------------------------------------
            val_loss = 0
            val_acc = 0
            for X_val, Y_val in validation_dataloader: 
                val_pred = model(X_val)
                loss = lossfunction(val_pred, Y_val.float())
                val_loss += loss.item()
                
                logit = torch.sigmoid(val_pred)
                Y_val_pred = torch.round(logit).long()
                for i in range(len(Y_val_pred)):
                    if Y_val_pred[i] == Y_val[i]:
                        val_acc += 1

            
            val_loss /= len(validation_dataloader)    
            val_loss_list.append(val_loss)            
            val_acc /= len(val_dataset)                    
            val_acc_list.append(val_acc)  
            print((f'Epoch {epoch + 1}/{args.epochs}, validation loss: {val_loss:.4f}, val_acc: {val_acc:.4f}'))                          


            #test ---------------------------------
            test_acc = 0
            for X_test, Y_test in test_dataloader:
                
                test_pred = model(X_test)
                logit = torch.sigmoid(test_pred)
                Y_test_pred = torch.round(logit).long()
                for i in range(len(Y_test_pred)):
                    if Y_test_pred[i] == Y_test[i]:
                        test_acc += 1

            test_acc /= len(test_dataset)
            test_acc_list.append(test_acc)
            print("test accuracy: {}".format(test_acc))

        else:
            #overfit-------------------------------
            overfit_acc = 0
            overfit_loss = 0
            for X_of, Y_of in overfit_dataloader:  
                model.train()
                optimizer.zero_grad()
                out = model(X_of)
                loss = lossfunction(out, Y_of.float())
                overfit_loss += loss.item()

                logit = torch.sigmoid(out)
                Y_of_pred = torch.round(logit).long()
                for i in range(len(Y_of_pred)):
                        if Y_of_pred[i] == Y_of[i]:
                            overfit_acc += 1
                loss.backward()
                optimizer.step()
                model.eval()

            
            overfit_loss /= len(overfit_dataloader)
            overfit_acc = overfit_acc/len(overfit_dataset)
            overfit_loss_list.append(overfit_loss)  
            overfit_acc_list.append(overfit_acc)
            print(f'Epoch {epoch + 1}/{args.epochs}, overfit_acc_loss: {overfit_loss:.4f},overfit_acc: {overfit_acc:.4f}')
    
    
    return model, train_loss_list,train_acc_list,val_loss_list,val_acc_list,test_acc_list,overfit_loss_list,overfit_acc_list
    




