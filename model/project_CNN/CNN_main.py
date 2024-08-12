import os
import matplotlib.pyplot as plt
from CNN_train import *
import args_parameter

glove = torchtext.vocab.GloVe(name="6B", dim=100)

args = args_parameter.args


if __name__ == "__main__":

    model, train_loss_list,train_acc_list,val_loss_list,val_acc_list,test_acc_list,overfit_loss_list,overfit_acc_list = train(args)

    if args.of == False:
        print(test_acc_list[-1])
        plt.subplot(3, 1, 1)  
        plt.plot(train_loss_list, color="r", linestyle="-", linewidth=1, label="train_loss")  
        plt.plot(val_loss_list, color="b", linestyle="-",  linewidth=1, label="validation_loss") 
        plt.legend(loc='upper left')
        plt.title('Train and validation loss')  

        plt.subplot(3, 1, 2)  
        plt.plot(train_acc_list, color="r", linestyle="-", linewidth=1, label="train_accuracy")  
        plt.plot(val_acc_list, color="b", linestyle="-",  linewidth=1, label="validation_accuracy") 
        plt.legend(loc='upper left')
        plt.title('Train and Validation Accuracy')  
        
        plt.subplot(3, 1, 3) 
        plt.plot(test_acc_list, color="r", linestyle="-", linewidth=1, label="test_accuracy")  
        plt.title('Test Accuracy')  
        plt.legend(loc='upper left')
        #torch.save(model.state_dict(),'cnn2.pt')
        

    else:
        plt.subplot(2, 1, 1) 
        plt.plot(overfit_loss_list, color="r", linestyle="-", linewidth=1)  
        plt.title('Overfit_loss')  


        plt.subplot(2, 1, 2) 
        plt.plot(overfit_acc_list, color="r", linestyle="-", linewidth=1)  
        plt.title('Overfit_acc')  


    plt.tight_layout()  
    plt.show() 

    


