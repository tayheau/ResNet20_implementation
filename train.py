import argparse
import torch
import numpy as np
from tqdm import tqdm
from utils._utils import make_data_loader
from model import ResNet20, ResNet9
import matplotlib.pyplot as plt


def acc(pred,label):
    pred = pred.argmax(dim=-1)
    return torch.sum(pred == label).item()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(args, data_loader, validation_loader, model):
    """
    TODO: Change the training code. (e.g. different optimizer, different loss function, etc.)
            Add validation code. -> This will increase the accuracy.
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.learning_rate, epochs = args.epochs, steps_per_epoch=len(tqdm(data_loader)))
   
    train_losses_total = []
    valid_losses_total = []
   

    for epoch in range(args.epochs):
        train_losses = [] 
        train_acc = 0.0
        total=0
        lr = [] 
        print(f"[Epoch {epoch+1} / {args.epochs}]")
        
        model.train()
        pbar = tqdm(data_loader)
        for i, (x, y) in enumerate(pbar):
            image = x.to(args.device)
            label = y.to(args.device)          
            optimizer.zero_grad()

            output = model(image)
            
            label = label.squeeze()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            lr.append(get_lr(optimizer))
            scheduler.step()

            train_losses.append(loss.item())
            

            total += label.size(0)

            train_acc += acc(output, label)

        
        epoch_train_loss = np.mean(train_losses)
        epoch_train_acc = train_acc/total
        train_losses_total.append(epoch_train_loss)
        print(f'Epoch {epoch+1}') 
        print(f'train_loss : {epoch_train_loss}')
        print('train_accuracy : {:.3f}'.format(epoch_train_acc*100))
        #print('last lr : {:.3f}'.format(lr[-1]) )

        #Validation
        model.eval()
        valid_losses = []
        valid_acc = 0.0
        total_valid = 0

        with torch.no_grad():
            for x_valid, y_valid in validation_loader:
                image_valid = x_valid.to(args.device)
                label_valid = y_valid.to(args.device)

                output_valid = model(image_valid)

                label_valid = label_valid.squeeze()
                loss_valid = criterion(output_valid, label_valid)

                valid_losses.append(loss_valid.item())

                total_valid += label_valid.size(0)

                valid_acc += acc(output_valid, label_valid)

        epoch_valid_loss = np.mean(valid_losses)
        epoch_valid_acc = valid_acc / total_valid

        valid_losses_total.append(epoch_valid_loss)

        print('validation_loss : {:.3f}'.format(epoch_valid_loss))
        print('validation_accuracy : {:.3f}'.format(epoch_valid_acc * 100))
        

        torch.save(model.state_dict(), f'{args.save_path}/model_A16.pth')

    #Ploting
    plt.plot( train_losses_total, label='Train loss')
    plt.plot(valid_losses_total, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot - ResNet20, 8- 0.03 - 500')
    plt.legend()
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='2023 DL Term Project')
    parser.add_argument('--save-path', default='checkpoints/', help="Model's state_dict")
    parser.add_argument('--data', default='data/', type=str, help='data folder')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    """
    TODO: Change the hyperparameters.
            (e.g. change epochs etc.)
    """
    
    # hyperparameters
    args.epochs = 8
    args.learning_rate = 0.03
    args.batch_size = 5

    # check settings
    print("==============================")
    print("Save path:", args.save_path)
    print('Using Device:', device)
    print('Number of usable GPUs:', torch.cuda.device_count())
    
    # Print Hyperparameter
    print("Batch_size:", args.batch_size)
    print("learning_rate:", args.learning_rate)
    print("Epochs:", args.epochs)
    print("==============================")
    
    # Make Data loader and Model
    train_loader, validation_loader, _ = make_data_loader(args)

    model = ResNet20()
    model.to(device)

    # Training The Model
    train(args, train_loader, validation_loader, model)