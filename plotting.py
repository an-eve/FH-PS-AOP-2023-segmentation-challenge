import matplotlib.pyplot as plt
import numpy as np
import torch
import random


def plot_loss_accuracy(train_loss_list, val_loss_list, val_acc_list):

    num_epochs = len(train_loss_list)
    # Create a figure with two subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(np.arange(1, num_epochs+1), train_loss_list, color="green", label='Train Loss')
    ax1.plot(np.arange(1, num_epochs+1), val_loss_list, color="red", label='Validation Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.legend()

    ax2.plot(np.arange(1, num_epochs+1), val_acc_list, color="blue", label='Validation Accuracy')
    ax2.set_title('Accuracy')
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()
    

def plot_img_mask_pred(dataset, index=None, plot_pred=False, model=None, device = "cuda"):
    if not index:
        index = random.randint(0, len(dataset) - 1)

    image = dataset[index][0].permute(1,2,0)
    mask = dataset[index][1].permute(1,2,0)

    if plot_pred:
        img_to_pred = dataset[index][0].unsqueeze(0).type(torch.float32).to(device)
        pred = model(img_to_pred)
        pred = pred.squeeze(0).cpu().detach().permute(1,2,0)
        pred[pred < 0.5]=0
        pred[pred > 0.5]=1


        # Plot the image
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 4, 1)
        plt.imshow(image)
        plt.title("Image")

        # Plot the mask
        plt.subplot(2, 4, 2)
        plt.imshow(mask[:,:,0])
        plt.title("GT Background")

        plt.subplot(2, 4, 3)
        plt.imshow(mask[:,:,1])
        plt.title("GT Pubic")

        plt.subplot(2, 4, 4)
        plt.imshow(mask[:,:,2])
        plt.title("GT Head")


        # Plot the image
        plt.subplot(2, 4, 5)
        plt.imshow(image)
        plt.title("Image")

        # Plot the predicted mask
        plt.subplot(2, 4, 6)
        plt.imshow(pred[:,:,0])
        plt.title("Prediction Background")

        plt.subplot(2, 4, 7)
        plt.imshow(pred[:,:,1])
        plt.title("Prediction Pubic")

        plt.subplot(2, 4, 8)
        plt.imshow(pred[:,:,2])
        plt.title("Prediction Head")

        # Show the plots
        plt.tight_layout()
        plt.show()

    else:
        # Plot the image
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 4, 1)
        plt.imshow(image)
        plt.title("Image")

        # Plot the mask
        plt.subplot(1, 4, 2)
        plt.imshow(mask[:,:,0])
        plt.title("GT Background")

        plt.subplot(1, 4, 3)
        plt.imshow(mask[:,:,1])
        plt.title("GT Pubic")

        plt.subplot(1, 4, 4)
        plt.imshow(mask[:,:,2])
        plt.title("GT Head")

        # Show the plots
        plt.tight_layout()
        plt.show()

