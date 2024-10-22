import matplotlib.pyplot as plt
import torch
import numpy as np
import tqdm

device = "mps" if torch.backends.mps.is_available() else "cpu"



def gradient(img, label, net, criterion):
    img.requires_grad = True
    logits = net(img)
    #logits = logits.long()
    label = label.long()
    loss = criterion(logits, label)

    loss.backward()

    return img.grad





def demo_grad(dataloader, net, criterion):
    def get_images(dataloader):
        try:
            for images, labels, human_labels in dataloader:
                image = images
                label = labels
                human_label = human_labels
                break
            return image[0].unsqueeze(0), label[0].unsqueeze(0), human_label[0]
        except:
            for images, labels in dataloader:
                image = images
                label = labels
                break
            return image[0].unsqueeze(0), label[0].unsqueeze(0), ""


    def plot_images(orig_img,grad_img, real_label, human_label):
        fig, ax = plt.subplots(1, 4, figsize=(10, 5))
        def process_img(img):
            try:
                return img.squeeze().transpose(1, 2, 0)
            except ValueError:
                return img.squeeze()

        ax[0].imshow(process_img(orig_img), cmap='gray')
        ax[0].set_title(f'Original')
        ax[0].axis('off')

        ax[1].text(0.5, 0.5, f'Real: {real_label.item()} {human_label}', fontsize=12, ha='center')
        ax[1].set_title(f'Prediction')
        ax[1].axis('off')

        ax[2].imshow(process_img(grad_img), cmap='gray')
        ax[2].set_title(f'Map')
        ax[2].axis('off')

        plt.tight_layout()
        plt.show()

    net.eval()
    orig_image,orig_label,human_label = get_images(dataloader)


    new_img = gradient(orig_image, orig_label, net, criterion)



    plot_images(orig_image.detach().cpu().numpy(), new_img.detach().cpu().numpy(), orig_label, human_label)