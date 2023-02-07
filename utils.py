import torch
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Indicate the path to the dataset
dirs = {
    'train' : 'data/COVID-19_Radiography_Dataset/train',
    'val' : 'data/COVID-19_Radiography_Dataset/val',
    'test' : 'data/COVID-19_Radiography_Dataset/test'
}

#Set of transformations to be applied to the images
transform = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
}

#Return the predictions of the model on the dataloader
def get_all_preds(model, loader):
    # we dont want the model to change the weights during prediction so we set it to eval mode
    model.eval()
    # no need to compute gradients during predictions (save some memory and speed up computations)
    with torch.no_grad():
        all_preds = torch.tensor([], device=device)
        for batch in loader:
            images = batch[0].to(device)
            preds = model(images)
            # we concatenate the predictions of each batch
            all_preds = torch.cat((all_preds, preds), dim=0)

    return all_preds


# Return how many corrections are correct
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()



def get_confmat(test_set, test_preds):
    """return numpy array of confusion matrix

    Args:
        test_preds (pytorch tensor): predictions of the model (probabilities)

    Returns:
        _type_: confusion matrix (np array)
    """
    y_pred = torch.argmax(test_preds, dim=1)
    return confusion_matrix(test_set.targets, y_pred)
    


def deprocess_image(image):
    """ take a pytorch tensor (normalized) and return a numpy array which is denormalized (and clipped to 0-1)

    Args:
        image (tensor): _description_

    Returns:
        np array: denormalized image
    """
    image = image.cpu().numpy()
    image = np.squeeze(np.transpose(image[0], (1, 2, 0)))
    image = image * np.array((0.229, 0.224, 0.225)) + \
        np.array((0.485, 0.456, 0.406))  # un-normalize
    image = image.clip(0, 1)
    return image