import torch
import torchvision

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_densenet121(pretrained=False, out_features=None, path=None):
    model = torchvision.models.densenet121(pretrained=pretrained)
    if out_features is not None:
        model.classifier = torch.nn.Linear(
            in_features=1024, out_features=out_features
        )
    if path is not None:
        model.load_state_dict(torch.load(path, map_location=device))

    return model.to(device)