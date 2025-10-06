import json
import torch

from torchsummary import summary
from models.model import PretrainedModel, DownstreamClassifier, Encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instancier le modèle pré-entraîné et charger les poids
pretrain_model_name = 'model_v3.2'

# Charger la config du pré-entraînement pour les hyperparamètres
with open(f'results/pretrain/{pretrain_model_name}/used_config.json', 'r') as f:
    pretrain_config = json.load(f)
pretrain_params = pretrain_config['model']

input_size = 1024
# Logging the backbone
backbone = Encoder(
    num_patch= input_size // pretrain_params.get('patch_size', 64),
    patch_size= pretrain_params.get('patch_size', 64),
    encoder_dim= pretrain_params.get('encoder_dim', 128),
    n_layers= pretrain_params.get('n_layers', 4),
    heads= pretrain_params.get('heads', 8),
    dropout= pretrain_params.get('dropout', 0.4),
).to(device)

# Downstrean factory
model = DownstreamClassifier(
    backbone= backbone,
    num_classes= 10,
    freeze_backbone= False,
).to(device)

print("u")

for name, parameter in model.named_parameters():
    print(f'{name} : {parameter.data.size()}')