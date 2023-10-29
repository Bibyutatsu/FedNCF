from dataloader import MovielensDatasetLoader
from model import NeuralCollaborativeFiltering
from server_model import ServerNeuralCollaborativeFiltering
from train_single import MatrixLoader
import torch
import numpy as np
from metrics import compute_metrics

def layer_setter(model_source, model_copy):
    for m, mc in zip(model_source.parameters(), model_copy.parameters()):
        mc.data[:] = m.data[:]

def set_weights(model_source, model_copy):
    layer_setter(model_source.mlp_item_embeddings, model_copy.mlp_item_embeddings)
    layer_setter(model_source.gmf_item_embeddings, model_copy.gmf_item_embeddings)
    layer_setter(model_source.mlp, model_copy.mlp)
    layer_setter(model_source.gmf_out, model_copy.gmf_out)
    layer_setter(model_source.mlp_out, model_copy.mlp_out)
    layer_setter(model_source.output_logits, model_copy.output_logits)


## Loading the server model
## This model will be on the centralized server and user is going to load it on his local device. It contains
## the item embeddings
server_model = torch.jit.load('./models/central/server49.pt').eval()

## Loading the local model for Client 0 
## Copy the server weights into the local model weights
local_model = torch.jit.load('./models/local/dp0.pt').eval()
set_weights(server_model, local_model)

## Loading the data we want to infer on
dataloader = MovielensDatasetLoader()
num_users = list(local_model.gmf_user_embeddings.parameters())[0].shape[0]
loader = MatrixLoader(dataloader.ratings[:num_users])

## From the entire movie data we are loading 10 data points
## x is the USER ID
## y is the movie ID for which we want the prediction
x, y = loader.get_batch(10)

## Predictions
with torch.no_grad():
    predictions = local_model(x).detach().numpy()

## Calculation of Hit Rate and NDCG (normalized discounted cumulative gain)
metrics = compute_metrics(y.numpy(), predictions)
print(f"Hit Rate: {metrics[0]}\nNDCG    : {metrics[1]}")