"""
Decode summarization models (generate summaries) trained with finetune.py
Multi-GPU decoding not working yet.
"""

import argparse
import logging
import numpy as np
import json
import pickle

import torch
from torch.utils.data import DataLoader, SequentialSampler, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class RoleDataset(torch.utils.data.Dataset):

  def __init__(self,file_name):
    with open(file_name, 'r') as f:
        dataset = json.load(f)

    x=dataset['roles']
    y=dataset['protoroles']

    self.role_matrix = dataset['role_matrix'][0][0][0]

    self.x_train=torch.tensor(x)
    self.y_train=torch.tensor(y)

  def __len__(self):
    return len(self.y_train)
  
  def __getitem__(self,idx):
    return self.x_train[idx],self.y_train[idx]

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


def generate_summaries(args):
    kmeans = pickle.load(open("kmeans_8.pkl", "rb"))

    role_dataset = RoleDataset(args.dataset_path)
    train_set, val_set, test_set = torch.utils.data.random_split(role_dataset, [args.train_size, args.val_size, args.test_size])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set)

    inputDim = args.role_dim
    outputDim = args.protorole_dim
    model = linearRegression(inputDim, outputDim)
    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    epoch_losses = []
    epoch_val_losses = []

    for epoch in tqdm(range(args.epochs)):
        ########################### Training #####################################
        model.train()
        epoch_loss = []

        # Batch
        for j, (inp, tar) in enumerate(train_loader):
            pred = model.forward(inp)
            loss = criterion(pred, tar)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.detach().item())

        epoch_loss = sum(epoch_loss) / len(epoch_loss)
        epoch_losses.append(epoch_loss)


        ########################### Validation #####################################
        model.eval()
        epoch_val_loss = []

        for k, (inp, tar) in enumerate(val_loader):
            output = model(inp)
            loss = criterion(output, tar)
            epoch_val_loss.append(loss.detach().item())

        epoch_val_loss = sum(epoch_val_loss) / len(epoch_val_loss)
        epoch_val_losses.append(epoch_val_loss)

    correct = 0
    for h, (inp, tar) in enumerate(test_loader):
        tar_role = kmeans.predict(np.array(tar.tolist(), dtype='float32'))[0]
        output = model(inp)
        out_role = kmeans.predict(np.array(output.tolist(), dtype='float32'))[0]
        if tar_role == out_role:
            correct+=1
    accuracy = correct / args.test_size
    print("Accuracy on test set:")
    print(accuracy)

    role_matrix_protoroles = model(torch.tensor(role_dataset.role_matrix))
    role_matrix_roles = kmeans.predict(np.array(role_matrix_protoroles.tolist(), dtype='float32'))

    role_matrix_dataset = {'protoroles': [], 'roles': [], 'role_matrix': []}
    role_matrix_protoroles = role_matrix_protoroles.tolist()
    role_matrix_roles = role_matrix_roles.tolist()
    role_matrix = role_dataset.role_matrix

    for u in range(len(role_matrix_roles)):
        role_matrix_dataset['protoroles'].append(role_matrix_protoroles[u])
        role_matrix_dataset['roles'].append(role_matrix_roles[u])
        role_matrix_dataset['role_matrix'].append(role_matrix[u])

    name1 = 'role_matrix_dataset_' + str(args.run_id) + '.json'
    with open(name1, "w") as f:
        json.dump(role_matrix_dataset, f, indent=2)

    name2 = 'MSE_probe_' + str(args.run_id) + '.png'
    # Plot loss
    fig0=plt.figure(0)
    plt.plot(epoch_losses, color='red', label='train')
    plt.xlabel('Epoch number')
    plt.ylabel('Epoch loss')
    # Plot test loss
    plt.plot(epoch_val_losses, color='blue', label='val')
    plt.legend()
    #plt.savefig(name2)
    plt.show()


def run_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path", type=str, default="uds_train.json", help="",
    )
    parser.add_argument(
        "--train_size", type=int, default=15204, help="",
    )
    parser.add_argument(
        "--val_size", type=int, default=844, help="",
    )
    parser.add_argument(
        "--test_size", type=int, default=845, help="",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="",
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=8, help="",
    )
    parser.add_argument(
        "--role_dim", type=int, default=64, help="",
    )
    parser.add_argument(
        "--protorole_dim", type=int, default=14, help="",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="",
    )
    parser.add_argument(
        "--epochs", type=int, default=500, help="",
    )
    parser.add_argument(
        "--run_id", type=str, default="00", help="",
    )

    args = parser.parse_args()

    generate_summaries(args)


if __name__ == "__main__":
    run_generate()