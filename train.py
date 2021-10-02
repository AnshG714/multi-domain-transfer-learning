import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from datasets import DomainDataset
from utils.checkpoints import load_checkpoint, save_checkpoint
import transforms
from model import Net

BATCH_SIZE = 64
NUM_WORKERS = 0

cars_train_ds = DomainDataset(
  './datasets/cars/cars_train', 
  './datasets/cars/train_csv.csv', 
  transform=transforms.train_transforms
)
cars_train = DataLoader(cars_train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)


cars_test_ds = DomainDataset(
  './datasets/cars/cars_test', 
  './datasets/cars/train_csv.csv', 
  transform=transforms.val_transforms
)
cars_test = DataLoader(cars_test_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

cars_val_ds = DomainDataset(
  './datasets/cars/cars_val', 
  './datasets/cars/val_csv.csv', 
  transform=transforms.val_transforms
)
cars_val = DataLoader(cars_val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)


def train_fn(loader, model, optimizer, loss_fn, scaler, device):
  for batch_idx, (data, targets) in enumerate(loader):
    # Get data to cuda if its available
    data = data.to(device=device)
    targets = targets.to(device=device)

    with torch.cuda.amp.autocast():
      scores = model(data)
      loss = loss_fn(scores, targets.float())

  optimizer.zero_grad()
  scaler.scale(loss).backward()
  scaler.step(optimizer)
  scaler.update()

loss_fn = nn.CrossEntropyLoss()
model = Net(net_version="b0", num_classes=102).to("cpu")
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
n_epochs = 100








