import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.utils.batch_memory_manager import BatchMemoryManager


def get_device():
    use_cuda = torch.cuda.is_available()
    assert use_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def train_ohm(model, train_loader, optimizer, criterion, epoch, device):
    model.train()
    losses = []
    acc = []

    for i, batch in enumerate(train_loader):
        images = batch[0].to(device)
        targets = batch[1].to(device)
        labels = targets if len(batch) == 2 else batch[2].to(device)
        # compute output
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, targets)
        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = labels.detach().cpu().numpy()
        # measure accuracy and record loss
        acc1 = (preds == labels).mean()
        losses.append(loss.item())
        acc.append(acc1)
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

    lossm = np.mean(losses)
    accm = np.mean(acc)
    print(
        f"Train epoch {epoch}:",
        f"Loss: {lossm:.6f} ",
        f"Acc@1: {accm * 100 :.6f} ",
    )

    return lossm, accm


def train_dpsgd(model, train_loader, phy_batch_size, optimizer, 
          privacy_engine, cross, delta, epoch, device):
    model.train()
    losses = []
    top1_acc = []

    with BatchMemoryManager(
        data_loader=train_loader, 
        max_physical_batch_size=phy_batch_size, 
        optimizer=optimizer
    ) as memory_safe_data_loader:
        for i, (images, target) in enumerate(memory_safe_data_loader):   
            optimizer.zero_grad()
            images = images.to(device)
            target = target.to(device)
            # compute output
            output = model(images)
            if cross == True:
                criterion = nn.CrossEntropyLoss()
                loss = criterion(output, target)
            else:
                loss = F.nll_loss(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            # measure accuracy and record loss
            acc = (preds == labels).mean()
            losses.append(loss.item())
            top1_acc.append(acc)
            loss.backward()
            optimizer.step()

    epsilon = privacy_engine.get_epsilon(delta)
    lossm = np.mean(losses)
    top1_accm = np.mean(top1_acc)
    print(f"Train Epoch: {epoch} \t"
          f"Loss: {lossm:.6f} \t"
          f"Acc@1: {top1_accm * 100:.6f} \t"
          f"(ε = {epsilon:.2f}, δ = {delta}) \t")

    return lossm, top1_accm, epsilon



def train_dfa(model, train_loader, optimizer, alignment, device, epoch):
    model.train()
    training_data = {"epoch": epoch, "loss": [], "alignment": None}
    for b, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        if b == 0:
            angles, alignments = alignment(data, target, F.nll_loss)
            training_data["alignment"] = alignments
            print(alignments)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        print(f"Train Epoch {epoch} \t"
              f"Loss at batch {b}/{len(train_loader)}: {loss.item():.4f}", end="\r")

        training_data["loss"].append(float(loss.item()))
    return training_data



def test(model, test_loader, cross, epoch, device):
    model.eval()
    losses = []
    top1_acc = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            if cross == True:
                criterion = nn.CrossEntropyLoss()
                loss = criterion(output, target)
            else:
                loss = F.nll_loss(output, target)

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = (preds == labels).mean()
            top1_acc.append(acc)
            losses.append(loss.item())

    top1_avg = np.mean(top1_acc)
    lossm = np.mean(losses)
    print(
        f"Test Epoch: {epoch} \t"
        f"Loss: {lossm:.6f} \t"
        f"Acc: {top1_avg * 100:.6f} \t")
    return lossm, top1_avg




