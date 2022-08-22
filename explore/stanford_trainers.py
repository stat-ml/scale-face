import torch
import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
import faiss
from pytorch_metric_learning import miners, losses
from tqdm import tqdm


class CrossEntropyTrainer:
    def __init__(self, model, epochs):
        self.model = model
        self.epochs = epochs

    def train(self, train_loader, val_loader):
        model = self.model
        optimizer = torch.optim.SGD(model.parameters(), lr=3e-3)
        criterion = torch.nn.CrossEntropyLoss()
        train_iter = 0
        writer = SummaryWriter()
        for epoch in tqdm(range(self.epochs)):
            model.train()
            for x, y in train_loader:
                optimizer.zero_grad()
                x, y = x.cuda(), y.cuda()
                preds = model(x)
                loss = criterion(preds, y)
                loss.backward()
                optimizer.step()
                writer.add_scalar('Loss/train', loss.item(), train_iter)
                train_iter += 1

            with torch.no_grad():
                model.eval()
                epoch_losses = []
                correct = []

                for x, y in val_loader:
                    x, y = x.cuda(), y.cuda()
                    preds = model(x)
                    loss = criterion(preds, y)
                    epoch_losses.append(loss.item())
                    correct.extend(list((torch.argmax(preds, dim=-1) == y).detach().cpu()))

                writer.add_scalar('Loss/val', np.mean(epoch_losses), train_iter)
                writer.add_scalar('Accuracy/val', np.mean(correct), train_iter)

        writer.close()
        return model


def knn_eval(model, loader, k=5, n_iter=0):
    embeddings = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            batch_embeddings = model(x.cuda())
            embeddings.extend(batch_embeddings.detach().cpu().tolist())
            labels.extend(y.tolist())
    embeddings = np.array(embeddings).astype(np.float32)

    index = faiss.IndexFlatL2(embeddings.shape[-1])
    index.add(embeddings.astype(np.float32))
    _, idx = index.search(embeddings, k+1)
    idx = torch.tensor(np.array(labels)[idx[:, 1:]])
    predictions = torch.mode(idx, dim=-1).values
    accuracy = (predictions == torch.tensor(labels)).to(torch.float).mean()
    return accuracy


class ScaleTrainer():
    pass




class PFETrainer:
    def __init__(self):
        pass

    def train(self, train_loader, val_loader):
        pass



class TripletsTrainer(CrossEntropyTrainer):
    def train(self, train_loader, val_loader):
        model = self.model
        optimizer = torch.optim.SGD(model.parameters(), lr=3e-3)
        miner = miners.MultiSimilarityMiner()
        criterion = losses.TripletMarginLoss()

        train_iter = 0
        writer = SummaryWriter()

        for epoch in tqdm(range(self.epochs)):
            model.train()
            for x, y in train_loader:
                optimizer.zero_grad()
                x, y = x.cuda(), y.cuda()
                model(x)
                embeddings = model.features
                hard_pairs = miner(embeddings, y)
                loss = criterion(embeddings, y, hard_pairs)
                loss.backward()
                optimizer.step()
                writer.add_scalar('Loss/train', loss.item(), train_iter)
                train_iter += 1

            with torch.no_grad():
                model.eval()
                epoch_losses = []

                for x, y in val_loader:
                    x, y = x.cuda(), y.cuda()
                    model(x)
                    loss = criterion(model.features, y)
                    epoch_losses.append(loss.item())

                writer.add_scalar('Loss/val', np.mean(epoch_losses), train_iter)
                accuracy = knn_eval(model, val_loader, k=1, n_iter=train_iter)
                writer.add_scalar('Accuracy/val', accuracy, train_iter)

        writer.close()
        return model


class ArcFaceTrainer():
    def __init__(self, model, epochs, embedding_size, num_classes):
        self.model = model
        self.epochs = epochs
        self.embedding_size = embedding_size
        self.num_classes = num_classes

    def train(self, train_loader, val_loader):
        model = self.model
        optimizer = torch.optim.SGD(model.parameters(), lr=3e-3)
        criterion = losses.ArcFaceLoss(self.num_classes, self.embedding_size)

        train_iter = 0
        writer = SummaryWriter()

        for epoch in tqdm(range(self.epochs)):
            model.train()
            for x, y in train_loader:
                optimizer.zero_grad()
                x, y = x.cuda(), y.cuda()
                embeddings = model(x)
                loss = criterion(embeddings, y)
                loss.backward()
                optimizer.step()
                writer.add_scalar('Loss/train', loss.item(), train_iter)
                train_iter += 1

            with torch.no_grad():
                model.eval()
                epoch_losses = []

                for x, y in val_loader:
                    x, y = x.cuda(), y.cuda()
                    embeddings = model(x)
                    loss = criterion(embeddings, y)
                    epoch_losses.append(loss.item())

                writer.add_scalar('Loss/val', np.mean(epoch_losses), train_iter)
                accuracy = knn_eval(model, val_loader, k=1, n_iter=train_iter)
                writer.add_scalar('Accuracy/val', accuracy, train_iter)

        writer.close()
        return model
