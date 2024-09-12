import cachetools.func
import pytorch_lightning as pl
import torch

from HetFLModel import HetFLModel


class ListWiseLoss(torch.nn.Module):
    @cachetools.func.lfu_cache
    def get_minus_nlogn(self, n):
        return -n * torch.log(n)

    def forward(self, x, y):
        log_prob = torch.nn.functional.log_softmax(x, dim=-1)
        return (self.get_minus_nlogn(torch.count_nonzero(y)).to(y.device) - torch.sum(log_prob * y, dim=-1)).mean()


class TrainModel(pl.LightningModule):
    def __init__(self, input_embed_d=300, output_embed_d=500):
        super().__init__()
        self.input_embed_d = input_embed_d
        self.output_embed_d = output_embed_d
        self.model = HetFLModel(input_embed_d=self.input_embed_d, output_embed_d=self.output_embed_d, heads=64)
        self.example_input_array = torch.zeros((5, 10, 2, 300)), torch.zeros((5, 9, 2, 300)), torch.zeros(
            (5, 8, 3, 300)), torch.zeros((5, 7, 3, 300)), torch.zeros((5, 1, 3, 300))
        self.best_FR = 9999
        self.val_history = list()
        self.list_wise_loss = ListWiseLoss()

    def forward(self, passed_test_cases, failed_test_cases, history_change, call_information, methods):
        x = self.model(passed_test_cases, failed_test_cases, history_change, call_information, methods)
        return x

    def training_step(self, batch, batch_idx):
        batch_size = int(batch[-1].shape[0])
        passed_test_cases, failed_test_cases, history_change, call_information, methods, labels = batch
        preds = self(passed_test_cases, failed_test_cases, history_change, call_information, methods)
        preds = preds.squeeze(-1)
        loss = self.list_wise_loss(preds, labels)
        # loss = torch.nn.functional.l1_loss(predicts, y)
        self.log("train_loss", loss, logger=True, prog_bar=True, on_epoch=True, on_step=False,
                 batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        passed_test_cases, failed_test_cases, history_change, call_information, methods, labels = batch
        preds = self(passed_test_cases, failed_test_cases, history_change, call_information, methods)
        preds = preds.squeeze(-1).detach().cpu()
        labels = labels.detach().cpu()
        for t in zip(preds, labels):
            self.val_history.append((t[0].item(), t[1].item()))

    def validation_epoch_end(self, outputs):
        self.val_history.sort(key=lambda x: x[0], reverse=True)
        x = torch.tensor([t[0] for t in self.val_history])
        y = torch.tensor([t[1] for t in self.val_history])
        loss = self.list_wise_loss(x, y)
        p = 99999
        for i, t in enumerate(self.val_history):
            if int(t[1]) == 1:
                p = (i + 1)
                break
        self.val_history = list()
        self.log("val_loss", loss, logger=True, prog_bar=True, on_epoch=True)
        self.log("FR", p, logger=True, prog_bar=True, on_epoch=True)
        self.log("score", p + (loss.detach().cpu().item() * 1e-7), logger=True, prog_bar=True, on_epoch=True)
        if self.best_FR > p:
            self.best_FR = p
        self.log("best_FR", self.best_FR, logger=True, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer
