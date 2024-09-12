import torch

from CNN import CNN
from agg_model import HetAggModel


class HetFLModel(torch.nn.Module):
    def __init__(self, input_embed_d=300, output_embed_d=500, heads=4):
        super(HetFLModel, self).__init__()
        self.heads = heads
        self.agg_model = HetAggModel(input_embed_d=input_embed_d, output_embed_d=output_embed_d, heads=self.heads)
        self.cnn = CNN(channel_in=self.heads)

    def forward(self, passed_test_cases, failed_test_cases, history_change, call_information, methods):
        x = self.agg_model(passed_test_cases, failed_test_cases, history_change, call_information, methods)
        x = self.cnn(x)
        return x
