import torch
import torch.nn as nn

from src.models.model import Model
from src.utils.config_parser import Config
from lib.glass_defect_dataset.config.consts import General as _CG


class BiLSTM(Model):

    def __init__(self,config: Config, input_size: int, hidden_size: int, batch_size: int, n_lay: int=1):
        super().__init__(config)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_lay = n_lay

        # batch_first = False, so input is provided as (seq_len, batch_size, features)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=n_lay, bidirectional=True)
        self.d = 2 if self.lstm.bidirectional else 1
    
    def forward(self, *input_):
        assert len(input_) == 1 or len(input_) == 3, f"{len(input_)} values inserted. Accepted only 1 or 3"
        x = input_[0]
        h0 = torch.randn(self.d * self.n_lay, self.batch_size, self.hidden_size, device=_CG.DEVICE)
        c0 = torch.randn(self.d * self.n_lay, self.batch_size, self.hidden_size, device=_CG.DEVICE)
        if len(input_) > 1:
            h0 = input_[1]
            c0 = input_[2]
        out, (h_, c_) = self.lstm(x, (h0, c0))
        return out, h_, c_