import torch
from torch import nn
from torchinfo import summary


class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_size=512, num_layers=2, dropout_rate=0.5):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size = embedding_dim,
            hidden_size = hidden_size,
            num_layers = num_layers,
            dropout=dropout_rate,
            batch_first=True
        )
        """
        Args:
            sequence: 输入序列 [batch_size, seq_len, embedding_dim]
            hs: 初始隐藏状态 (h_0, c_0) 或 None (自动零初始化)
        Returns:
            output: 最后一层输出向量 [batch_size, hidden_embedding_dim] 
            hs: 所以层隐藏状态 (l_h_n, l_c_n), [batch_size, hidden_embedding_dim] 
        """
    
        self.linear = nn.Linear(hidden_size, embedding_dim)

    def forward(self, sequence, hs=None):     
        out, hs = self.lstm(sequence, hs)     
        last_out = out[:, -1, :]              
        output = self.linear(last_out)   
        return output, hs


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    batch_size = 32
    seq_len = 50
    embedding_dim = 5000

    model = LSTM(embedding_dim, 512, 2, 0.5).to(device)
    # input shape: [batch_size, seq_len, embedding_dim]
    summary(model, input_size=(batch_size, seq_len, embedding_dim))