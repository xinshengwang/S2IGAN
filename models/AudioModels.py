import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from cfg.Pretrain.config import cfg

# class for making multi headed attenders.
class multi_attention(nn.Module):
    def __init__(self, in_size, hidden_size, n_heads):
        super(multi_attention, self).__init__()
        self.att_heads = nn.ModuleList()
        for x in range(n_heads):
            self.att_heads.append(attention(in_size, hidden_size))
    def forward(self, input):
        out, self.alpha = [], []
        for head in self.att_heads:
            o = head(input)
            out.append(o) 
            # save the attention matrices to be able to use them in a loss function
            self.alpha.append(head.alpha)
        # return the resulting embedding 
        return torch.cat(out, 1)

class attention(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(attention, self).__init__()
        self.hidden = nn.Linear(in_size, hidden_size)
        nn.init.orthogonal(self.hidden.weight.data)
        self.out = nn.Linear(hidden_size, in_size)
        nn.init.orthogonal(self.hidden.weight.data)
        self.softmax = nn.Softmax(dim = 1)
    def forward(self, input):
        # calculate the attention weights
        self.alpha = self.softmax(self.out(nn.functional.tanh(self.hidden(input))))
        # apply the weights to the input and sum over all timesteps
        x = torch.sum(self.alpha * input, 1)
        # return the resulting embedding
        return x 


class CNN_ENCODER(nn.Module):
    def __init__(self, embedding_dim=2048):
        super(CNN_ENCODER, self).__init__()
        self.embedding_dim = embedding_dim
        self.batchnorm1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(40,1), stride=(1,1), padding=(0,0))
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(1,11), stride=(1,1), padding=(0,5))
        self.conv3 = nn.Conv2d(256, 256, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.conv5 = nn.Conv2d(512, 512, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.pool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2),padding=(0,1))
        self.gpool = nn.AvgPool2d(kernel_size = (1,64),stride=(1,1),padding =(0,0))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.batchnorm1(x)
        x = F.relu(self.conv1(x))     #128*1*2048
        x = self.pool(x)              #1024
        x = F.relu(self.conv2(x))     #256*1*1024
        x = self.pool(x)              #256*1*512
        x = F.relu(self.conv3(x))     #512*1*512
        x = self.pool(x)              #512*1*256
        x = F.relu(self.conv4(x))     #1024*1*256
        x = self.pool(x)              #1024*1*128
        x = F.relu(self.conv5(x))     #1024*1*128
        x = self.pool(x)              #1024*1*128
        x = F.relu(self.conv6(x))     #2048*1*64
        x = self.gpool(x)
        # x = x.squeeze(2)
        x = x.view(x.size(0), -1)
        return nn.functional.normalize(x,p=2,dim=1)



class RNN_ENCODER(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers ,dropout=0.0, bidirectional=True):
        super(RNN_ENCODER, self).__init__()

        self.rnn_type = cfg.rnn_type

        if self.rnn_type == 'LSTM':
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(input_size, hidden_size , num_layers, batch_first=True, dropout=dropout,
                          bidirectional=bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout,
                          bidirectional=bidirectional)
        else:
            raise NotImplementedError
        
    def forward(self, input_x, enc_len):
        total_length = input_x.size(1)  # get the max sequence length
        # print('total_length: ' + str(total_length))
        # print('input_x.size(): ' + str(input_x.size()))
        packed_input = pack_padded_sequence(input_x, enc_len, batch_first=True)
        # print('enc_len: ' + str(enc_len))
        packed_output, hidden = self.rnn(packed_input)
        
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=total_length)
        local_emb = output.transpose(1, 2)
        global_emb = F.avg_pool1d(local_emb,kernel_size = total_length)
        global_emb = global_emb.squeeze(-1)
        # if self.rnn_type == 'LSTM':
        #     global_emb = hidden[0].transpose(0, 1).contiguous()
        # else:
        #     global_emb = hidden.transpose(0, 1).contiguous()       
        
        return  global_emb



class CNN_RNN_ENCODER(nn.Module):
    def __init__(self):
        super(CNN_RNN_ENCODER,self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=cfg.CNNRNN.in_channels,out_channels=cfg.CNNRNN.hid_channels,
                              kernel_size=cfg.CNNRNN.kernel_size,stride=cfg.CNNRNN.stride,
                              padding=cfg.CNNRNN.padding)
        self.Conv2 = nn.Conv1d(in_channels=cfg.CNNRNN.hid_channels,out_channels=cfg.CNNRNN.out_channels,
                              kernel_size=cfg.CNNRNN.kernel_size,stride=cfg.CNNRNN.stride,
                              padding=cfg.CNNRNN.padding)
        
        # self.Conv3 = nn.Conv1d(in_channels=cfg.CNNRNN.hid2_channels,out_channels=cfg.CNNRNN.out_channels,
        #                       kernel_size=cfg.CNNRNN.kernel_size,stride=cfg.CNNRNN.stride,
        #                       padding=cfg.CNNRNN.padding)
        self.bnorm1 = nn.BatchNorm1d(cfg.CNNRNN.hid_channels)
        self.bnorm2 = nn.BatchNorm1d(cfg.CNNRNN.out_channels)
        # self.bnorm3 = nn.BatchNorm1d(cfg.CNNRNN.out_channels)
        if cfg.CNNRNN.rnn_type == 'LSTM':
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(cfg.CNNRNN_RNN.input_size, cfg.CNNRNN_RNN.hidden_size , cfg.CNNRNN_RNN.num_layers, batch_first=True, dropout=cfg.CNNRNN_RNN.dropout,
                          bidirectional=cfg.CNNRNN_RNN.bidirectional)
        elif cfg.CNNRNN.rnn_type == 'GRU':
            self.rnn = nn.GRU(cfg.CNNRNN_RNN.input_size, cfg.CNNRNN_RNN.hidden_size , cfg.CNNRNN_RNN.num_layers, batch_first=True, dropout=cfg.CNNRNN_RNN.dropout,
                          bidirectional=cfg.CNNRNN_RNN.bidirectional)
        else:
            raise NotImplementedError

        self.att = multi_attention(in_size = cfg.CNNRNN_ATT.in_size, hidden_size = cfg.CNNRNN_ATT.hidden_size, n_heads = cfg.CNNRNN_ATT.n_heads)
    def forward(self, input, l):
            input = input.transpose(2,1)
            x = self.Conv1(input)
            x = self.bnorm1(x)
            x = self.Conv2(x)
            x = self.bnorm2(x)
            # x = self.Conv3(x)
            # x = self.bnorm3(x)

            # update the lengths to compensate for the convolution subsampling
            l = [int((y-(self.Conv1.kernel_size[0]-self.Conv1.stride[0]))/self.Conv1.stride[0]) for y in l]
            l = [int((y-(self.Conv2.kernel_size[0]-self.Conv2.stride[0]))/self.Conv2.stride[0]) for y in l]
            # l = [int((y-(self.Conv3.kernel_size[0]-self.Conv3.stride[0]))/self.Conv3.stride[0]) for y in l]
            # create a packed_sequence object. The padding will be excluded from the update step
            # thereby training on the original sequence length only
            x = torch.nn.utils.rnn.pack_padded_sequence(x.transpose(2,1), l, batch_first=True)
            # self.rnn.flatten_parameters()
            x, hx = self.rnn(x)
            # unpack again as at the moment only rnn layers except packed_sequence objects
            x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)

            if cfg.SPEECH.self_att:
                x = self.att(x)
            else:
                x = x.mean(dim=1)
            x = nn.functional.normalize(x, p=2, dim=1)    
            return x

