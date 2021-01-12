import torch
from argparse import Namespace
from torch.nn import functional as F, init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AdamW, BertConfig, BertForSequenceClassification, \
    get_constant_schedule_with_warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau


class LSTM_MODEL(torch.nn.Module):
    def __init__(self,
                 embeddings,
                 args: Namespace,
                 n_labels: int,
                 device='cuda'):
        super().__init__()
        self.args = args
        self.n_labels = n_labels
        self.device = device
        # TODO embeddings name, too
        self.embedding = torch.nn.Embedding(embeddings.size(0), embeddings.size(1))
        self.embedding.weight = embeddings
        # TODO extract other params, too: bidirectional
        self.enc_p = torch.nn.LSTM(input_size=args.embedding_dim,
                                   hidden_size=args.hidden_lstm,
                                   num_layers=args.num_layers,
                                   bidirectional=True,
                                   dropout=args.dropout,
                                   batch_first=True)

        self.dropout = torch.nn.Dropout(args.dropout)

        self.hidden_layers = torch.nn.ModuleList()
        self.hidden_layers.append(
            torch.nn.Linear(args.hidden_lstm * 2, args.hidden_sizes[0]))
        for i in range(1, len(args.hidden_sizes)):
            self.hidden_layers.append(torch.nn.Linear(args.hidden_sizes[i - 1],
                                                      args.hidden_sizes[i]))

        self.hidden_layers.append(
            torch.nn.Linear(args.hidden_sizes[-1], n_labels))

        self.hidden_layers.apply(self.init_weights)
        for name, param in self.enc_p.named_parameters():
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.xavier_uniform_(param)

    def init_weights(self, w):
        if isinstance(w, torch.nn.Linear):
            torch.nn.init.xavier_normal_(w.weight)
            w.bias.data.fill_(0.01)

    def forward(self, input, seq_lenghts=None):
        embedded = self.embedding(input)

        if seq_lenghts == None:
            seq_lenghts = []
            for instance in input:
                ilen = 1
                for _i in range(len(instance) - 1, -1, -1):
                    if instance[_i] != 0:
                        ilen = _i + 1
                        break
                seq_lenghts.append(ilen)

        packed_input = pack_padded_sequence(embedded, seq_lenghts,
                                            batch_first=True,
                                            enforce_sorted=False)
        lstm_out, self.hidden = self.enc_p(packed_input)
        output, input_sizes = pad_packed_sequence(lstm_out, batch_first=True)

        last_idxs = (input_sizes - 1).to(self.device)
        output = torch.gather(output, 1,
                              last_idxs.view(-1, 1).unsqueeze(2).repeat(1, 1,
                                                                        self.args.hidden_lstm * 2)).squeeze()

        for hidden_layer in self.hidden_layers:
            output = self.dropout(hidden_layer(output))

        return output


class CNN_MODEL(torch.nn.Module):
    def __init__(self,
                 embeddings,
                 args: Namespace,
                 n_labels: int = 2):
        super().__init__()
        self.n_labels = n_labels
        self.args = args

        self.embedding = torch.nn.Embedding(embeddings.size(0),
                                            embeddings.size(1))

        self.dropout = torch.nn.Dropout(args.dropout)

        self.embedding.weight = embeddings

        self.conv_layers = torch.nn.ModuleList(
            [torch.nn.Conv2d(args.in_channels, args.out_channels,
                             (kernel_height, args.embedding_dim),
                             args.stride, args.padding)
             for kernel_height in args.kernel_heights])

        self.final = torch.nn.Linear(
            len(args.kernel_heights) * args.out_channels, n_labels)

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)
        # TODO: extract this
        activation = F.relu(conv_out.squeeze(3))
        # TODO: extract pooling
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)

        return max_out

    def forward(self, input):
        input = self.embedding(input)
        input = input.unsqueeze(1)
        input = self.dropout(input)

        conv_out = [self.conv_block(input, self.conv_layers[i]) for i in
                    range(len(self.conv_layers))]
        all_out = torch.cat(conv_out, 1)
        fc_in = self.dropout(all_out)
        logits = self.final(fc_in)
        return logits


def get_model(model_args, device, embeddings=None):
    if model_args.model == 'transformer':
        transformer_config = BertConfig.from_pretrained('bert-base-uncased',
                                                        num_labels=model_args.labels)
        if model_args.init_only:
            model = BertForSequenceClassification(config=transformer_config).to(
                device)
        else:
            model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                config=transformer_config).to(device)
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if
                           not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in param_optimizer if
                           any(nd in n for nd in no_decay)], 'weight_decay': 0.0
            }]

        optimizer = AdamW(optimizer_grouped_parameters, lr=model_args.lr)

        scheduler = get_constant_schedule_with_warmup(optimizer,
                                                      num_warmup_steps=0.05)
    else:
        if model_args.model == 'cnn':
            model = CNN_MODEL(embeddings, model_args,
                              n_labels=model_args.labels).to(device)
        elif model_args.model == 'lstm':
            model = LSTM_MODEL(embeddings, model_args,
                               n_labels=model_args.labels).to(device)

        optimizer = AdamW(model.parameters(), lr=model_args.lr)
        scheduler = ReduceLROnPlateau(optimizer, verbose=True)

    return model, optimizer, scheduler