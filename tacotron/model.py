import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import Tensor
from typing import List
from math import sqrt
import logging
import os
from tacotron.layers import ConvNorm, LinearNorm, PositionalEncoding, ACTIVATION_GROUP
import hydra
from torch.nn import functional as F
from tacotron.utils import get_abspath
import numpy as np
import math

class NonAttentiveTacotron(nn.Module):
    """
        Non-Attentive Tacotron module:
            - Encoder
            - Positional Encoding

            - Duration Predictor
            - Range Predictor
            - Gaussian Upsampling

            - Decoder
            - Postnet
    """

    model_save_name = 'pretrained_model.bin'

    def __init__(self, cfg):
        super(NonAttentiveTacotron, self).__init__()

        """
            Embedding with uniform init
            https://github.com/NVIDIA/tacotron2/blob/185cd24e046cc1304b4f8e564734d2498c6e2e6f/model.py#L464
        """
        assert cfg.symbols_embedding_dim > cfg.symbols_special_embedding_dim, 'must be bigger'
        self.embedding = nn.Embedding(
            cfg.num_labels, cfg.symbols_embedding_dim - cfg.symbols_special_embedding_dim)
        std = sqrt(2.0 / (cfg.num_labels + cfg.symbols_embedding_dim - cfg.symbols_special_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)

        self.special_embedding = nn.Embedding(
            cfg.num_special_labels, cfg.symbols_special_embedding_dim)
        std = sqrt(2.0 / (cfg.num_special_labels + cfg.symbols_special_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.special_embedding.weight.data.uniform_(-val, val)

        ## Encoder
        self.encoder = Encoder(cfg)

        ## Duration Predictor
        self.duration_predictor = DurationPredictor(cfg)

        ## Range Predictor
        self.range_predictor = RangePredictor(cfg)

        ## Gaussian Unsampling Module
        self.gaussian_upsampling = GaussianUpsampling(cfg)

        ## Positional Encoding
        self.positional_embedding = PositionalEncoding(cfg.positional_embedding_dim)

        ## decoder
        self.decoder = Decoder(cfg)

        ## postnet
        self.postnet = Postnet(cfg)

        self.cfg = cfg
        self.sampling_rate = cfg.sampling_rate

        ## loss params
        self.mse = nn.MSELoss(reduction='none')
        self.mae = nn.L1Loss(reduction='none')
        self.total_duration = cfg.total_duration
        self.duration_lambda = cfg.duration_lambda

        logging.info('build non-attentive-tacotron model')


    def get_positional_embedding(self, durations):
        """
            :param durations: [B, N]
            :return positional_embedding_outputs: [B, T, positional_hidden]
        """

        B = durations.size(0)
        N = durations.size(1)

        pos_durations = durations.long()
        sum_len = pos_durations.sum(dim=1)
        max_len = sum_len.max()
        diff_len = max_len - sum_len
        pos_durations[:, -1] = pos_durations[:, -1] + diff_len

        ids = torch.arange(max_len, device=durations.device).expand(B, N, max_len)
        pos_mask = (ids < pos_durations.view(B, N, 1))
        pos_ids = ids[pos_mask].view(-1, max_len)  # [B, T]
        positional_embedding_outputs = self.positional_embedding(pos_ids)

        return positional_embedding_outputs



    def forward(self, input_ids, special_input_ids,input_lengths, mel_specs, durations, **kwargs):
        """
            :param input_ids: token ids corresponding to input sentences [B, N]
            :param special_input_ids: special token ids corresponding to input sentences [B, N]
            :param input_lengths: length of sentences [B]
            :param mel_specs: log mel-spectrogram extracted from raw audio [B, T, n_mel_channels]
            :param durations: integer duration of phonemes corresponding to mel-spectrogram length [B, N]
            :param kwargs: extra
        """

        ## Encoder have 1D Conv layer
        basic_embedded_inputs = self.embedding(input_ids)
        special_embedded_inputs = self.special_embedding(special_input_ids)

        embedded_inputs = torch.cat([basic_embedded_inputs, special_embedded_inputs], dim=2)

        encoder_outputs = self.encoder(embedded_inputs, input_lengths)

        predicted_durations = self.duration_predictor(encoder_outputs, input_lengths)
        range_outputs = self.range_predictor(encoder_outputs, durations, input_lengths)

        encoder_upsampling_outputs = self.gaussian_upsampling(encoder_outputs, durations, range_outputs, input_lengths)

        positional_embedding_outputs = self.get_positional_embedding(durations)
        encoder_concated_outputs = torch.cat([encoder_upsampling_outputs, positional_embedding_outputs], dim=2)

        # [B, T, n_mel_channels]
        predicted_mel_specs = self.decoder(encoder_concated_outputs, durations, mel_specs)

        predicted_postnet_mel_specs = self.postnet(predicted_mel_specs) # [B, n_mel_channels, T]
        predicted_postnet_mel_specs = predicted_postnet_mel_specs + predicted_mel_specs

        return {
            'predicted_durations' : predicted_durations,
            'predicted_mel_specs' : predicted_mel_specs,
            'predicted_postnet_mel_specs' : predicted_postnet_mel_specs
        }

    def inference(self, input_ids, special_input_ids, input_lengths=None, pace_input_ids=None, **kwargs):
        """
            :param input_ids: token ids corresponding to input sentences [B, N]
            :param special_input_ids: special token ids corresponding to input sentences [B, N]
            :param input_lengths: length of sentences [B]
            :param pace_input_ids: speed rate corresponding to input sentences [B, N] : defalut = 1
            :param kwargs: extra
        """

        ## init to one
        if pace_input_ids is None:
            pace_input_ids = torch.ones_like(input_ids)

        ## Encoder have 1D Conv layer
        basic_embedded_inputs = self.embedding(input_ids)
        special_embedded_inputs = self.special_embedding(special_input_ids)

        embedded_inputs = torch.cat([basic_embedded_inputs, special_embedded_inputs], dim=2)

        encoder_outputs = self.encoder(embedded_inputs, input_lengths)

        predicted_durations = self.duration_predictor(encoder_outputs, input_lengths)
        pace_controlled_predicted_durations = pace_input_ids * predicted_durations

        ## rounding predicted values
        pace_controlled_predicted_durations = torch.round(pace_controlled_predicted_durations)

        ## fullfill zoro valus to one
        zeor_durations = pace_controlled_predicted_durations <= 0
        pace_controlled_predicted_durations[zeor_durations] = 1

        range_outputs = self.range_predictor(encoder_outputs, pace_controlled_predicted_durations, input_lengths)

        encoder_upsampling_outputs = self.gaussian_upsampling(encoder_outputs, pace_controlled_predicted_durations, range_outputs, input_lengths)
        positional_embedding_outputs = self.get_positional_embedding(pace_controlled_predicted_durations)
        encoder_concated_outputs = torch.cat([encoder_upsampling_outputs, positional_embedding_outputs], dim=2)

        # [B, T, n_mel_channels]
        predicted_mel_specs = self.decoder.inference(encoder_concated_outputs, pace_controlled_predicted_durations)

        predicted_postnet_mel_specs = self.postnet(predicted_mel_specs)  # [B, n_mel_channels, T]
        predicted_postnet_mel_specs = predicted_postnet_mel_specs + predicted_mel_specs

        return {
            'predicted_durations': predicted_durations,
            'pace_controlled_predicted_durations' : pace_controlled_predicted_durations,
            'predicted_mel_specs': predicted_mel_specs,
            'predicted_postnet_mel_specs': predicted_postnet_mel_specs
        }

    def get_loss(self,
                 predicted_durations, predicted_mel_specs, predicted_postnet_mel_specs, ## output
                 durations, mel_specs, mel_length, ## input
                 **kwargs):
        """
            total loss are sum of
            1. mse loss of pre-defined mel-spectrogram
            2. mae loss of pre-defined mel-spectrogram
            3. mse loss of post-processed(postnet) mel-spectrogram
            4. mae loss of post-processed(postnet) mel-spectrogram
            5. mse loss of duration in second
              - https://arxiv.org/abs/2010.04301
        """

        ## generate mel mask : [B, n_mel_channels, T]
        B = predicted_mel_specs.size(0)
        N = predicted_mel_specs.size(1) ## n_mel_channels
        max_len = mel_length.max()
        ids = torch.arange(max_len, device=mel_length.device).expand(B, N, max_len)
        mel_mask = (ids < mel_length.view(B, 1, 1)).transpose(1, 2).float()
        non_zero_elements = mel_mask.sum()

        ## 1. mel
        predicted_mse = self.mse(predicted_mel_specs.transpose(1, 2), mel_specs.transpose(1, 2))
        predicted_mse = (predicted_mse * mel_mask).sum() / non_zero_elements
        predicted_mse = torch.nan_to_num(predicted_mse)

        predicted_mae = self.mae(predicted_mel_specs.transpose(1, 2), mel_specs.transpose(1, 2))
        predicted_mae = (predicted_mae * mel_mask).sum() / non_zero_elements
        predicted_mae = torch.nan_to_num(predicted_mae)

        ## 2. postnet mel
        predicted_postnet_mse = self.mse(predicted_postnet_mel_specs.transpose(1, 2), mel_specs.transpose(1, 2))
        predicted_postnet_mse = (predicted_postnet_mse * mel_mask).sum() / non_zero_elements
        predicted_postnet_mse = torch.nan_to_num(predicted_postnet_mse)

        predicted_postnet_mae = self.mae(predicted_postnet_mel_specs.transpose(1, 2), mel_specs.transpose(1, 2))
        predicted_postnet_mae = (predicted_postnet_mae * mel_mask).sum() / non_zero_elements
        predicted_postnet_mae = torch.nan_to_num(predicted_postnet_mae)

        ## 3. duration
        if self.total_duration:
            duration_mask = (durations <= 0.0)
            predicted_durations = predicted_durations.masked_fill(duration_mask, 0.0)
            
            ## in_second
            in_seconds = (self.cfg.sampling_rate / self.cfg.hop_length)
            predicted_durations = predicted_durations / in_seconds
            durations = durations / in_seconds

            predicted_durations_mse = self.mse(predicted_durations.sum(dim=1), durations.sum(dim=1)).mean()
        else:
            duration_mask = (durations > 0.0).float()
            non_zero_duration = duration_mask.sum()

            in_seconds = (self.cfg.sampling_rate / self.cfg.hop_length)
            predicted_durations = predicted_durations / in_seconds
            durations = durations / in_seconds

            predicted_durations_mse = self.mse(predicted_durations, durations)
            predicted_durations_mse = (predicted_durations_mse * duration_mask).sum() / non_zero_duration

        predicted_durations_mse = torch.nan_to_num(predicted_durations_mse)

        loss = predicted_mse + predicted_mae + predicted_postnet_mse + predicted_postnet_mae + self.duration_lambda * predicted_durations_mse
        return { 
            'loss':loss,
            'predicted_mse':predicted_mse,
            'predicted_mae':predicted_mae,
            'predicted_postnet_mse':predicted_postnet_mse,
            'predicted_postnet_mae':predicted_postnet_mae,
            'predicted_durations_mse':predicted_durations_mse,
        }


    @classmethod
    def from_pretrained(cls, pretrained_path):
        pretrained_path = get_abspath(pretrained_path)
        logging.info('load files from [{}]'.format(pretrained_path))
        state = torch.load(os.path.join(pretrained_path, cls.model_save_name))
        cfg = state['cfg']
        model = cls(cfg)
        model.load_state_dict(state['model'])

        return model

    def save_pretrained(self, save_path):
        save_path = get_abspath(save_path)
        os.makedirs(save_path, exist_ok=True)
        state = {
            'cfg' : self.cfg,
            'model' : self.state_dict(),
        }
        torch.save(state, os.path.join(save_path, self.model_save_name))
        logging.info('save files to [{}]'.format(save_path))



class Encoder(nn.Module):
    """Encoder module:
        - Three (dropout, batch normalization, convolution)
        - single Bidirectional LSTM
    """
    def __init__(self, cfg):
        super(Encoder, self).__init__()


        assert cfg.encoder_activation in ACTIVATION_GROUP, 'activation must either one of them [{}]'.format(", ".join(ACTIVATION_GROUP.keys()))

        convolutions = []
        for _ in range(cfg.encoder_n_convolutions):

            conv_layer = nn.Sequential(
                ConvNorm(cfg.encoder_embedding_dim,
                         cfg.encoder_embedding_dim,
                         kernel_size=cfg.encoder_kernel_size, stride=1,
                         padding=int((cfg.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain=cfg.encoder_activation),
                nn.BatchNorm1d(cfg.encoder_embedding_dim, momentum=1-cfg.encoder_batch_norm_decay))
            convolutions.append(conv_layer)

        self.convolutions = nn.ModuleList(convolutions)
        self.dropout = nn.Dropout(p=cfg.encoder_dropout_p)

        self.lstm = nn.LSTM(input_size=cfg.encoder_embedding_dim,
                            hidden_size=int(cfg.encoder_lstm_dim/2), num_layers=1,
                            batch_first=True, bidirectional=True)

        self.activation = ACTIVATION_GROUP[cfg.encoder_activation]()
        #torch.nn.init.xavier_uniform_(self.lstm.weight)

    def forward(self, x, input_lengths=None):
        """
            :param x: [B, N]
            :param input_lengths: [B]
            :return: [B, N, encoder_lstm_dim]
        """

        x = x.transpose(1, 2)

        for conv in self.convolutions:
            x = self.dropout(self.activation(conv(x)))

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        if input_lengths is not None:
            x = pack_padded_sequence(
                x, input_lengths, batch_first=True, enforce_sorted=False)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        if input_lengths is not None:
            outputs, _ = pad_packed_sequence(
                outputs, batch_first=True)

        return outputs



class DurationPredictor(nn.Module):
    """Duration Predictor module:
        - two stack of BiLSTM
        - one projection layer
    """
    def __init__(self, cfg):
        super(DurationPredictor, self).__init__()
        assert cfg.duration_lstm_dim%2==0, "duration_lstm_dim must be even [{}]".format(cfg.duration_lstm_dim)

        self.lstm = nn.LSTM(cfg.encoder_lstm_dim,
                            int(cfg.duration_lstm_dim/2), 2,
                            batch_first=True, bidirectional=True)

        self.proj = LinearNorm(cfg.duration_lstm_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_outputs, input_lengths=None):
        """
            :param encoder_outputs: [B, N, encoder_lstm_dim]
            :param input_lengths: [B, N]
            :return: [B, N]
        """

        B = encoder_outputs.size(0)
        N = encoder_outputs.size(1)

        ## remove pad activations
        if input_lengths is not None:
            encoder_outputs = pack_padded_sequence(
                encoder_outputs, input_lengths, batch_first=True, enforce_sorted=False)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(encoder_outputs)

        if input_lengths is not None:
            outputs, _ = pad_packed_sequence(
                outputs, batch_first=True)

        outputs = self.relu(self.proj(outputs))

        return outputs.view(B, N)

class RangePredictor(nn.Module):
    """Duration Predictor module:
        - two stack of BiLSTM
        - one projection layer
    """
    def __init__(self, cfg):
        super(RangePredictor, self).__init__()
        assert cfg.range_lstm_dim%2==0, "range_lstm_dim must be even [{}]".format(cfg.range_lstm_dim)

        self.lstm = nn.LSTM(cfg.encoder_lstm_dim + 1,
                            int(cfg.range_lstm_dim/2), 2,
                            batch_first=True, bidirectional=True)

        self.proj = LinearNorm(cfg.range_lstm_dim, 1)

    def forward(self, encoder_outputs, durations, input_lengths=None):
        """
            :param encoder_outputs:
            :param durations:
            :param input_lengths:
            :return:
        """


        concated_inputs = torch.cat([encoder_outputs, durations.unsqueeze(-1)], dim=-1)

        ## remove pad activations
        if input_lengths is not None:
            concated_inputs = pack_padded_sequence(
                concated_inputs, input_lengths, batch_first=True, enforce_sorted=False)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(concated_inputs)

        if input_lengths is not None:
            outputs, _ = pad_packed_sequence(
                outputs, batch_first=True)

        outputs = self.proj(outputs)
        outputs = F.softplus(outputs)
        return outputs.squeeze()


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes, dropout_p=0.5, activation='relu'):
        super(Prenet, self).__init__()


        assert activation in ACTIVATION_GROUP, 'activation must either one of them [{}]'.format(
            ", ".join(ACTIVATION_GROUP.keys()))

        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

        self.dropout = nn.Dropout(p=dropout_p)
        self.activation = ACTIVATION_GROUP[activation]()

    def forward(self, x):
        for linear in self.layers:
            x = self.dropout(self.activation(linear(x)))
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, cfg):
        super(Postnet, self).__init__()

        assert cfg.postnet_activation in ACTIVATION_GROUP, 'activation must either one of them [{}]'.format(
            ", ".join(ACTIVATION_GROUP.keys()))

        self.activation = ACTIVATION_GROUP[cfg.postnet_activation]()
        self.dropout = nn.Dropout(p=cfg.postnet_dropout_p)

        self.convolutions = nn.ModuleList()
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(cfg.n_mel_channels, cfg.postnet_embedding_dim,
                         kernel_size=cfg.postnet_kernel_size, stride=1,
                         padding=int((cfg.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain=cfg.postnet_activation),
                nn.BatchNorm1d(cfg.postnet_embedding_dim))
        )

        for i in range(1, cfg.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(cfg.postnet_embedding_dim,
                             cfg.postnet_embedding_dim,
                             kernel_size=cfg.postnet_kernel_size, stride=1,
                             padding=int((cfg.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain=cfg.postnet_activation),
                    nn.BatchNorm1d(cfg.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(cfg.postnet_embedding_dim, cfg.n_mel_channels,
                         kernel_size=cfg.postnet_kernel_size, stride=1,
                         padding=int((cfg.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(cfg.n_mel_channels))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = self.dropout(self.activation(self.convolutions[i](x)))
        x = self.dropout(self.convolutions[-1](x))

        return x



class Decoder(nn.Module):
    """Decoder module:
        - Three (dropout, batch normalization, convolution)
        - single Bidirectional LSTM
    """

    def __init__(self, cfg):
        super(Decoder, self).__init__()
        ## decoder

        self.n_mel_channels = cfg.n_mel_channels
        self.decoder_lstm_dim = cfg.decoder_lstm_dim
        self.decoder_lstm_n = cfg.decoder_lstm_n

        self.prenet = Prenet(
            cfg.n_mel_channels,
            [cfg.prenet_dim, cfg.prenet_dim],
            dropout_p=cfg.prenet_dropout_p,
            activation=cfg.prenet_activation
        )

        self.decoder_lstm = nn.LSTM(
            input_size=cfg.prenet_dim + cfg.encoder_lstm_dim + cfg.positional_embedding_dim,
            hidden_size=cfg.decoder_lstm_dim,
            num_layers=cfg.decoder_lstm_n,
            batch_first=True,
            dropout=cfg.decoder_dropout_p,
        )

        self.linear_projection = LinearNorm(
            cfg.decoder_lstm_dim + cfg.encoder_lstm_dim + cfg.positional_embedding_dim,
            cfg.n_mel_channels)

    def forward(self, encoder_concated_outputs, durations, mel_specs):
        """ Prepares decoder inputs, i.e. mel outputs
            PARAMS
            ------
            encoder_concated_outputs: [B, T, encoder_lstm_dim + positional_embedding_dim]
            input_lengths : [B, N]
            mel_specs : [B, n_mel_channels, T]
            mel_length : [B, ]

            RETURNS
            -------
            predicted_mel_specs: processed decoder mel_specs

        """

        B = encoder_concated_outputs.size(0)

        # [B, n_mel_channels, T-1]
        guided_mel_specs = mel_specs[:, :, :-1]

        ## dummy mel_spec input must be zeors
        dummy_spec = torch.zeros(B, self.n_mel_channels, 1, device=guided_mel_specs.device)  # [B, n_mel_channels, 1]

        guided_mel_specs = torch.cat([dummy_spec, guided_mel_specs], dim=2)  # [B, n_mel_channels, T]

        # [B, n_mel_channels, T] -> [B, T, n_mel_channels]
        guided_mel_specs = guided_mel_specs.transpose(1, 2)

        prenet_outputs = self.prenet(guided_mel_specs) # [B, T, prenet_dim]

        concated_prenet_outputs = torch.cat([prenet_outputs, encoder_concated_outputs], dim=2)

        ## remove pad activations

        concated_prenet_outputs = pack_padded_sequence(
            concated_prenet_outputs, durations.long().sum(dim=1).detach().cpu(), batch_first=True, enforce_sorted=False)

        self.decoder_lstm.flatten_parameters()
        decoder_lstm_outputs, _ = self.decoder_lstm(concated_prenet_outputs) # [B, T, decoder_lstm_dim]

        decoder_lstm_outputs, _ = pad_packed_sequence(
            decoder_lstm_outputs, batch_first=True)

        concated_decoder_lstm_outputs = torch.cat([decoder_lstm_outputs, encoder_concated_outputs], dim=2) # [B, T, decoder_lstm_dim + encoder_lstm_dim + positional_embedding_dim]

        predicted_mel_specs = self.linear_projection(concated_decoder_lstm_outputs) # [B, T, n_mel_channels]
        predicted_mel_specs = predicted_mel_specs.transpose(1, 2) # [B, n_mel_channels, T]

        return predicted_mel_specs

    def inference(self, encoder_concated_outputs, durations):
        """ Prepares decoder inputs, i.e. mel outputs
            PARAMS
            ------
            encoder_concated_outputs: [B, T, encoder_lstm_dim + positional_embedding_dim]
            input_lengths : [B, N]
            mel_specs : [B, n_mel_channels, T]
            mel_length : [B, ]

            RETURNS
            -------
            predicted_mel_specs: processed decoder mel_specs

        """

        B = encoder_concated_outputs.size(0)

        ## total mel length to generate
        total_length = durations.long().sum(dim=1).max()

        ## initialization
        ## 1. first mel_spec input must be zeors [B, 1, n_mel_channels]
        predicted_mel_spec = torch.zeros(B, 1, self.n_mel_channels, device=encoder_concated_outputs.device)
        ## 2. decoder_lstm_hidden, decoder_lstm_cell
        decoder_lstm_hidden = torch.zeros(self.decoder_lstm_n, B, self.decoder_lstm_dim, device=encoder_concated_outputs.device)
        decoder_lstm_cell = torch.zeros(self.decoder_lstm_n, B, self.decoder_lstm_dim, device=encoder_concated_outputs.device)

        predicted_mel_specs = []
        for idx in range(total_length):
            prenet_output = self.prenet(predicted_mel_spec)  # [B, 1, prenet_dim]
            encoder_concated_output = encoder_concated_outputs[:, idx, :].unsqueeze(1)
            concated_prenet_output = torch.cat([prenet_output, encoder_concated_output], dim=2) # [B, 1, decoder_lstm_dim + encoder_lstm_dim + positional_embedding_dim]
            self.decoder_lstm.flatten_parameters()

            # decoder_lstm_output: [B, 1, decoder_lstm_dim]
            decoder_lstm_output, (decoder_lstm_hidden, decoder_lstm_cell) = self.decoder_lstm(concated_prenet_output, (decoder_lstm_hidden, decoder_lstm_cell))

            # concated_decoder_lstm_output : [B, 1, decoder_lstm_dim + encoder_lstm_dim + positional_embedding_dim]
            concated_decoder_lstm_output = torch.cat([decoder_lstm_output, encoder_concated_output], dim=2)

            predicted_mel_spec = self.linear_projection(concated_decoder_lstm_output)  # [B, 1, n_mel_channels]
            predicted_mel_specs.append(predicted_mel_spec)

        # predicted_mel_specs : [B, T, n_mel_channels]
        predicted_mel_specs = torch.cat(predicted_mel_specs, dim=1)
        predicted_mel_specs = predicted_mel_specs.transpose(1, 2)  # [B, n_mel_channels, T]

        return predicted_mel_specs




class GaussianUpsampling(nn.Module):
    """
        Non-attention Tacotron:
            - https://arxiv.org/abs/2010.04301
        this source code is implemenation of the ExpressiveTacotron from BridgetteSong
            - https://github.com/BridgetteSong/ExpressiveTacotron/blob/master/model_duration.py
    """
    def __init__(self, hparams):
        super(GaussianUpsampling, self).__init__()
        self.mask_score = -1e15

    def forward(self, encoder_outputs, durations, vars, input_lengths=None):
        """ Gaussian upsampling
        PARAMS
        ------
        encoder_outputs: Encoder outputs  [B, N, H]
        durations: phoneme durations  [B, N]
        vars : phoneme attended ranges [B, N]
        input_lengths : [B]
        RETURNS
        -------
        encoder_upsampling_outputs: upsampled encoder_output  [B, T, H]
        """
        B = encoder_outputs.size(0)
        N = encoder_outputs.size(1)
        T = int(torch.sum(durations, dim=1).max().item())
        c = torch.cumsum(durations, dim=1).float() - 0.5 * durations
        c = c.unsqueeze(2) # [B, N, 1]
        t = torch.arange(T, device=encoder_outputs.device).expand(B, N, T).float()  # [B, N, T]
        vars = vars.view(B, -1, 1) # [B, N, 1]


        w_t = -0.5 * (np.log(2.0 * np.pi) + torch.log(vars) + torch.pow(t - c, 2) / vars) # [B, N, T]

        if input_lengths is not None:
            input_masks = ~self.get_mask_from_lengths(input_lengths, N) # [B, N]
            input_masks = torch.tensor(input_masks, dtype=torch.bool, device=w_t.device)
            masks = input_masks.unsqueeze(2)
            w_t.data.masked_fill_(masks, self.mask_score)
        w_t = F.softmax(w_t, dim=1)

        encoder_upsampling_outputs = torch.bmm(w_t.transpose(1, 2), encoder_outputs)  # [B, T, encoder_hidden_size]

        return encoder_upsampling_outputs


    def get_mask_from_lengths(self, lengths, max_len=None):
        if max_len is None:
            max_len = max(lengths)
        ids = np.arange(0, max_len)
        mask = (ids < lengths.reshape(-1, 1))
        return mask
