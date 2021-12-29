from torch.utils.data import (
    DataLoader, Dataset
)
import logging
import tgt
import numpy as np
import os
import torch
import torchaudio
from tacotron.tokenizer import BaseTokenizer
from tacotron.utils import get_abspath

class TextMelProcessor:
    def __init__(self, cfg):
        super(TextMelProcessor, self).__init__()
        self.cfg = cfg

    def load_script(self, temp_path):
        raise NotImplementedError

    def get_dataset(self, split='train'):
        raise NotImplementedError


class TextMelDataset(Dataset):
    def __init__(self, cfg):
        super(TextMelDataset, self).__init__()
        self.cfg = cfg
        self.sampling_rate = cfg.sampling_rate
        self.filter_length = cfg.filter_length
        self.hop_length = cfg.hop_length
        self.win_length = cfg.win_length
        self.n_mel_channels = cfg.n_mel_channels
        self.mel_fmin = cfg.mel_fmin
        self.mel_fmax = cfg.mel_fmax
        self.train_script = cfg.train_script
        self.val_script = cfg.val_script
        self.load_mel_from_disk = cfg.load_mel_from_disk


    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def collater(self, batch):
        raise NotImplementedError

    def load_dataloader(self, shuffle: bool = True, batch_size: int = 2):
        return DataLoader(
            self, shuffle=shuffle, batch_size=batch_size, collate_fn=self.collater
        )


class TGTProcessor(TextMelProcessor):
    def __init__(self, cfg):
        super(TGTProcessor, self).__init__(cfg)


    def get_alignment(self, item):
        """
        This is from Korean FastSpeech2
        https://github.com/HGU-DLLAB/Korean-FastSpeech2-Pytorch/blob/ca9aae1b4931baa807ff884dc8d791c928ff57bc/utils.py#L19
        :param item:
        :return:
        """
        sil_phones = ['sil', 'sp', 'spn', '', '|']

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in item._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trimming leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s
            if p not in sil_phones:
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                p = ' '
                phones.append(p)
            durations.append(int(e * self.cfg.sampling_rate / self.cfg.hop_length) - int(s * self.cfg.sampling_rate / self.cfg.hop_length))

        # Trimming tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def build_tokenizer(self):
        wav_text_scripts = get_abspath(self.cfg.train_script)
        wav_paths, text_infos = self.load_script(wav_text_scripts)
        phones = list()
        for phone, duration, start, end in text_infos:
            phones.append(phone)
        tokenzier = BaseTokenizer.build_tokenizer(phones, self.cfg.normalize_option)
        return tokenzier

    def load_script(self, temp_path):

        with open(temp_path, 'r') as f:
            lines = f.readlines()

        wav_paths = list()
        text_infos = list()

        for line in lines:
            line = line.strip().split("|")
            wav_paths.append(get_abspath(line[0]))

            textgrid = tgt.io.read_textgrid(get_abspath(line[1]))
            phone, duration, start, end = self.get_alignment(textgrid.get_tier_by_name('phones'))
            text_infos.append([phone, duration, start, end])

        logging.info("load scripts from [{}]".format(temp_path))

        return wav_paths, text_infos

    def get_dataset(self, tokenizer, split='train'):
        if split=='train':
            wav_text_scripts = self.cfg.train_script
        else:
            wav_text_scripts = self.cfg.val_script
        wav_text_scripts = get_abspath(wav_text_scripts)
        wav_paths, text_infos = self.load_script(wav_text_scripts)

        logging.info("Convert raw lines to dataset")
        return TGTDataset(
            self.cfg, tokenizer, wav_paths, text_infos,
        )

class TGTDataset(TextMelDataset):
    def __init__(self, cfg, tokenizer, wav_paths, text_infos):
        super(TGTDataset, self).__init__(cfg)
        self.tokenizer = tokenizer
        self.wav_paths = wav_paths
        self.text_infos = text_infos

        self.pad_id = tokenizer.pad_id
        self.special_pad_id = tokenizer.special_pad_id

        ## most vocoder use 'slaney' for mel scale.
        self.mel_converter = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sampling_rate,
            n_fft=cfg.filter_length,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            f_min=cfg.mel_fmin,
            f_max=cfg.mel_fmax,
            n_mels=cfg.n_mel_channels,
            power=1,
            #normalized=True,
            norm='slaney',
            mel_scale='slaney',
        )

        self.C = 1
        self.clip_val = 1e-5

    def get_mel(self, filename, start_time=None, end_time=None):
        dir, name = os.path.split(filename)
        cache_name = 'cache_{}'.format(name).replace('.wav', '.npy')
        cache_filename = os.path.join(dir, cache_name)

        load_flag = True
        melspec = None

        if self.load_mel_from_disk and os.path.exists(cache_filename):
            melspec = torch.from_numpy(np.load(cache_filename))
            if melspec.size(0) == self.n_mel_channels:
                load_flag = False

        if load_flag or melspec == None:
            ## torchaudio normalize audio to the interval [-1, 1]
            audio, sampling_rate = torchaudio.load(filename)

            ## only use mono
            if audio.size(0) > 1:
                audio = audio[0, :].view(1, -1)

            ## crop
            if start_time is not None and end_time is not None:
                audio = audio[:,int(start_time*sampling_rate):int(end_time*sampling_rate)]

            ## sample rate check
            if sampling_rate != self.sampling_rate:
                audio = torchaudio.transforms.Resample(sampling_rate, self.sampling_rate)(audio)

                ## Sometimes resmapling make problem to its boundary[-1, 1]
                ## So, add normalize function to scale audio
                if audio.max() > 1 or audio.min() < -1:
                    def normalize(tensor):
                        # Subtract the mean, and scale to the interval [-1,1]
                        tensor_minusmean = tensor - tensor.mean()
                        return tensor_minusmean / tensor_minusmean.abs().max()

                    audio = normalize(audio)

            try:
                melspec = self.mel_converter(audio)
                melspec = torch.log(torch.clamp(melspec, min=self.clip_val) * self.C)
            except Exception:
                raise ValueError("ERROR : {}".format(filename))
            melspec = torch.squeeze(melspec, 0)

            if self.load_mel_from_disk:
                np.save(cache_filename, melspec.numpy())

        return melspec

    def __getitem__(self, idx):
        phones, durations, start_time, end_time = self.text_infos[idx]
        wav_path = self.wav_paths[idx]

        ## items
        input_ids = list()
        special_input_ids = list()
        for phone in phones:
            encoded_output = self.tokenizer.encode(phone)
            input_ids.extend(encoded_output['input_ids'])
            special_input_ids.extend(encoded_output['special_input_ids'])
            assert len(encoded_output['input_ids']) == 1 and len(encoded_output['special_input_ids']) == 1, 'this must be size of one [{}]'.format(self.wav_paths[idx])

        mel_specs = self.get_mel(wav_path, start_time, end_time)

        return {
            'input_ids':input_ids,
            'special_input_ids' : special_input_ids,
            'mel_specs': mel_specs,
            'durations': durations,
        }

    def __len__(self):
        return len(self.wav_paths)

    def collater(self, batch):
        input_ids = [b['input_ids'] for b in batch]
        special_input_ids = [b['special_input_ids'] for b in batch]
        mel_specs = [b['mel_specs'] for b in batch]
        durations = [b['durations'] for b in batch]

        ## output
        mel_lengths = [mel_spec.size(1) for mel_spec in mel_specs]
        target_size = max(mel_lengths)

        collated_mel_specs = torch.FloatTensor(len(batch), self.n_mel_channels, target_size)
        collated_mel_specs.zero_()
        collated_mel_lengths = list()
        collated_durations = list()

        for i, (mel_spec, mel_length, duration) in enumerate(zip(mel_specs, mel_lengths, durations)):
            diff = mel_length - target_size
            if diff > 0:
                collated_mel_specs[i, :, :target_size] = mel_spec[:, :target_size]
                mel_length = target_size

            else:
                collated_mel_specs[i, :, :mel_length] = mel_spec

            collated_mel_lengths.append(mel_length)
            diff = mel_length - sum(duration)

            ## Cover difference to the last index
            duration[-1] += diff
            assert duration[-1]>0, 'Errors in last index'
            collated_durations.append(duration)

        durations = collated_durations

        ## input
        input_lengths = [len(s) for s in input_ids]
        target_size = max(input_lengths)

        collated_ids = list()
        collated_special_ids = list()
        collated_input_lengths = list()
        collated_durations = list()

        for i, (input_id, special_input_id, input_length, duration) in enumerate(zip(input_ids, special_input_ids, input_lengths, durations)):
            diff = input_length - target_size
            if diff > 0:
                input_id = input_id[:target_size]
                special_input_id = special_input_id[:target_size]
                input_length = target_size
                duration = duration[:target_size]

            elif diff < 0:
                input_id = input_id + [self.pad_id] * -diff
                special_input_id = special_input_id + [self.special_pad_id] * -diff
                duration = duration + [0] * -diff

            collated_ids.append(input_id)
            collated_special_ids.append(special_input_id)
            collated_input_lengths.append(input_length)
            collated_durations.append(duration)

        return {
            'input_ids': torch.tensor(collated_ids, dtype=torch.long),
            'special_input_ids' : torch.tensor(collated_special_ids, dtype=torch.long),
            'input_lengths': np.array(collated_input_lengths),
            'mel_specs': collated_mel_specs,
            'mel_length': torch.tensor(collated_mel_lengths, dtype=torch.long),
            'durations': torch.tensor(collated_durations, dtype=torch.float),
        }


