import hydra
import os
import torch
from tacotron.utils import reset_logging, set_seed, get_abspath
from tacotron.configs import ForcedAlignerConfig
from hydra.core.config_store import ConfigStore
import logging
from sklearn.model_selection import train_test_split
import tgt
from pathlib import Path
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import unicodedata
from dataclasses import _MISSING_TYPE, dataclass
import torchaudio
import string
import re

def init():
    cs = ConfigStore.instance()

    ## base
    cs.store(group="base", name='forced_aligner', node=ForcedAlignerConfig)


@hydra.main(config_path=os.path.join(".", "configs"), config_name="preprocess")
def main(cfg):

    print(cfg)

    ## initialize remove token
    special_regex = re.compile('[%s]' % re.escape(string.punctuation))
    remove_regex = re.compile('[%s]' % re.escape('"#$%&()*+-/:;<=>@[\\]^_{}”“'))

    ## this is not supported in Wav2vec2 vocabs
    unexpected_letters = re.compile('[%s]' % re.escape("àâäèéêëîïôöœùûüÿçÀÂÄÈÉÊËÎÏÔÖŒÙÛÜŸÇß"))

    selected_token = {
        '’' : "'",
        "," : ","
    }

    ## Resent Logging
    reset_logging()

    args = cfg.base

    ## GPU setting
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device

    ############# INIT #################
    ## set meta and audio data path of LJSpeech-1.1
    audio_path = get_abspath(args.audio_path)
    script_path = get_abspath(args.script_path)

    assert os.path.exists(audio_path), 'There is no file in this audio_path [{}]'.format(audio_path)
    assert os.path.exists(script_path), 'There is no file in this script_path [{}]'.format(script_path)

    ## the path for grid
    save_grid_path = get_abspath(args.save_grid_path)
    os.makedirs(save_grid_path, exist_ok=True)

    ## the path for splited metadata
    save_script_path = get_abspath(args.save_script_path)
    os.makedirs(save_script_path, exist_ok=True)

    ## normalize options
    NORMALIZE_OPTIONS = ['NFC', 'NFKC', 'NFD', 'NFKD']
    assert args.normalize_option in NORMALIZE_OPTIONS, 'normalize option should either {}. ex) Korean for "NFKD", English for "NFC"'.format(
        ', '.join(NORMALIZE_OPTIONS))

    ## we only support facebook wav2vec or personal one.
    pretrained_model = args.pretrained_model
    if 'facebook' not in pretrained_model:
        pretrained_model = get_abspath(pretrained_model)
        assert os.path.exists(pretrained_model), 'There is no pretrained_model [{}]'.format(pretrained_model)

    ## load model
    processor = Wav2Vec2Processor.from_pretrained(pretrained_model)
    model = Wav2Vec2ForCTC.from_pretrained(pretrained_model)
    model.to(device)

    logging.info("loaded pre-trained wav2vec2 model from [{}]".format(pretrained_model))

    ## get dict
    dictionary = processor.tokenizer.get_vocab()
    labels = [item for item, idx in dictionary.items()]

    ## get sampling_rate
    w2v_sample_rate = processor.feature_extractor.sampling_rate

    with open(script_path, 'r') as f:
        scripts = f.readlines()

    logging.info("[Start] make gridtext information")

    ## 1. building grid
    percent = 100/len(scripts) if len(scripts) > 0 else 1
    new_scripts = list()
    removed_count = 0
    for script_idx, script in enumerate(scripts):
        if script_idx % 50 == 0:
            logging.info("{:.4}% progressed   [{}] passed, [{}] removed".format(percent*script_idx, len(new_scripts), removed_count))

        script = script.strip()
        items = script.split('|')
        temp_path = os.path.join(audio_path, items[0])
        if Path(temp_path).suffix == '':
            temp_path = '{}.wav'.format(temp_path)

        file_name = "{}.TextGrid".format(Path(temp_path).stem)

        ## should fix it for your own dataset
        transcript = items[2]
        transcript = remove_regex.sub('', transcript)

        for before, after in selected_token.items():
            transcript.replace(before, after)

        ## replace space token for wav2vec2
        transcript = transcript.replace(' ', '|')

        ## I put space at the end of sentence to make wav2vec2 extract timeline stable.
        transcript = "|" + transcript + "|"

        ## wav2vec2 only support upper string for English
        transcript = transcript.upper()

        ## normalize sentences for Korean
        transcript = unicodedata.normalize(args.normalize_option, transcript)

        ## get probs from wav2vec2
        with torch.inference_mode():
            waveform, sample_rate = torchaudio.load(temp_path)

            ## only use mono
            if waveform.size(0) > 1:
                waveform = waveform[0, :].view(1, -1)

            if sample_rate != w2v_sample_rate:
                waveform = torchaudio.transforms.Resample(sample_rate, w2v_sample_rate)(waveform)
                sample_rate = w2v_sample_rate

            emissions = model(waveform.to(device))
            emissions = torch.log_softmax(emissions.logits, dim=-1)

        emission = emissions[0].cpu().detach()

        ratio = waveform.size(1) / emission.size(0)
        text_grid = tgt.core.TextGrid(file_name)
        tier = tgt.core.IntervalTier(name='phones')

        tokens = list()
        converted_items = list()

        for c in list(transcript):
            if c not in dictionary and len(tokens)>0:
                converted_items.append([len(tokens) - 1, c])
            else:
                tokens.append(dictionary[c])

        trellis = get_trellis(emission, tokens)
        back_path = backtrack(trellis, emission, tokens)
        segments = merge_repeats(back_path, [labels[token] for token in tokens])

        ## we remove grapheme longer than 2
        for (idx, c) in converted_items:
            candidate = segments[idx].label + c
            segments[idx].label = candidate[:2]


        flag = True
        for segment in segments[1:-1]:
            x0 = ratio * segment.start / w2v_sample_rate
            x1 = ratio * segment.end / w2v_sample_rate
            word = segment.label

            count = 0
            for p in list(word.replace('|', ' ')):
                if special_regex.search(p):
                    count += 1

            if count > 1:
                flag = False

            if unexpected_letters.search(word):
                flag = False

            interval = tgt.core.Interval(x0, x1, word)
            tier.add_interval(interval)

        text_grid.add_tier(tier)
        grid_path = os.path.join(save_grid_path, file_name)
        tgt.io.write_to_file(text_grid, grid_path, format='long')

        if flag:
            new_scripts.append("|".join([temp_path, grid_path]))
        else:
            logging.info(transcript)
            removed_count+=1
    logging.info("[End] make gridtext information, [{}] passed, [{}] removed".format(len(new_scripts), removed_count))


    logging.info("[Start] make splited script data")
    ## 2. split data
    train_scripts, val_scripts = train_test_split(new_scripts, test_size=args.test_size)

    with open(os.path.join(save_script_path, "train.txt"), 'w') as write_f:
        for train_item in train_scripts:
            print(train_item, file=write_f)

    with open(os.path.join(save_script_path, "dev.txt"), 'w') as write_f:
        for dev_item in val_scripts:
            print(dev_item, file=write_f)

    logging.info("[End] make splited script data")



@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.full((num_frame + 1, num_tokens + 1), -float('inf'))
    trellis[:, 0] = 0
    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


def backtrack(trellis, emission, tokens, blank_id=0):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When refering to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when refering to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise ValueError('Failed to align')
    return path[::-1]


def merge_repeats(path, temp_tokens):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(temp_tokens[path[i1].token_index], path[i1].time_index, path[i2 - 1].time_index + 1, score))
        i1 = i2
    return segments



if __name__ == "__main__":
    init()
    main()

