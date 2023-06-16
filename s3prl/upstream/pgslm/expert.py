from collections import OrderedDict
from typing import Dict, List, Union
import sys
import os

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from ..interfaces import UpstreamBase
from ...downstream.emotion.dataset import IEMOCAPDataset

TEXTLESSLIB_PATH = os.getenv('TEXTLESSLIB_PATH','/home/fcurti/tesis/textlesslib/textless')

sys.path.append(os.path.dirname(TEXTLESSLIB_PATH))
from textless.data.speech_encoder import SpeechEncoder
from textless.data.f0_preprocess import SpeakerMeanNormalize, PromptNormalize
from fairseq import checkpoint_utils

DENSE_MODEL_NAME = "hubert-base-ls960"
QUANTIZER_NAME = "kmeans"
VOCAB_SIZE = 100

def collate_tensors(stream, pad, length):
    """
    >>> tensors = [torch.tensor(x) for x in [[1,2,3], [1]]]
    >>> pad = 0
    >>> collate_tensors(tensors, pad)
    tensor([[1, 2, 3],
        [1, 0, 0]])
    """
    assert len(stream) > 0

    n_samples = len(stream)

    collated = stream[0].new_full((n_samples, length), pad)

    for i, v in enumerate(stream):
        collated[i, : v.size(0)] = v

    return collated

def precalculate_encodings(encoder):
    train_dataset = IEMOCAPDataset('/mnt/c/Users/Felip/Downloads/IEMOCAP', './downstream/emotion/meta_data/Session1/test_meta_data.json')
    test_dataset = IEMOCAPDataset('/mnt/c/Users/Felip/Downloads/IEMOCAP', './downstream/emotion/meta_data/Session1/train_meta_data.json')
    encoded_wavs = {}
    print("Precalculando")
    for wav,_,filename in tqdm(train_dataset):
        encoded_audio = encoder(torch.FloatTensor(wav))
        encoded_audio['durations'] = encoded_audio['durations'].cuda().roll(1)
        encoded_audio['durations'][0] = 0
        encoded_audio['f0'] = encoded_audio['f0'].cuda().roll(1)
        encoded_audio['f0'][0] = 0
        encoded_wavs[filename] = encoded_audio
    for wav,_,filename in tqdm(test_dataset):
        encoded_audio = encoder(torch.FloatTensor(wav))
        encoded_audio['durations'] = encoded_audio['durations'].cuda().roll(1)
        encoded_audio['durations'][0] = 0
        encoded_audio['f0'] = encoded_audio['f0'].cuda().roll(1)
        encoded_audio['f0'][0] = 0
        encoded_wavs[filename] = encoded_audio
    torch.save(encoded_wavs, 'IEMOCAP_Encoded.pt')
        
        
class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt: str = None, model_config: str = None, **kwargs):
        """
        Args:
            ckpt:
                The checkpoint path for loading your pretrained weights.
                Can be assigned by the -k option in run_downstream.py

            model_config:
                The config path for constructing your model.
                Might not needed if you also save that in your checkpoint file.
                Can be assigned by the -g option in run_downstream.py
        """
        super().__init__()
        self.name = "[pgslm UpstreamExpert]"

        self.normalizer = PromptNormalize()
        
        self.encoder = SpeechEncoder.by_name(
            dense_model_name=DENSE_MODEL_NAME,
            quantizer_model_name=QUANTIZER_NAME,
            vocab_size=VOCAB_SIZE,
            need_f0=True,
            deduplicate=True,
            f0_normalizer=self.normalizer,
            f0_quantizer=None,
        )
        
        self.encoding_dict = torch.load('IEMOCAP_Encoded.pt')
        models, model_args, task = checkpoint_utils.load_model_ensemble_and_task(["./upstream/pgslm/checkpoints/continuous_prosody_shift_1_1.pt"])
        self.ulm = models[0]
        self.unit_pad = task.source_dictionary.pad()
        
        for i in range(12):
            self.add_hook(f"self.ulm.decoder.layers[{i}]", lambda input, output: output[0].transpose(0, 1))
        
    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs: List[Tensor], filenames=None) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        When the returning Dict contains the List with more than one Tensor,
        those Tensors should be in the same shape to train a weighted-sum on them.
        """

        # wavs = pad_sequence(wavs, batch_first=True)
        # wavs: (batch_size, max_len, 1)
        
        max_len = int(max([s.size()[0] for s in wavs]) / self.get_downsample_rates('a'))
        
        encoded_audios = []
        if filenames is None:
            for wav in wavs:
                encoded_audio = self.encoder(wav)
                encoded_audio['durations'] = encoded_audio['durations'].cuda().roll(1)
                encoded_audio['durations'][0] = 0
                encoded_audio['f0'] = encoded_audio['f0'].cuda().roll(1)
                encoded_audio['f0'][0] = 0
                encoded_audios.append(encoded_audio)
        else:
            for file in filenames:
                encoded_audios.append(self.encoding_dict[file]) 
        
        units = collate_tensors([s["units"] for s in encoded_audios], pad=self.unit_pad, length=max_len).cuda()
        f0 = collate_tensors(
            [s["f0"] for s in encoded_audios], pad=torch.zeros_like(encoded_audios[0]["f0"][0]), length=max_len
        ).cuda()
        durations = collate_tensors(
            [s["durations"] for s in encoded_audios],
            pad=torch.zeros_like(encoded_audios[0]["durations"][0]),
            length=max_len
        ).cuda()
        ulm_forward = self.ulm(units, durations, f0)


        # The "hidden_states" key will be used as default in many cases
        # Others keys in this example are presented for SUPERB Challenge
        #return {
        #    "hidden_states": ulm_forward['token'],
        #}
