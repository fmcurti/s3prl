from .expert import UpstreamExpert as _UpstreamExpert
import urllib.request
import torch


def pgslm(*args, **kwargs):
    """
    To enable your customized pretrained model, you only need to implement
    upstream/example/expert.py and leave this file as is. This file is
    used to register the UpstreamExpert in upstream/example/expert.py
    The following is a brief introduction of the registration mechanism.

    The s3prl/hub.py will collect all the entries registered in this file
    (callable variables without the underscore prefix) as a centralized
    upstream factory. One can pick up this upstream from the factory via

    1.
    from s3prl.hub import customized_upstream
    model = customized_upstream(ckpt, model_config)

    2.
    model = torch.hub.load(
        'your_s3prl_path',
        'customized_upstream',
        ckpt,
        model_config,
        source='local',
    )

    Our run_downstream.py and downstream/runner.py follows the first usage
    """
    return _UpstreamExpert(*args, **kwargs)

def pgslm_download_ckpt(*args, **kwargs):
    CKPT_URL = "https://dl.fbaipublicfiles.com/textless_nlp/pgslm/ulm_checkpoints/continuous_prosody_shift_1_1.pt"

    urllib.request.urlretrieve(CKPT_URL, './upstream/pgslm/checkpoints/continuous_prosody_shift_1_1.pt')
    checkpoint = torch.load("./upstream/pgslm/checkpoints/continuous_prosody_shift_1_1.pt")
    checkpoint['cfg']['task']['data'] = './upstream/pgslm/checkpoints/data_config.json'
    torch.save(checkpoint, './upstream/pgslm/checkpoints/continuous_prosody_shift_1_1.pt')
    return _UpstreamExpert(*args, **kwargs)
