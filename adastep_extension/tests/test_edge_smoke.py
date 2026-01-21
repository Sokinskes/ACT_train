import os
import torch
from integrations.export_to_torchscript import export


def test_torchscript_export(tmp_path):
    out = tmp_path / 'smoke.pt'
    export(str(out))
    m = torch.jit.load(str(out))
    x = torch.randn(1,3,64,64)
    y = m(x)
    assert y.shape[0] == 1
