import types
import numpy as np


def test_adastep_adapter_smoke():
    # Direct smoke test for the AdaStep adapter shim (avoids heavy top-level imports)
    from predictors.adastep.adapter import AdaStepAdapter

    ada = AdaStepAdapter(checkpoint_path=None)
    assert ada is not None
    k = ada.predict_horizon({'top': np.zeros((224,224,3), dtype=np.uint8)})
    assert isinstance(k, int)
