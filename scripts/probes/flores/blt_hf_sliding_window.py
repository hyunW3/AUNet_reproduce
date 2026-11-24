"""Restore BLT's 512-byte sliding-window entropy patching in the transformers conversion.

`itazap/blt-1b-hf` builds the entropy patcher's attention with a plain causal mask, while BLT
specifies `attn_bias_type="local_block_causal"` with `sliding_window=512`. The consequence is not
subtle: on the DCLM held-out windows the shipped conversion patches at 1.151 bytes/patch where
native bytelatent patches at 4.071, and every downstream conclusion about BLT's compression moves
with it (see reports_afterAAAIsub/blt_threshold_ablation.md).

`enable(window=512)` swaps only the PATCHER's mask -- the local/global trunks keep plain causal --
and is validated against the native implementation on the 320 DCLM windows:

    released theta 1.335442  ->  4.0716 B/patch   (native 4.0710)
    theta fitted to 4.53     ->  1.468750         (native 1.46875 -> 4.5120, here 4.5101)
    entropy p50/p75/p90/p99  ->  0.4062 / 1.3047 / 2.4062 / 3.2344   (native: identical)

This matters because the native checkout (/mnt/ssd2/hyun2/lzspacebyte) was deleted along with its
consolidated weights and venv; the conversion plus this shim is now the only way to run BLT at its
intended patching.

    import blt_hf_sliding_window as sw; sw.enable()      # before loading the model
"""
import transformers.models.blt.modeling_blt as _MB
from transformers.masking_utils import create_sliding_window_causal_mask
from transformers.models.blt.configuration_blt import BltPatcherConfig

_ORIGINAL = _MB.create_causal_mask
_ENABLED = False


def enable(window: int = 512) -> None:
    global _ENABLED
    if _ENABLED:
        return

    def _dispatch(config=None, **kw):
        # BltPatcher passes its own BltPatcherConfig; the trunks pass BltLocal/GlobalConfig.
        if isinstance(config, BltPatcherConfig):
            config.sliding_window = window
            return create_sliding_window_causal_mask(config=config, **kw)
        return _ORIGINAL(config=config, **kw)

    _MB.create_causal_mask = _dispatch
    _ENABLED = True
    print(f"[blt] entropy patcher -> sliding window {window} (upstream local_block_causal)")


def disable() -> None:
    global _ENABLED
    _MB.create_causal_mask = _ORIGINAL
    _ENABLED = False
