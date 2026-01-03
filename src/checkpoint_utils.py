import torch

def extract_checkpoint_components(raw):
    """Return (state_dict, config_dict) from a loaded checkpoint payload."""
    state_dict = raw
    config = None
    if isinstance(raw, dict):
        maybe_state = raw.get('state_dict')
        if isinstance(maybe_state, dict):
            state_dict = maybe_state
            config = raw.get('config')
        else:
            state_dict = raw
            config = raw.get('config') if isinstance(raw.get('config'), dict) else None
    return state_dict, config


def load_checkpoint_payload(path, map_location=None):
    """Load a checkpoint file and split it into state_dict and config."""
    raw = torch.load(path, map_location=map_location)
    return extract_checkpoint_components(raw)

from src.modelparam import ModelParam
from src.model import AutoRegressiveTransformerModel

def build_model_param(modelsize, ntokens, checkpoint_config):
    base_param = ModelParam(modelsize, ntokens)
    if not checkpoint_config:
        return base_param
    ckpt_ntokens = checkpoint_config.get('ntoken')
    if ckpt_ntokens is not None and ckpt_ntokens != ntokens:
        raise ValueError(
            f"Checkpoint expects {ckpt_ntokens} tokens but current vocab has {ntokens}."
        )
    base_param.setParam(
        checkpoint_config.get('model_type', base_param.type()),
        ntokens,
        checkpoint_config.get('ninp', base_param.NInp),
        checkpoint_config.get('nhead', base_param.NHead),
        checkpoint_config.get('nhid', base_param.NHid),
        checkpoint_config.get('nlayers', base_param.NLayers),
        checkpoint_config.get('dropout', base_param.Dropout),
        checkpoint_config.get('num_experts', base_param.NumExperts),
        checkpoint_config.get('active_experts', base_param.ActiveExperts)
    )
    return base_param

def infer_model_hparams(state_dict):
    """Infer ntoken, ninp, nhid, nlayers from an autoregressive checkpoint state_dict."""
    emb_key = 'tok_emb.weight'
    if emb_key not in state_dict:
        # Provide a fallback or raise error? raising error seems safer as strictly typed
        # But wait, maybe the key is different? Let's check typical keys.
        # Assuming standard mathFormer keys.
        pass
        
    if emb_key in state_dict:
        ntoken, ninp = state_dict[emb_key].shape
    else:
        # Trying to find embedding weight
        for k in state_dict.keys():
            if 'emb.weight' in k:
                ntoken, ninp = state_dict[k].shape
                break
        else:
             raise ValueError(f"Embedding key '{emb_key}' not found in checkpoint.")

    block_prefix = 'blocks.'
    block_ids = set()
    linear1_key = None
    for key in state_dict.keys():
        if not key.startswith(block_prefix):
            continue
        parts = key.split('.')
        if len(parts) < 3:
            continue
        block_idx = parts[1]
        if block_idx.isdigit():
            block_ids.add(block_idx)
        if linear1_key is None and parts[-2:] == ['linear1', 'weight']:
            linear1_key = key

    # Defaults if we can't completely infer (though we should be able to)
    default_param = ModelParam('small', ntoken)
    nhid_infer = state_dict[linear1_key].shape[0] if linear1_key else default_param.NHid
    nlayers_infer = len(block_ids) or default_param.NLayers
    return ntoken, ninp, nhid_infer, nlayers_infer

def load_model_from_checkpoint(model_path, device, model_size='small'):
    state_dict, config = load_checkpoint_payload(model_path, map_location=device)
    ntoken, ninp_ckpt, nhid_ckpt, nlayers_ckpt = infer_model_hparams(state_dict)
    
    # ModelParam でモデルパラメータを取得
    model_param = ModelParam(model_size, ntoken)
    nhead = model_param.NHead
    dropout = model_param.Dropout

    if config:
        ninp_ckpt = config.get('ninp', ninp_ckpt)
        nhid_ckpt = config.get('nhid', nhid_ckpt)
        nlayers_ckpt = config.get('nlayers', nlayers_ckpt)
        nhead = config.get('nhead', nhead)
        dropout = config.get('dropout', dropout)
    
    model = AutoRegressiveTransformerModel(
        ntoken,
        ninp_ckpt,
        nhead,
        nhid_ckpt,
        nlayers_ckpt,
        dropout,
        max_len=2048, # Assuming default max_len
        num_experts=model_param.NumExperts,
        active_experts=model_param.ActiveExperts
    ).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[WARN] Missing keys: {missing}, Unexpected keys: {unexpected}")
    model.eval()
    return model

