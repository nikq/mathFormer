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
