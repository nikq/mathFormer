
model_type = "tiny"

# Model parameters
NTokens = 0 # will be set after building vocab
NInp = 64 # embedding dimension
NHead = 2 # number of attention heads
NHid = 64 # dimension of the feedforward network
NLayers = 2 # number of transformer layers
Dropout = 0.05 # dropout rate

if model_type == "tiny":
    NInp = 32 # embedding dimension
    NHead = 2 # number of attention heads
    NHid = 64 # dimension of the feedforward network
    NLayers = 2 # number of transformer layers
    Dropout = 0.05 # dropout rate
elif model_type == "small":
    NInp = 64 # embedding dimension
    NHead = 4 # number of attention heads
    NHid = 128 # dimension of the feedforward network
    NLayers = 4 # number of transformer layers
    Dropout = 0.05 # dropout rate
elif model_type == "medium":
    NInp = 1024 # embedding dimension
    NHead = 8 # number of attention heads
    NHid = 2048 # dimension of the feedforward network
    NLayers = 8 # number of transformer layers
    Dropout = 0.05 # dropout rate
else:
    raise ValueError(f"Unknown model type: {model_type}")

def modelhash():
    return hash((NTokens, NInp, NHead, NHid, NLayers, Dropout))