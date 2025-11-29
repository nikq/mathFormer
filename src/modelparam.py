
# Model parameters
NTokens = 0 # will be set after building vocab
NInp = 256 # embedding dimension
NHead = 8 # number of attention heads
NHid = 2048 # dimension of the feedforward network
NLayers = 8 # number of transformer layers
Dropout = 0.05 # dropout rate

def modelhash():
    return hash((NTokens, NInp, NHead, NHid, NLayers, Dropout))