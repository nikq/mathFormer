
import sys
import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# Add project root to path
sys.path.append(os.getcwd())

from src.model import AutoRegressiveTransformerModel
from src.modelparam import ModelParam
from src.prepare_data import build_vocab
from src.checkpoint_utils import load_checkpoint_payload

app = FastAPI()

# Global variables to hold model and vocab
model = None
vocab = None
idx2char = {}
device = torch.device("cpu")

class InferenceRequest(BaseModel):
    text: str

class LoadCheckpointRequest(BaseModel):
    filename: str

class DiagnosticResponse(BaseModel):
    tokens: List[str]
    embeddings: List[List[float]] # (T, dim) - using list for JSON
    attentions: List[List[List[List[float]]]] # (Layers, Heads, T, T)
    activations: List[List[List[float]]] # (Layers, T, dim)

def load_model(checkpoint_name: Optional[str] = None):
    global model, vocab, idx2char, device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if vocab is None:
        vocab = build_vocab()
        idx2char = {v: k for k, v in vocab.items()}
    ntokens = len(vocab)
    
    checkpoint_path = None
    if checkpoint_name:
         checkpoint_path = os.path.join("checkpoints", checkpoint_name)
         if not os.path.exists(checkpoint_path):
             print(f"Checkpoint not found: {checkpoint_path}")
             checkpoint_path = None
    
    if not checkpoint_path:
        # Try to find the latest checkpoint
        checkpoint_dir = "checkpoints"
        checkpoints = []
        if os.path.exists(checkpoint_dir):
            checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
            checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        if checkpoints:
            checkpoint_path = checkpoints[0]
            
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        state_dict, config = load_checkpoint_payload(checkpoint_path, map_location=device)
        if config is None:
            print("Warning: No config found in checkpoint. Using default parameters.")
            config = {}
        
        # Try to infer model type from filename
        model_type = 'small' # default
        import re
        match = re.search(r'_([a-z]+)_step', os.path.basename(checkpoint_path))
        if match:
            model_type = match.group(1)
            print(f"Inferred model type from filename: {model_type}")
        
        # Get defaults from ModelParam
        try:
            defaults = ModelParam(model_type, ntokens)
            default_ninp, default_nhead, default_nhid, default_nlayers, default_dropout = \
                defaults.NInp, defaults.NHead, defaults.NHid, defaults.NLayers, defaults.Dropout
        except:
             print(f"Could not get defaults for {model_type}, using hardcoded fallbacks.")
             default_ninp = 256
             default_nhead = 8
             default_nhid = 1024
             default_nlayers = 4
             default_dropout = 0.1
        
        # Build model param from config or defaults
        ninp = config.get('ninp', default_ninp)
        nhead = config.get('nhead', default_nhead)
        nhid = config.get('nhid', default_nhid)
        nlayers = config.get('nlayers', default_nlayers)
        dropout = config.get('dropout', default_dropout)
        dropout = config.get('dropout', 0.1)
        
        model = AutoRegressiveTransformerModel(ntokens, ninp, nhead, nhid, nlayers, dropout=dropout).to(device)
        model.load_state_dict(state_dict)
    else:
        print("No checkpoint found. Using initialized random model.")
        model = AutoRegressiveTransformerModel(ntokens, 128, 4, 256, 2).to(device)
    
    model.eval()

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/checkpoints")
async def get_checkpoints():
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        return []
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    return checkpoints

@app.post("/load_checkpoint")
async def load_checkpoint_endpoint(request: LoadCheckpointRequest):
    try:
        load_model(request.filename)
        return {"status": "success", "message": f"Loaded {request.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze(request: InferenceRequest) -> Dict[str, Any]:
    global model, vocab, idx2char, device
    
    if not request.text:
       raise HTTPException(status_code=400, detail="Empty text")
       
    # Tokenize (simple char level for now based on what I see in prepare_data, 
    # but actually prepare_data uses a vocab dict. Let's assume input is valid chars)
    # Actually, let's look at how text is converted. 
    # For now, I'll filter chars that are in vocab.

    tokens = [vocab['<sos>']] + [vocab['<big>']] + [vocab[c] for c in request.text if c in vocab] + [vocab['<scratchpad>']] # Assume big endian for eval
    if not tokens:
         raise HTTPException(status_code=400, detail="No valid tokens found")
         
    input_tensor = torch.tensor([tokens], dtype=torch.long).to(device) # (1, T)
    prompt_len = input_tensor.size(1)
    
    with torch.no_grad():
        # Generate prediction
        # Use a reasonable max length, e.g. 50 new tokens
        generated = model.generate(input_tensor, max_new_tokens=50, eos_token=vocab['<eos>']) # (1, T_total)
        
        # Now run forward pass on full generated sequence to get diagnostics
        # generate returns squeezed tensor (T,), so unsqueeze to (1, T)
        if generated.dim() == 1:
            generated = generated.unsqueeze(0)
        logits, diagnostics = model(generated, return_diagnostics=True)
        
    # Process results for JSON
    
    # Convert tokens back to chars for display
    # generated is (1, T_total)
    full_tokens = generated[0].cpu().tolist()
    token_chars = [idx2char[t] for t in full_tokens]
    
    # Embeddings: (1, T, dim) -> (T, dim)
    embeddings = diagnostics['embeddings'].squeeze(0).cpu().tolist()
    
    # Attentions: list of (1, H, T, T) -> list of (H, T, T)
    attentions = [attn.squeeze(0).cpu().tolist() for attn in diagnostics['attentions']]
    
    # Activations: list of (1, T, dim) -> list of (T, dim)
    activations = [act.squeeze(0).cpu().tolist() for act in diagnostics['activations']]
    
    return {
        "tokens": token_chars,
        "embeddings": embeddings,
        "attentions": attentions,
        "activations": activations,
        "prompt_length": prompt_len
    }

# Also serve static files
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app.mount("/static", StaticFiles(directory="viewer"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("viewer/index.html")
