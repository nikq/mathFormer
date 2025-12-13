
import sys
import os
import torch
sys.path.append(os.getcwd())

from src.model import AutoRegressiveTransformerModel

def test_diagnostics():
    ntoken = 100
    ninp = 32
    nhead = 4
    nhid = 64
    nlayers = 2
    
    model = AutoRegressiveTransformerModel(ntoken, ninp, nhead, nhid, nlayers)
    
    dummy_input = torch.randint(0, ntoken, (1, 10))
    
    # Test normal forward
    logits = model(dummy_input)
    assert logits.shape == (1, 10, ntoken), f"Expected (1, 10, {ntoken}), got {logits.shape}"
    print("Normal forward pass passed.")
    
    # Test diagnostics forward
    logits, diagnostics = model(dummy_input, return_diagnostics=True)
    assert logits.shape == (1, 10, ntoken)
    
    assert 'embeddings' in diagnostics
    assert diagnostics['embeddings'].shape == (1, 10, ninp)
    
    assert 'activations' in diagnostics
    assert len(diagnostics['activations']) == nlayers
    for act in diagnostics['activations']:
        assert act.shape == (1, 10, ninp)
        
    assert 'attentions' in diagnostics
    assert len(diagnostics['attentions']) == nlayers
    for attn in diagnostics['attentions']:
        assert attn.shape == (1, 4, 10, 10) # B, num_heads, T, T
        
    print("Diagnostics forward pass passed.")

if __name__ == "__main__":
    test_diagnostics()
