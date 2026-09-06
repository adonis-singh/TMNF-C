import pytest
import torch
from tmnf_rl.encoder import encode_flat, feature_names, EXCLUDED_COLUMNS, FlatEncoder
from tmnf_rl.agents.ppo import Agent


def observation(n=3, device='cpu'):
    x=torch.zeros(n,237,device=device)
    x[:,6]=1
    x[:,81+15]=1
    x[:,81:84]=torch.tensor([20.,3.,-10.],device=device)
    x[:,233:]=torch.tensor([2.,0.,1.,0.],device=device)
    return x


def test_gate_encoding_masks_padding_and_excludes_coordinates():
    x=observation(); y=encode_flat(x,3)
    assert y.shape==(3,405) and len(feature_names(3))==405
    altered=x.clone(); altered[:,list(EXCLUDED_COLUMNS)]=9999
    altered[:,101:116]=float('nan')  # geometry in invalid second gate
    torch.testing.assert_close(y,encode_flat(altered,3),rtol=0,atol=0)
    altered=x.clone();altered[:,81]+=50
    assert not torch.equal(y,encode_flat(altered,3))
    with pytest.raises(ValueError):encode_flat(x,2)
    with pytest.raises(ValueError):encode_flat(x[:,:81],3)


@pytest.mark.parametrize('arch',['mlp','transformer_s'])
def test_gate_network_backward_and_load(arch,monkeypatch):
    monkeypatch.setattr(torch,'compile',lambda f:f)
    agent=Agent(arch,64,encoder_version=3,cuda_graphs=False)
    x=observation();h=agent.hidden(x);h.square().mean().backward()
    assert torch.isfinite(h).all()
    clone=Agent(arch,64,encoder_version=3,cuda_graphs=False)
    clone.load_state_dict(agent.state_dict())
    torch.testing.assert_close(h,clone.hidden(x),rtol=0,atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires CUDA')
def test_gate_cuda_graph_owns_outputs():
    x=observation(33,'cuda:0');encoder=FlatEncoder(version=3)
    first=encoder(x);saved=first.clone()
    torch.testing.assert_close(first,encode_flat(x,3),rtol=0,atol=0)
    x[:,81]+=10;second=encoder(x)
    assert not torch.equal(first,second)
    torch.testing.assert_close(first,saved,rtol=0,atol=0)
