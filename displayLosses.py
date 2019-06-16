import sys
import torch
import torchvision
import visdom

vis = visdom.Visdom()

t = torch.load(sys.argv[1])

print(t.shape)
vis.line(Y=t[:,200:].t(), X=torch.arange(t.shape[1])[200:], win="GRADIENTS", opts={'legend':['G1', 'G2', 'G3', 'G4', 'G5', 'H1', 'H2', 'H3', 'H4', 'H5']})
