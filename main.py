import torch
import torch.nn as nn
from ffnn.model import QMLP

def main():
    test_in = torch.randn((64, 168, 5))
    model = QMLP(in_dim=168*5)
    
    out = model(test_in.view(64, -1))
    print(out.shape)

if __name__ == "__main__":
    main()