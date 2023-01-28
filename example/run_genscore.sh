#!/bin/bash

# input is protein (needs to be converted to pocket)
python genscore.py -p ./1qkt_p.pdb -l ./1qkt_decoys.sdf -rl ./1qkt_l.sdf -gen_pocket -c 10.0 -e gt -m ../trained_models/GT_0.0_1.pth

# input is pocket
python genscore.py -p ./1qkt_p_pocket_10.0.pdb -l ./1qkt_decoys.sdf -e gatedgcn -m ../trained_models/GatedGCN_0.5_1.pth


# calculate the atom contributions of the score
python genscore.py -p ./1qkt_p_pocket_10.0.pdb -l ./1qkt_decoys.sdf -e gatedgcn -ac -m ../trained_models/GatedGCN_ft_1.0_1.pth


# calculate the residue contributions of the score
python genscore.py -p ./1qkt_p_pocket_10.0.pdb -l ./1qkt_decoys.sdf -e gatedgcn -rc -m ../trained_models/GatedGCN_ft_1.0_1.pth

