#!/bin/bash
# python train.py -s dataset/NeRF-DS/plate_novel_view/ -m output/plate_cosine --eval --iterations 20000 --ast_strategy "cosine" --ast_interval_steps 20000 --ast_decay_coef 0.5
# python train.py -s dataset/NeRF-DS/press_novel_view/ -m output/press_cosine --eval --iterations 20000 --ast_strategy "cosine" --ast_interval_steps 20000 --ast_decay_coef 0.5
# python train.py -s dataset/D-NeRF/hook/ -m output/hook_cosine --eval --ast_strategy "cosine" --ast_interval_steps 20000 --ast_decay_coef 0.5

python train.py -s dataset/NeRF-DS/plate_novel_view/ -m output/plate_cosine_restart --eval --iterations 20000 --ast_strategy "cosine" --ast_interval_steps 5000 --ast_decay_coef 0.5
python train.py -s dataset/NeRF-DS/press_novel_view/ -m output/press_cosine_restart --eval --iterations 20000 --ast_strategy "cosine" --ast_interval_steps 5000 --ast_decay_coef 0.5
python train.py -s dataset/D-NeRF/hook/ -m output/hook_cosine_restart --eval --ast_strategy "cosine" --ast_interval_steps 5000 --ast_decay_coef 0.5