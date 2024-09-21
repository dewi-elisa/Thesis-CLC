#!/bin/bash

python train.py \
    --images \
    --load_data_path='/Users/dewi-elisa/Documents/Uni_uva/scriptie CLC/Thesis-CLC/Code/data/data_image_4_distractors.npz' \
    --shuffle_train_data \
    --sender_hidden=50 \
    --receiver_hidden=50 \
    --sender_embedding=50 \
    --receiver_embedding=50 \
    --sender_cell='lstm' \
    --receiver_cell='lstm' \
    --sender_lr=0.01 \
    --receiver_lr=0.001 \
    --temperature=1.0 \
    --mode='gs' \
    --dump_msg_folder='/Users/dewi-elisa/Documents/Uni_uva/scriptie CLC/Thesis-CLC/Code/data/messages/images/' \
    --random_seed=2085993795 \
    --checkpoint_freq=0 \
    --validation_freq=1 \
    --n_epochs=15 \
    --batch_size=2 \
    --optimizer='adam' \
    --lr=0.0001 \
    --update_freq=1 \
    --vocab_size=100 \
    --max_len=10 \
    --evaluate