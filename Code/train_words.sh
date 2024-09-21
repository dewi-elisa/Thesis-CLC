#!/bin/bash

python train.py \
    --load_data_path='/Users/dewi-elisa/Documents/Uni_uva/scriptie CLC/Thesis-CLC/Code/data/data_word_4_distractors.npz' \
    --shuffle_train_data \
    --sender_hidden=50 \
    --receiver_hidden=50 \
    --sender_embedding=10 \
    --receiver_embedding=10 \
    --sender_cell='lstm' \
    --receiver_cell='lstm' \
    --sender_lr=0.001 \
    --receiver_lr=0.001 \
    --temperature=1.0 \
    --mode='gs' \
    --dump_msg_folder='/Users/dewi-elisa/Documents/Uni_uva/scriptie CLC/Thesis-CLC/Code/data/messages/words' \
    --random_seed=42 \
    --data_seed=42 \
    --checkpoint_freq=0 \
    --validation_freq=1 \
    --n_epochs=50 \
    --batch_size=32 \
    --optimizer='adam' \
    --lr=0.0001 \
    --update_freq=1 \
    --vocab_size=100 \
    --max_len=10 \
    --evaluate