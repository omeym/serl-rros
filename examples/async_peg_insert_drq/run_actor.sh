export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
export XLA_PYTHON_CLIENT_ALLOCATOR=platform &&\
python3 async_drq_randomized.py "$@" \
    --actor 1\
    --render \
    --env KukaPegInsert-Vision-v0 \
    --exp_name=serl_dev_drq_rlpd10demos_peg_insert_random_resnet \
    --seed 0 \
    --random_steps 0 \
    --training_starts 200 \
    --encoder_type resnet-pretrained \
    --demo_path /home/omey/SERL/src/data_collector/data_collector/feb_28_data/kuka_csv_logs/all_success_final.pkl \
    --eval_checkpoint_step 0 \
    --loaded_checkpoint_step 1000 \
    --eval_n_trajs 4 \
    --load_checkpoint_path /home/omey/SERL/src/examples/async_peg_insert_drq/checkpoints_12 \
    --checkpoint_path /home/omey/SERL/src/examples/async_peg_insert_drq/checkpoints_new/ \
    # --load_checkpoint 0 \
