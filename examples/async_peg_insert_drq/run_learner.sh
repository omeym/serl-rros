export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
export XLA_PYTHON_CLIENT_ALLOCATOR=platform &&\
python3 async_drq_randomized.py "$@" \
    --learner 1\
    --env KukaPegInsert-Vision-v0 \
    --exp_name=serl_dev_drq_rlpd10demos_peg_insert_random_resnet_097 \
    --seed 0 \
    --random_steps 1000 \
    --training_starts 200 \
    --critic_actor_ratio 4 \
    --batch_size 256 \
    --eval_period 2000 \
    --encoder_type resnet-pretrained \
    --demo_path /home/omey/nisara/expert_data_collector/processed_replay/processed_replay/2013-01-01_01-09-43/recorded_data.pkl \
    --checkpoint_period 1000 \
    --loaded_checkpoint_step 1000 \
    --load_checkpoint_path /home/omey/SERL/src/examples/async_peg_insert_drq/checkpoints_12/ \
    --checkpoint_path /home/omey/SERL/src/examples/async_peg_insert_drq/checkpoints/
    # --load_checkpoint 0 \
    
