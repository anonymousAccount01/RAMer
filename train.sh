

torchrun --nproc_per_node=1 \
main.py \
--do_train \
--measure_inference \
--epochs=150 \
--lr 5e-5  \
--gpu_id '0' \
--seed 42 \
--id 1 \
--visual_num_hidden_layers 3 \
--text_num_hidden_layers 3 \
--audio_num_hidden_layers 3 \
--binary_threshold 0.25 \
--recon_mse_weight 1.0 \
--aug_mse_weight 1.0 \
--beta_mse_weight 0.0 \
--lsr_clf_weight 0.01 \
--recon_clf_weight 1.0 \
--aug_clf_weight 0.1 \
--beta_clf_weight 1.0 \
--shuffle_aug_clf_weight 1.0 \
--cl_weight 1.0 \
--com_pri_weight 0.01 \
--diff_weight 5e-6 \
--cml_weight 0.5 \
--aligned \
--comp_rec_loss \

