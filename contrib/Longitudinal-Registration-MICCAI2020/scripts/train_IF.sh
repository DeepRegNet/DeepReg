cd ..

python main_h5.py \
	--key_file ./data/exp1-key-ordered.pkl \
        --batch_size 4 \
        --gpu 0 \
        --w_bde 50 \
	--w_mmd 0 \
        --epochs 1000 \
        --patient_cohort intra \
	--gpu_memory_control 1 \
        --exp_name IF
