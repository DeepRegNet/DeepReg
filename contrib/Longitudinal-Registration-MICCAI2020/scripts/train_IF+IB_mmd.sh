cd ..

python main_h5.py \
	--key_file ./data/exp1-key-random.pkl \
	--data_file ./data/ImageWithSeg-0.7-0.7-0.7-64-64-51-SimpleNorm-RmOut.h5 \
        --batch_size 4 \
        --gpu 0 \
        --w_bde 50 \
	--w_mmd 0.01 \
        --epochs 1000 \
        --patient_cohort intra \
	--gpu_memory_control 1 \
        --exp_name IF+IB_mmd
