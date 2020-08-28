cd ..
python test.py --exp_name The_name_of_the_experiments_you_want_to_test \
	       --data_file /path/to/your/data_file/xxx.h5 \
	       --key_file /path/to/your/key_file/xxx.pkl \
	       --continue_epoch 999 \
	       --test_phase holdout \
	       --test_gen_pred_imgs 1 \
               --gpu 0 \
	       --suffix dsc \
               --test_mode 1
