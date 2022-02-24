# python3 main.py --dataset ali --dim 64 --batch_size 1024 --gpu_id 0 --context_hops 2 --pool mean --ns rns --K 1 --n_negs 32

# python3 main.py --dataset ali --dim 64 --batch_size 1024 --gpu_id 0 --context_hops 2 --pool mean --ns mixgcf --K 1 --n_negs 32
# python3 co_training_main.py --dataset ali --dim 64 --batch_size 1024 --gpu_id 0 --context_hops 2 --pool mean --ns mixgcf --K 1 --n_negs 32 --load_pretrain_model True

# python3 co_training_main.py --dataset ali --dim 64 --batch_size 1024 --gpu_id 0 --context_hops 2 --pool mean --ns mixgcf --K 1 --n_negs 32 --classifier_lr 0.001 --load_pretrain_model True

# python3 co_training_main.py --dataset ali --dim 64 --batch_size 1024 --gpu_id 0 --context_hops 2 --pool mean --ns mixgcf --K 1 --n_negs 32 --classifier_lr 0.0001 --load_pretrain_model True

# python3 co_training_main.py --dataset ali --dim 64 --batch_size 1024 --gpu_id 0 --context_hops 2 --pool mean --ns mixgcf --K 1 --n_negs 32 --classifier_lambda_cot_max 10 --load_pretrain_model True

# python3 co_training_main.py --dataset ali --dim 64 --batch_size 1024 --gpu_id 0 --context_hops 2 --pool mean --ns mixgcf --K 1 --n_negs 32 --classifier_lambda_cot_max 1 --load_pretrain_model True

# python3 co_training_main.py --dataset ali --dim 64 --batch_size 1024 --gpu_id 0 --context_hops 2 --pool mean --ns mixgcf --K 1 --n_negs 32 --classifier_decay 0.001 --load_pretrain_model True

# python3 co_training_main.py --dataset ali --dim 64 --batch_size 1024 --gpu_id 0 --context_hops 2 --pool mean --ns mixgcf --K 1 --n_negs 32 --classifier_decay 0.00001 --load_pretrain_model True

python3 co_training_main.py --dataset yelp2018 --dim 64 --batch_size 1024 --gpu_id 0 --context_hops 2 --pool mean --ns mixgcf --K 1 --n_negs 32 --save_address_pretrain_model /tmp2/syliu/light_mixgcf/model_stored/lightgcn_pretrain_model_yelp2018.ckpt

# python3 co_training_main.py --dataset ali --dim 64 --batch_size 1024 --gpu_id 0 --context_hops 2 --pool mean --ns mixgcf --K 1 --n_negs 32 --classifier_lr 0.001 --load_pretrain_model True

# python3 co_training_main.py --dataset ali --dim 64 --batch_size 1024 --gpu_id 0 --context_hops 2 --pool mean --ns mixgcf --K 1 --n_negs 32 --classifier_lr 0.0001 --load_pretrain_model True

# python3 co_training_main.py --dataset ali --dim 64 --batch_size 1024 --gpu_id 0 --context_hops 2 --pool mean --ns mixgcf --K 1 --n_negs 32 --classifier_lambda_cot_max 10 --load_pretrain_model True

# python3 co_training_main.py --dataset ali --dim 64 --batch_size 1024 --gpu_id 0 --context_hops 2 --pool mean --ns mixgcf --K 1 --n_negs 32 --classifier_lambda_cot_max 1 --load_pretrain_model True

# python3 co_training_main.py --dataset ali --dim 64 --batch_size 1024 --gpu_id 0 --context_hops 2 --pool mean --ns mixgcf --K 1 --n_negs 32 --classifier_decay 0.001 --load_pretrain_model True

# python3 co_training_main.py --dataset ali --dim 64 --batch_size 1024 --gpu_id 0 --context_hops 2 --pool mean --ns mixgcf --K 1 --n_negs 32 --classifier_decay 0.00001 --load_pretrain_model True

python3 co_training_main.py --dataset yelp2018 --dim 64 --batch_size 1024 --gpu_id 0 --context_hops 2 --pool mean --ns mixgcf --K 1 --n_negs 32 --classifier_lr 0.001 --load_pretrain_model True --save_address_pretrain_model /tmp2/syliu/light_mixgcf/model_stored/lightgcn_pretrain_model_yelp2018.ckpt

python3 co_training_main.py --dataset yelp2018 --dim 64 --batch_size 1024 --gpu_id 0 --context_hops 2 --pool mean --ns mixgcf --K 1 --n_negs 32 --classifier_lr 0.0001 --load_pretrain_model True --save_address_pretrain_model /tmp2/syliu/light_mixgcf/model_stored/lightgcn_pretrain_model_yelp2018.ckpt

python3 co_training_main.py --dataset yelp2018 --dim 64 --batch_size 1024 --gpu_id 0 --context_hops 2 --pool mean --ns mixgcf --K 1 --n_negs 32 --classifier_lambda_cot_max 10 --load_pretrain_model True --save_address_pretrain_model /tmp2/syliu/light_mixgcf/model_stored/lightgcn_pretrain_model_yelp2018.ckpt

python3 co_training_main.py --dataset yelp2018 --dim 64 --batch_size 1024 --gpu_id 0 --context_hops 2 --pool mean --ns mixgcf --K 1 --n_negs 32 --classifier_lambda_cot_max 1 --load_pretrain_model True --save_address_pretrain_model /tmp2/syliu/light_mixgcf/model_stored/lightgcn_pretrain_model_yelp2018.ckpt

python3 co_training_main.py --dataset yelp2018 --dim 64 --batch_size 1024 --gpu_id 0 --context_hops 2 --pool mean --ns mixgcf --K 1 --n_negs 32 --classifier_decay 0.001 --load_pretrain_model True --save_address_pretrain_model /tmp2/syliu/light_mixgcf/model_stored/lightgcn_pretrain_model_yelp2018.ckpt

python3 co_training_main.py --dataset yelp2018 --dim 64 --batch_size 1024 --gpu_id 0 --context_hops 2 --pool mean --ns mixgcf --K 1 --n_negs 32 --classifier_decay 0.00001 --load_pretrain_model True --save_address_pretrain_model /tmp2/syliu/light_mixgcf/model_stored/lightgcn_pretrain_model_yelp2018.ckpt