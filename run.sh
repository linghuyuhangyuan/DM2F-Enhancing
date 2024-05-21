python train_alg1.py --gpus 0 \
    --load-dm2f ckpts/ckpt/O-Haze/iter_20000_loss_0.04937_lr_0.000000.pth \
    --ckpt-path ckpts/ckpt_alg1

python test_alg1.py

python train_alg2.py --gpus 1 \
    --ckpt-path ckpts/ckpt_alg2

python test_alg2.py

