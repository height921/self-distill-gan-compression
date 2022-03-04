set -ex
python train.py \
 --dataroot ./datasets/horse2zebra/ \
 --name horse2zebra_cyclegan \
 --model cycle_gan \
 --ngf 12 \
 --netG mobile_resnet_9blocks \
 --real_stat_A_path ./real_stat/horse2zebra_A.npz \
 --real_stat_B_path ./real_stat/horse2zebra_B.npz \
 --gpu_ids 1 \
 --lambda_self_distill 1e1
