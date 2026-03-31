# a single A100 (80GB) for 600K steps, batch size 8 on each gpu

# small model
# python -m src.main +experiment=re10k \
# mode=test \
# wandb.name=re10k \
# dataset/view_sampler=evaluation \
# checkpointing.load=pretrained/re10k-256x256-surfsplat-small/checkpoints/epoch_18-step_300000.ckpt \
# test.save_image=true \
# test.save_gt_image=true \
# test.save_input_images=true \
# test.save_depth=true \
# test.save_depth_concat_img=true \
# test.save_depth_npy=true \
# test.save_gaussian=true \
# output_dir=outputs/re10k_small

# base model
# python -m src.main +experiment=re10k \
# mode=test \
# wandb.name=re10k \
# dataset/view_sampler=evaluation \
# model.encoder.num_scales=2 \
# model.encoder.upsample_factor=2 \
# model.encoder.lowest_feature_resolution=4 \
# model.encoder.monodepth_vit_type=vitb \
# checkpointing.load=checkpoints/re10k-256x256-surfsplat-base/checkpoints/epoch_72-step_600000.ckpt \
# test.save_image=true \
# test.save_gt_image=true \
# test.save_input_images=true \
# test.save_depth=true \
# test.save_depth_concat_img=true \
# test.save_depth_npy=true \
# test.save_gaussian=true \
# output_dir=outputs/re10k_base

# large model
# python -m src.main +experiment=re10k \
# mode=test \
# wandb.name=re10k \
# dataset/view_sampler=evaluation \
# model.encoder.num_scales=2 \
# model.encoder.upsample_factor=2 \
# model.encoder.lowest_feature_resolution=4 \
# model.encoder.monodepth_vit_type=vitl \
# checkpointing.load=checkpoints/re10k-256x256-surfsplat-large/checkpoints/epoch_72-step_600000.ckpt \
# test.save_image=true \
# test.save_gt_image=true \
# test.save_input_images=true \
# test.save_depth=true \
# test.save_depth_concat_img=true \
# test.save_depth_npy=true \
# test.save_gaussian=true \
# output_dir=outputs/re10k_base

# base model with higher resolution
# python -m src.main +experiment=re10k \
# mode=test \
# wandb.name=re10k \
# dataset/view_sampler=evaluation \
# model.encoder.num_scales=2 \
# model.encoder.upsample_factor=2 \
# model.encoder.lowest_feature_resolution=4 \
# model.encoder.monodepth_vit_type=vitb \
# checkpointing.load=checkpoints/re10k-256x448-surfsplat-base/checkpoints/epoch_72-step_600000.ckpt \
# test.save_image=true \
# test.save_gt_image=true \
# test.save_input_images=true \
# test.save_depth=true \
# test.save_depth_concat_img=true \
# test.save_depth_npy=true \
# test.save_gaussian=true \
# output_dir=outputs/re10k_base