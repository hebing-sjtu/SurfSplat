# a single A100 (80GB) for 600K steps, batch size 8 on each gpu

# small model
# python -m src.main +experiment=re10k \
# data_loader.train.batch_size=8 \
# dataset.test_chunk_interval=10 \
# trainer.max_steps=600000 \
# model.encoder.upsample_factor=4 \
# model.encoder.lowest_feature_resolution=4 \
# checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
# checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
# output_dir=checkpoints/re10k-256x256-surfsplat-small

# base model
# python -m src.main +experiment=re10k \
# data_loader.train.batch_size=8 \
# dataset.test_chunk_interval=10 \
# trainer.max_steps=600000 \
# model.encoder.num_scales=2 \
# model.encoder.upsample_factor=2 \
# model.encoder.lowest_feature_resolution=4 \
# model.encoder.monodepth_vit_type=vitb \
# checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vitb.pth \
# checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
# output_dir=checkpoints/re10k-256x256-surfsplat-base

# large model
# python -m src.main +experiment=re10k \
# data_loader.train.batch_size=8 \
# dataset.test_chunk_interval=10 \
# trainer.max_steps=600000 \
# model.encoder.num_scales=2 \
# model.encoder.upsample_factor=2 \
# model.encoder.lowest_feature_resolution=4 \
# model.encoder.monodepth_vit_type=vitl \
# checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vitl.pth \
# checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
# output_dir=checkpoints/re10k-256x256-surfsplat-large

# base model + higher resolution
# python -m src.main +experiment=re10k \
# data_loader.train.batch_size=8 \
# dataset.test_chunk_interval=10 \
# dataset.image_shape=[256,448] \
# trainer.max_steps=600000 \
# model.encoder.num_scales=2 \
# model.encoder.upsample_factor=4 \
# model.encoder.lowest_feature_resolution=8 \
# model.encoder.monodepth_vit_type=vitb \
# checkpointing.resume=true \
# checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vitb.pth \
# checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
# output_dir=checkpoints/re10k-256x448-surfsplat-base

