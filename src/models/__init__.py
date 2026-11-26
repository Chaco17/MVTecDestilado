"""LightningModules of the three models. Each one exposes get_embedding(x) for the anomaly detector.

cnn.py       model A: ResNet-18 up to conv3_x + FC classifier, random weights
distill.py   model B: same student as A, distilled from a full ResNet-18
unet_ae.py   model C: U-Net autoencoder, the embedding is the bottleneck
"""
