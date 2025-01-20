from transformers import ViTForImageClassification
pretrained_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')


pretrained_model.save_pretrained('pretrained_weights')
