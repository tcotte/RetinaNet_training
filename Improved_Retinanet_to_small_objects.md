# Improved RetinaNet model for the application of small target detection in the aerial images

## Main ideas

### Model architecture

1. Change ResNet50 to ResNet152

2. Replace 3x3 convolution blocks by Scale Aggregation blocks

3. Add P2 feature layer to FPN

4. Custom anchor sizes

5. Maximum and minimum IoU between the anchor and the GT box values decrease

6. Focal loss hyperparameters: α = 0,25 and γ = 3
    α = 0.25 and γ = 2 currently. It is possible to change it subclassing `torchvision.models.detection.retinanet.RetinaNetClassificationHead`
    and changing the `sigmoid_focal_loss` with *alpha* and *gamma* suggested values.