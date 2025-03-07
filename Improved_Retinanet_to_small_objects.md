# Improved RetinaNet model for the application of small target detection in the aerial images

## Main ideas

### Model architecture

1. Change ResNet50 to ResNet152

2. Replace 3x3 convolution blocks by Scale Aggregation blocks
     -> this implementation was done with ScaleNet but the integration is really heavy in terms of GPU consumption. This    
        why we did not succeed to train with {image_size=2048*2048 / batch_size=1 and training_mixed_precision}.
        It could be a good idea to integrate mixed_precision in this project:     

    ```
        scaler = GradScaler()
        for epoch in range(nb_epochs):
            model.train()
            with tqdm(train_data_loader, unit="batch") as t_epoch:
                for images, targets in t_epoch:
                    optimizer.zero_grad()

                    with autocast(): # Enables mixed precision training
                        loss_dict = model(images, targets)
                    ...
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
    ```

3. Add P2 feature layer to FPN

4. Custom anchor sizes

5. Maximum and minimum IoU between the anchor and the GT box values decrease

6. Focal loss hyperparameters: α = 0,25 and γ = 3
    α = 0.25 and γ = 2 currently. It is possible to change it subclassing `torchvision.models.detection.retinanet.RetinaNetClassificationHead`
    and changing the `sigmoid_focal_loss` with *alpha* and *gamma* suggested values.