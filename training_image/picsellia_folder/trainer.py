import logging
import os

import torch
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

from evaluator import plot_precision_recall_curve
from utils import EarlyStopper, Averager, apply_loss_weights

try:
   from torch import GradScaler           # torch >= 2.3
except ImportError:
   from torch.cuda.amp import GradScaler  # torch < 2.3

try:
   from torch import autocast
except ImportError:
   from torch.cuda.amp import autocast


def evaluate_one_epoch(model, val_data_loader, device, metric):
    model.eval()

    for images, targets in val_data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            predictions = model(images)

        # send targets to GPU
        targets_gpu = []
        for j in range(len(targets)):
            targets_gpu.append({k: v.to(device=device, non_blocking=True) for k, v in targets[j].items()})

        metric.update(predictions, targets_gpu)

    return metric.compute()


def train_model(model, optimizer, train_data_loader, val_data_loader, lr_scheduler, lr_warmup, nb_epochs,
                path_saved_models: str, loss_coefficients: dict, patience: int, device, callback,
                mixed_precision: bool = False):
    def _on_end_training():
        torch.save(model.state_dict(), os.path.join(path_saved_models, 'latest.pth'))

        precision_recall_curve = plot_precision_recall_curve(
            validation_metrics=validation_metrics,
            recall_thresholds=torch.linspace(0.0, 1.00, round(1.00 / 0.01) + 1).tolist())

        output_dir = os.path.join(os.path.dirname(os.getcwd()), 'run', str(callback.get_experiment_id()))
        os.makedirs(output_dir, exist_ok=True)

        precision_recall_curve_plot_path = os.path.join(output_dir, 'precision_recall_curve.png')
        plt.savefig(precision_recall_curve_plot_path)

        callback.on_train_end(best_validation_map=best_map, path_saved_models=path_saved_models,
                              path_precision_recall_plot=precision_recall_curve_plot_path)

    early_stopper = EarlyStopper(patience=patience)
    visualisation_val_loss = True
    best_map = 0.0
    metric = MeanAveragePrecision(iou_type="bbox", max_detection_thresholds=[3000, 5000, 10000],
                                  extended_summary=True)

    loss_training_hist = Averager()
    loss_validation_hist = Averager()

    if mixed_precision:
        scaler = GradScaler()

    for epoch in range(nb_epochs):
        loss_training_hist.reset()
        loss_validation_hist.reset()

        model.train()
        with tqdm(train_data_loader, unit="batch") as t_epoch:

            for images, targets in t_epoch:
                t_epoch.set_description(f"Epoch {epoch}")
                optimizer.zero_grad()

                images = list(image.type(torch.FloatTensor).to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                '''
                Losses: 
                - Sigmoid focal loss for classification
                - l1 for regression
                '''
                if not mixed_precision:
                    loss_dict = model(images, targets)
                    loss_dict = apply_loss_weights(loss_dict=loss_dict, loss_coefficients=loss_coefficients)

                    total_loss = sum(loss for loss in loss_dict.values())
                    total_loss_value = total_loss.item()

                    loss_training_hist.send({
                        "regression": round(loss_dict['bbox_regression'].item(), 4),
                        "classification": round(loss_dict['classification'].item(), 4),
                        "total": round(total_loss_value, 4)
                    })  # Average out the loss

                    total_loss.backward()
                    optimizer.step()

                else:
                    # Use autocast to enable mixed-precision
                    with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                        loss_dict = model(images, targets)
                        loss_dict = apply_loss_weights(loss_dict=loss_dict, loss_coefficients=loss_coefficients)

                        total_loss = sum(loss for loss in loss_dict.values())
                        total_loss_value = total_loss.item()

                        loss_training_hist.send({
                            "regression": round(loss_dict['bbox_regression'].item(), 4),
                            "classification": round(loss_dict['classification'].item(), 4),
                            "total": round(total_loss_value, 4)
                        })  # Average out the loss

                    # Backward pass with scaled gradients
                    scaler.scale(total_loss).backward()

                    # Optimizer step with gradient scaling
                    scaler.step(optimizer)

                    # Update the scaler
                    scaler.update()

                t_epoch.set_postfix(
                    total_loss=loss_training_hist.value['total'],
                    bbox_loss=loss_training_hist.value['regression'],
                    cls_loss=loss_training_hist.value['classification']
                )

            if visualisation_val_loss:
                for images, targets in val_data_loader:
                    images = [image.to(device) for image in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    with torch.no_grad():
                        val_loss_dict = model(images, targets)

                    # apply coefficients for regression and classification losses
                    val_loss_dict = apply_loss_weights(loss_dict=val_loss_dict, loss_coefficients=loss_coefficients)
                    total_val_loss = sum(loss for loss in val_loss_dict.values())

                    loss_validation_hist.send({
                        "regression": val_loss_dict['bbox_regression'].item(),
                        "classification": val_loss_dict['classification'].item(),
                        "total": total_val_loss.item()
                    })

                if early_stopper.early_stop(validation_loss=loss_validation_hist.value['total']):
                    _on_end_training()
                    break

            # update the learning rate
            if not isinstance(lr_scheduler, torch.optim.lr_scheduler.CyclicLR):
                with lr_warmup.dampening():
                    if isinstance(lr_scheduler, ReduceLROnPlateau):
                        lr_scheduler.step(total_val_loss.item())
                    elif isinstance(lr_scheduler, StepLR):
                        lr_scheduler.step()
                    else:
                        raise f'Invalid lr scheduler policy: {type(lr_scheduler)}'

            else:
                lr_scheduler.step()

        # Evaluation
        validation_metrics = evaluate_one_epoch(model, val_data_loader, device, metric)

        # TODO display precision / recall in Picsellia interface
        '''
        - ``precision``: a tensor of shape ``(TxRxKxAxM)`` containing the precision values. Here ``T`` is the
                  number of IoU thresholds, ``R`` is the number of recall thresholds, ``K`` is the number of classes,
                  ``A`` is the number of areas and ``M`` is the number of max detections per image.
        - ``recall``: a tensor of shape ``(TxKxAxM)`` containing the recall values. Here ``T`` is the number of
          IoU thresholds, ``K`` is the number of classes, ``A`` is the number of areas and ``M`` is the number
          of max detections per image
        '''

        logging.info(f"Epoch #{epoch + 1} Training loss: {loss_training_hist.value} "
                     f"Validation loss {loss_validation_hist.value}"
                     f"- Accuracies: 'mAP' {float(validation_metrics['map']):.3} / "
                     f"'mAP[50]': {float(validation_metrics['map_50']):.3} / "
                     f"'mAP[75]': {float(validation_metrics['map_75']):.3} /"
                     f"'Precision': {float(validation_metrics['precision'][0][25][0][0][-1]):.3} / "
                     f"'Recall': {float(validation_metrics['recall'][0][0][0][-1]):.3} ")
        if validation_metrics['map'] >= best_map:
            best_map = float(validation_metrics['map'])
            torch.save(model.state_dict(), os.path.join(path_saved_models, 'best.pth'))

        callback.on_epoch_end(training_losses=loss_training_hist.value,
                              validation_losses=loss_validation_hist.value,
                              accuracies={
                                  'map': float(validation_metrics['map']),
                                  'mAP[50]': float(validation_metrics['map_50']),
                                  'mAP[75]': float(validation_metrics['map_75']),
                                  'precision': float(validation_metrics['precision'][0][25][0][0][-1]),
                                  'recall': float(validation_metrics['recall'][0][0][0][-1])
                              },
                              current_lr=optimizer.param_groups[0]['lr'])

        if not epoch == nb_epochs - 1:
            metric.reset()

    _on_end_training()
