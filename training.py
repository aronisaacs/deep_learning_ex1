import os
import torch
from tqdm.auto import tqdm
from evaluators import EvaluatorHolder




def train_model_epoch_eval(
    model,
    optimizer,
    train_loader,
    test_loader,
    loss_module,
    train_eval_loader=None,
    evaluator_holder=None,
    num_epochs=100,
    device="cpu",
    model_save_path="trained_model.pt",
):
    """
    Trains 'model' for 'num_epochs' and evaluates using the provided loaders.
    History includes a baseline metric snapshot from before epoch 1 (epoch 0).

    train_loader is used for optimization (can be oversampled), while
    train_eval_loader is used for unbiased train-set evaluation.

    Returns:
        EvaluatorHolder: populated holder with history and evaluators.
    """
    model.to(device)
    if train_eval_loader is None:
        train_eval_loader = train_loader
    if evaluator_holder is None:
        evaluator_holder = EvaluatorHolder(
            evaluators=[],
            device=device,
            loss_module=loss_module,
        )
    else:
        evaluator_holder.device = device
        evaluator_holder.set_loss_module(loss_module)

    # Baseline metrics before any training updates (epoch 0).
    evaluator_holder.update(
        model,
        train_eval_loader,
        test_loader,
    )

    for epoch in tqdm(range(num_epochs)):
        ####################################################
        # Training phase
        ####################################################
        model.train()

        for data_inputs, data_labels in train_loader:
            # 1) Move input data and labels to device
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)

            # 2) Forward pass
            preds = model(data_inputs)
            # If model output shape is [B,1] but your labels are [B],
            # you can squeeze the prediction:
            preds = preds.squeeze(dim=1)

            # 3) Compute loss
            loss = loss_module(preds, data_labels.float())

            # 4) Backprop + weight update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        evaluator_holder.update(
            model,
            train_eval_loader,
            test_loader,
        )

    if model_save_path:
        save_dir = os.path.dirname(model_save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        evaluator_holder.history["saved_model_path"] = model_save_path
    else:
        evaluator_holder.history["saved_model_path"] = None

    return evaluator_holder



