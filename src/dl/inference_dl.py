import numpy as np


def predict_tft(model, val_loader, items, HORIZON):

    pred = model.predict(val_loader, return_x=True)

    preds = pred.output
    x = pred.x

    y_pred = preds.reshape(-1)
    y_true = x["decoder_target"].cpu().numpy().reshape(-1)

    dataset = val_loader.dataset
    encoder = dataset.categorical_encoders["item_nbr"]

    item_ids = encoder.inverse_transform(
        x["encoder_cat"][:, 0, 1].cpu().numpy()
    )

    item_ids = item_ids.astype(int)
    item_ids = np.repeat(item_ids, HORIZON)

    perishable_map = dict(zip(items["item_nbr"], items["perishable"]))

    perishable = np.array([perishable_map[i] for i in item_ids])
    weights = np.where(perishable == 1, 1.25, 1.0)

    min_len = min(len(y_pred), len(y_true))

    y_pred = y_pred[:min_len]
    y_true = y_true[:min_len]
    weights = weights[:min_len]

    y_pred = np.clip(y_pred, 0, None)
    y_true = np.clip(y_true, 0, None)

    y_pred = y_pred.cpu().numpy()

    return y_true, y_pred, weights