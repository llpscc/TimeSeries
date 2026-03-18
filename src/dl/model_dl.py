from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch import Trainer


def build_tft(training):

    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=1e-3,
        hidden_size=16,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=16,
        loss=QuantileLoss(),
        output_size=7
    )

    return model


def train_tft(model, train_loader, val_loader):

    trainer = Trainer(
        max_epochs=3,
        accelerator="gpu",
        devices=1,
        gradient_clip_val=0.1,
        enable_progress_bar=True,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    return model