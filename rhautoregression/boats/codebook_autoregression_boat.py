import torch

from rhautoregression.boats.base_autoregression_boat import BaseAutoregressionBoat
from rhtrain.utils.ddp_utils import move_to_device

class CodebookAutoregressionBoat(BaseAutoregressionBoat):
    """Autoregression boat that operates in a latent space with optional Vector Quantization (VQ)."""

    def __init__(self, config={}):
        super().__init__(config=config)

        # Optional freezing behavior
        if self.boat_config.get("freeze_latent_encoder", True):
            for p in self.models["latent_encoder"].parameters():
                p.requires_grad = False
            self.models['latent_encoder'].eval()

        if self.boat_config.get("freeze_vector_quantizer", True):
            for p in self.models["vector_quantizer"].parameters():
                p.requires_grad = False
            self.models['vector_quantizer'].eval()

        self.scaling_factor = float(getattr(getattr(self.models["latent_encoder"], "config", object()), "scaling_factor", 1.0))

    def predict(self, inputs):
        c_hat = super().predict(inputs)
        z_hat = self.dequantize_codes(c_hat)  # ensure continuous latents here
        x_hat = self.decode_latents(z_hat)
        return torch.clamp(x_hat, -1, 1)

    def training_calc_losses(self, batch):
        """Encode -> VQ -> AR-net -> loss."""
        gt = batch["gt"]

        with torch.no_grad():
            latents = self.encode_images(gt)
            codes = self.quantize_latents(latents)

        inputs = self._transform_inputs(codes)

        logits = self.models["net"](inputs)

        train_output = {"preds": logits, "targets": codes, **batch}
        net_loss = self.losses["net"](train_output)

        total_loss = net_loss

        losses = {
            "total_loss": total_loss,
            "net": net_loss,
        }

        return losses

    def validation_step(self, batch, batch_idx):
        """
        Report metrics in data space: encode -> (optional) VQ (for logging) -> predict from zeros -> decode -> metrics.
        """
        batch = move_to_device(batch, self.device)
        gt = batch["gt"]

        with torch.no_grad():
            codes = self.quantize_latents(self.encode_images(gt))
            x_recon = self.decode_latents(self.dequantize_codes(codes))

            # AR generation from zero codes in the same shape
            c_zeros = torch.zeros_like(codes)
            x_hat = self.predict(c_zeros)

            valid_output = {"preds": x_hat, "targets": gt}
            metrics = self._calc_metrics(valid_output)
            named_imgs = {"gt": gt, "gen": x_hat, "recon": x_recon}

        return metrics, named_imgs

    def decode_latents(self, z): # """Decode latents to the data space using the provided latent encoder/decoder."""
        return self.models["latent_encoder"].decode(z)

    def encode_images(self, x): # """Encode data to latent space using the provided latent encoder/decoder."""
        return self.models["latent_encoder"].encode(x)

    def quantize_latents(self, z): # """Quantize continuous latents to discrete codes using the provided vector quantizer."""
        return self.models["vector_quantizer"].assign(z)

    def dequantize_codes(self, c): # """indices (LongTensor) -> continuous latents using the codebook."""
        if hasattr(self.models["vector_quantizer"], "embed_code"):      # VQ-VAE style
            return self.models["vector_quantizer"].embed_code(c)
        if hasattr(self.models["vector_quantizer"], "dequantize"):      # some libs call it dequantize()
            return self.models["vector_quantizer"].dequantize(c)
        if hasattr(self.models["vector_quantizer"], "lookup"):          # or lookup()
            return self.models["vector_quantizer"].lookup(c)
        raise RuntimeError(
            "Vector quantizer does not expose an embedding/lookup method "
            "(tried: embed_code/dequantize/lookup)."
        )
