import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
import numpy as np

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w

        padding = tuple([self.pad] * 4)
        padded_x = F.pad(x, padding, 'replicate')

    
        eps = 1.0 / (h + 2 * self.pad)

        arange = torch.linspace(
            -1.0 + eps,
            1.0 - eps,
            h + 2 * self.pad,
            device=x.device,
            dtype=x.dtype
        )[:h]

        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)

        base_grid = torch.cat(
            [arange, arange.transpose(1, 0)],
            dim=2
        )

        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0,
            2 * self.pad + 1,
            size=(n, 1, 1, 2),
            device=x.device,
            dtype=x.dtype
        )

        shift_pixels = shift.clone()

        shift *= 2.0 / (h + 2 * self.pad)


        grid = base_grid + shift

        shifted_x = F.grid_sample(
            padded_x,
            grid,
            padding_mode='zeros',
            align_corners=False
        )

        return {
            "original": x,
            "padded": padded_x,
            "shifted": shifted_x,
            "shift_pixels": shift_pixels
        }



def tensor_to_image(tensor):
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    img = np.clip(img, 0, 1)

    return img


image_path = "Duck2.png"   

image = Image.open(image_path).convert("RGB")

transform = T.Compose([
    T.Resize((139, 139)),
    T.ToTensor()
])

img_tensor = transform(image).unsqueeze(0)

aug = RandomShiftsAug(pad=4)

results = aug(img_tensor)

original = tensor_to_image(results["original"])
padded = tensor_to_image(results["padded"])
shifted = tensor_to_image(results["shifted"])

shift_pixels = results["shift_pixels"].squeeze().cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(original)
axes[0].axis("off")
axes[1].imshow(padded)
axes[1].axis("off")
axes[2].imshow(shifted)
axes[2].axis("off")

plt.tight_layout()

plt.savefig("augmentation_stages.png", dpi=300)

plt.show()

print("Saved figure as augmentation_stages.png")