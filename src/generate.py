import os.path
import sys
from pathlib import Path

import torch
from PIL import Image
import matplotlib.pyplot as plt

import gan_model
import database_handler
import autoencoder


def generate(reference_path: str, animation_type: database_handler.AnimationType, frame_count: int) -> torch.Tensor:
    encoder = autoencoder.ConvolutionalAutoencoder(gan_model.ngpu)
    generator = gan_model.Generator(gan_model.ngpu, encoder.encoder)
    model = torch.load(os.path.join(gan_model.model_folder, gan_model.model_name), map_location=gan_model.device)
    generator.load_state_dict(model["generator"])
    generator.eval()

    reference_image = gan_model.dataset_transform([Image.open(reference_path)])[0]\
        .view(1, 1, gan_model.nc, database_handler.IMAGE_SIZE, database_handler.IMAGE_SIZE).to(gan_model.device)
    tags = torch.Tensor((animation_type.value, frame_count)).view(1, 2).to(gan_model.device)
    noise = torch.randn(1, gan_model.nz, 1, 1, 1, device=gan_model.device)
    return generator((noise, reference_image, tags)).detach().cpu()[0]


def main():
    if len(sys.argv) < 5:
        print("USAGE: python3 generate.py <path-to-reference-image> <animation-type> <animation-length> <out-folder>")
        return
    reference_path = sys.argv[1]
    animation_type = database_handler.AnimationType.parse_string(sys.argv[2].upper())
    # Input validity checking
    if animation_type is None:
        print("Animation type must be of the following:")
        for member in list(database_handler.AnimationType):
            print(member.name)
        return
    try:
        animation_length = int(sys.argv[3])
    except TypeError:
        print("Animation length must be an integer")
        return
    if not (1 <= animation_length <= database_handler.MAX_ANIMATION_LENGTH):
        print("Animation length must be between 1 and {}".format(database_handler.MAX_ANIMATION_LENGTH))
        return

    output = generate(reference_path, animation_type, animation_length)
    plt.title("Result")
    for j in range(database_handler.MAX_ANIMATION_LENGTH + 1):
        plt.subplot(4, 5, j + 1)
        plt.axis("off")
        plt.imshow(database_handler.IMAGE_TRANSFORM(output[j]))
    plt.show()
    out_folder = Path(sys.argv[4])
    if not out_folder.exists():
        out_folder.mkdir()
    for i in range(animation_length):
        frame = database_handler.IMAGE_TRANSFORM(output[i+1])
        frame.save(os.path.join(out_folder, "{}.png".format(i)))


if __name__ == "__main__":
    main()
