import os
import re
from enum import Enum
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import read_image
import matplotlib.pyplot as plt
from PIL import Image

"""Dataset Constants"""
DATASET_ROOT = "data"

GENERATOR_INPUT = "references"

LABELS = "labels.csv"

MAX_ANIMATION_LENGTH = 16

IMAGE_SIZE = 64

IMAGE_TRANSFORM = T.Compose([
        T.ToPILImage(),
        np.array,
    ])


class AnimationType(Enum):
    """ Enums for value representation """
    IDLE = 0
    WALK = 1
    RUN = 2
    JUMP = 3

    def regex_of(self):
        """
        Returns the regex used for matching according to naming convention
        :return: the string of the regex
        """
        return ".*{}(ALT)?[LR]?[0-9]?$".format(self.name)

    @classmethod
    def matching_type(cls, string: str):
        for member in list(cls):
            if re.match(cls.regex_of(member), string) is not None:
                return member
        return None

    @classmethod
    def parse_string(cls, string: str):
        for member in list(cls):
            if member.name == string:
                return member
        return None


class Direction(Enum):
    L = 0
    R = 1


class AnimationDataset(Dataset):
    """
    Dataset class made for loading our dataset.
    Every "data-point" is a folder containing:
    1. Reference image (titled REF.png)
    2. Animation frames (titled as numbers by order: 0.png to <MAL-1>.png)
    The labeling file contains the folder names, followed by the metadata of the animation (direction+type)
    """
    def __init__(self, labeling_file, root_dir, transform=None, target_transform=None, use_palette_swap=False, use_negative=False):
        self.label_set = pd.read_csv(labeling_file)
        self.root_dir = root_dir
        self.full_dataset = self.build_dataset()
        self.transform = transform
        self.target_transform = target_transform
        # Whether to allow shuffling of RGB channels.
        # Shuffling multiplies the usable dataset size by 6
        self.use_palette_swap = use_palette_swap
        self.palette_permutations = [
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        ]
        # Whether to allow negatives of RGB channels.
        # Flipping some or all channels multiplies the usable dataset size by 8
        self.use_negative = use_negative
        self.negative_filters = [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        ]

    def __len__(self):
        if self.use_palette_swap and self.use_negative:
            return 48*len(self.label_set)
        elif self.use_palette_swap:
            return 6*len(self.label_set)
        elif self.use_negative:
            return 8*len(self.label_set)
        return len(self.label_set)

    def negative_transformation(self, frame, mask_idx):
        channels = frame.split()
        neg_mask = self.negative_filters[mask_idx]
        out_channels = []
        for i in range(3):
            if neg_mask[i] == 1:
                out_channels.append(channels[i].point(lambda x: 255 - x))
            else:
                out_channels.append(channels[i])
        out_channels.append(channels[3])
        return Image.merge(mode="RGBA", bands=out_channels)

    def permutation_transformation(self, frame, perm_idx):
        channels = frame.split()
        permutation = self.palette_permutations[perm_idx]
        # Reorder the RGB channels. leave Alpha channel at place
        out_channels = (channels[permutation[0]],
                        channels[permutation[1]],
                        channels[permutation[2]],
                        channels[3])
        return Image.merge(mode="RGBA", bands=out_channels)

    def __getitem__(self, idx):
        if self.use_palette_swap and self.use_negative:
            char_path = Path(os.path.join(self.root_dir, self.label_set.iloc[idx//48, 0]))
        elif self.use_palette_swap:
            char_path = Path(os.path.join(self.root_dir, self.label_set.iloc[idx//6, 0]))
        elif self.use_negative:
            char_path = Path(os.path.join(self.root_dir, self.label_set.iloc[idx//8, 0]))
        else:
            char_path = Path(os.path.join(self.root_dir, self.label_set.iloc[idx, 0]))
        # print("Fetching {}".format(char_path))
        reference = None
        animation = {}
        for image in char_path.iterdir():
            if "REF" in image.name.upper():
                # Reference image
                reference = Image.open(str(image))
                # if self.transform:
                #     reference = self.transform(reference)
            else:
                index = int(image.name.split(".")[0])
                animation[index] = Image.open(str(image))
                # if self.transform:
                #     animation[index] = self.transform(animation[index])
        assert reference is not None, "Missing reference for {}".format(char_path)
        if len(animation) > MAX_ANIMATION_LENGTH:
            # Handling of animations that are "too long" for us.
            indices = np.random.choice(len(animation), size=MAX_ANIMATION_LENGTH, replace=False)
            indices.sort()
            animation = [animation[i] for i in indices]
        else:
            animation = [animation[i] for i in range(len(animation))]
        item = [reference] + animation
        if self.use_negative and self.use_palette_swap:
            item = [self.negative_transformation(frame, idx % 8) for frame in item]
            item = [self.permutation_transformation(frame, (idx//8) % 6) for frame in item]
        elif self.use_negative:
            item = [self.negative_transformation(frame, idx % 8) for frame in item]
        elif self.use_palette_swap:
            item = [self.permutation_transformation(frame, idx % 6) for frame in item]
        if self.transform:
            item = self.transform(item)
        try:
            # Label format:
            # (Type, Animation-Length)
            # Currently ignores direction
            if self.use_palette_swap:
                label = (AnimationType.parse_string(self.label_set.iloc[idx//6, 1]),
                     min(len(animation), MAX_ANIMATION_LENGTH))
            else:
                label = (AnimationType.parse_string(self.label_set.iloc[idx, 1]),
                         min(len(animation), MAX_ANIMATION_LENGTH))
            if self.target_transform:
                label = self.target_transform(label)
        except IndexError:
            # Data is unlabeled
            label = None
        return item, label

    def build_dataset(self):
        data_set = []
        for labeled_data in self.label_set:
            data_set.append(labeled_data)
        for folder in Path(self.root_dir).iterdir():
            if folder.is_dir() and folder.name not in data_set:
                data_set.append(folder.name)
        # print("{} {}".format(data_set[0], data_set[1]))
        return data_set

    def get_reference_frame(self, idx):
        return self[idx][0][0]

    def fetch_by_name(self, name):
        return self[self.full_dataset.index(name)]


def verify(width_limit=IMAGE_SIZE, height_limit=IMAGE_SIZE):
    """
    Validates the database and prints any errors in it.
    Prints statistics
    :return: None
    """
    animation_lengths = []
    max_height, max_width = 0, 0
    longest_image, widest_image = None, None
    exceeding = 0
    print("Database size: {} objects".format(len([x for x in Path(DATASET_ROOT).iterdir() if x.is_dir()])))
    for folder in Path(DATASET_ROOT).iterdir():
        if folder.is_dir():
            frames = [x.name for x in folder.iterdir()]
            for frame in frames:
                frame_tensor = read_image(os.path.join(DATASET_ROOT, folder.name, frame))
                dimensions = list(frame_tensor.size())
                if dimensions[0] < 4:
                    # Image is not in RGBA
                    print("{}\\{}: Erroneous color channels".format(folder.name, frame))
                if dimensions[len(dimensions)-2] > max_height:
                    max_height = dimensions[len(dimensions)-2]
                    longest_image = folder.name + "/" + frame
                if dimensions[-1] > max_width:
                    max_width = dimensions[-1]
                    widest_image = folder.name + "/" + frame
                if dimensions[1] > height_limit or dimensions[2] > width_limit:
                    exceeding += 1
            if len(frames) == 0:
                print("{}: empty".format(folder.name))
            elif len(frames) == 1:
                print("{}: missing data (only 1 file: {})".format(folder.name, frames[0]))
            else:
                has_reference = False
                for frame in frames:
                    if "REF" in frame.upper():
                        has_reference = True
                if not has_reference:
                    print("{}: no reference".format(folder.name))
                else:
                    animation_lengths.append(len(frames)-1)
    print("Recommended animation length: {}".format(np.percentile(animation_lengths, 99)))
    print("Greatest dimensions in dataset: {} ({}) x{} ({})".format(max_width, widest_image, max_height, longest_image))
    print("{} images are exceeding {}x{}".format(exceeding, width_limit, height_limit))


def generate_label_file():
    """
    A basic label generation script relying on naming convention:
    <Origin><Sprite><Animation><Alt?>
    :return: None
    """
    label_file = open(LABELS, "w")
    label_file.write("Name, Tag\n")
    for folder in Path(DATASET_ROOT).iterdir():
        if folder.is_dir():
            name = folder.name
            label = AnimationType.matching_type(name.upper())
            if label is not None:
                label_file.write("{},{}\n".format(name, label.name))
    label_file.close()


def test_data_fetching():
    dataset = AnimationDataset(labeling_file=LABELS, root_dir=DATASET_ROOT,
                               use_palette_swap=True, use_negative=True)
    done = False
    while not done:
        test_type = input("Fetch Type:\n"
                          "Specific\n"
                          "Randomized\n"
                          "Full\n")
        if test_type[0].upper() == "S":
            specific_image_fetch(dataset)
            done = True
        elif test_type[0].upper() == "R":
            random_data_fetch(dataset)
            done = True
        elif test_type[0].upper() == "F":
            full_fetch(dataset)
            done = True
        else:
            print("Invalid command")


def random_data_fetch(dataset):
    for i in range(10):
        fetch_image, fetch_label = dataset[np.random.randint(0, len(dataset))]
        print(fetch_label)
        print(fetch_image.shape)
        for j in range(0, len(fetch_image)):
            plt.subplot(5, 4, j+1)
            plt.imshow(IMAGE_TRANSFORM(fetch_image[j]))
        # plt.subplot(2, 2, 1)
        # plt.imshow(IMAGE_TRANSFORM(fetch_image[0][0]))  # R
        # plt.subplot(2, 2, 2)
        # plt.imshow(IMAGE_TRANSFORM(fetch_image[0][1]))  # G
        # plt.subplot(2, 2, 3)
        # plt.imshow(IMAGE_TRANSFORM(fetch_image[0][2]))  # B
        # plt.subplot(2, 2, 4)
        # plt.imshow(IMAGE_TRANSFORM(fetch_image[0][3]))  # Alpha
        plt.show()


def specific_image_fetch(dataset):
    try:
        index = int(input("Which image to fetch?\n"))
    except TypeError:
        print("Invalid Type")
        return
    fetch_image, fetch_label = dataset[index]
    print(fetch_label)
    for j in range(0, len(fetch_image)):
        plt.subplot(5, 4, j+1)
        plt.imshow(fetch_image[j])
        # plt.subplot(4, 2, 1)
        # plt.imshow(IMAGE_TRANSFORM(fetch_image[j][0]))  # B
        # plt.subplot(4, 2, 2)
        # plt.imshow(IMAGE_TRANSFORM(fetch_image[j][1]))  # R
        # plt.subplot(4, 2, 3)
        # plt.imshow(IMAGE_TRANSFORM(fetch_image[j][2]))  # G
        # plt.subplot(4, 2, 4)
        # plt.imshow(IMAGE_TRANSFORM(fetch_image[j][3]))  # Alpha
    plt.show()


def full_fetch(dataset):
    count = 0
    for i in range(len(dataset)):
        try:
            fetch_image, fetch_label = dataset[i]
            count += 1
        except Exception as e:
            print("Error fetching #{}:\n{}".format(i, e))
    print("done (fetched {}/{} successfully)".format(count, len(dataset)))


def main():
    while True:
        action = input("Which action to take?\n"
                       "VERIFY dataset and get statistics\n"
                       "GENERATE label file\n"
                       "TEST dataset fetching\n"
                       "QUIT\n")
        if action.upper() == "VERIFY":
            print("Verifying dataset")
            try:
                frame_boundary = input("What width/height should the sprites be limited to?")
                width_boundary, height_boundary = int(frame_boundary.split()[0]), int(frame_boundary.split()[1])
                verify(width_boundary, height_boundary)
            except TypeError:
                print("Please provide numbers")
            except IndexError:
                print("Not enough values provided. Proceeding with default.")
                verify()
        elif action.upper() == "GENERATE":
            label_file = Path(LABELS)
            if label_file.exists():
                create_file = input("Label file existing. Truncate? (Y/N)").upper()
                if create_file.upper() == "Y":
                    print("Generating label file")
                    generate_label_file()
            else:
                print("Generating label file")
                generate_label_file()
        elif action.upper() == "TEST":
            print("Begin testing")
            test_data_fetching()
        elif action.upper() == "QUIT":
            exit(0)


if __name__ == "__main__":
    main()
