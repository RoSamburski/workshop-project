import os
import re
from enum import Enum
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import read_image

"""Dataset Constants"""
DATASET_ROOT = "data"

GENERATOR_INPUT = "references"

LABELS = "labels.csv"

MAX_ANIMATION_LENGTH = 16


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
        return "{}(ALT)?[LR]?[0-9]?".format(self.name)

    @classmethod
    def matching_type(cls, string: str):
        for member in list(cls):
            if re.match(cls.regex_of(member), string) is not None:
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
    def __init__(self, labeling_file, root_dir, transform=None, target_transform=None):
        self.label_set = pd.read_csv(labeling_file)
        self.root_dir = root_dir
        self.data_set = self.build_dataset()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        char_path = Path(os.path.join(self.root_dir, self.data_set[idx]))
        reference = None
        animation = {}
        try:
            label = self.label_set.iloc([idx, 1])
            if self.target_transform:
                label = self.target_transform(label)
        except IndexError:
            # Data is unlabeled
            label = None
        for image in char_path.iterdir():
            if "REF" in image.name.upper():
                # Reference image
                reference = read_image(os.path.join(self.root_dir, self.data_set[idx], image))
                if self.transform:
                    reference = self.transform(reference)
            else:
                index = int(image.name.split(".")[0])
                animation[index] = read_image(os.path.join(self.root_dir, self.data_set[idx], image))
                if self.transform:
                    animation[index] = self.transform(animation[index])
        assert reference is not None, "Missing reference for {}".format(char_path)
        indices = [i for i in range(len(animation))]
        if len(animation) > MAX_ANIMATION_LENGTH:
            # Handling of animations that are "too long" for us.
            indices = np.random.choice(indices, replace=False).sort()
        item = np.ndarray([reference] + [animation[i] for i in indices] + [torch.zeros(len(animation[0]))])
        return item, label

    def build_dataset(self):
        data_set = []
        for labeled_data in self.label_set:
            data_set.append(labeled_data[0])
        for folder in Path(self.root_dir).iterdir():
            if folder not in data_set:
                data_set.append(folder.name)
        return data_set


def verify(width_limit=80, height_limit=80):
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
                if dimensions[1] > max_height:
                    max_height = dimensions[1]
                    longest_image = folder.name + "/" + frame
                if dimensions[2] > max_width:
                    max_width = dimensions[2]
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
    print("{} images are exceeding 80x80".format(exceeding))


def generate_label_file():
    """
    A basic label generation script relying on naming convention:
    <Origin><Sprite><Animation><Alt?>
    :return: None
    """
    label_file = open(LABELS, "w")
    for folder in Path(DATASET_ROOT).iterdir():
        if folder.is_dir():
            name = folder.name
            label = AnimationType.matching_type(name.upper())
            if label is not None:
                label_file.write("{},{}\n".format(name, label.name))
    label_file.close()


def main():
    while True:
        action = input("Which action to take?\nVERIFY dataset and get statistics\nGENERATE label file\nQUIT\n")
        if action.upper() == "VERIFY":
            print("Verifying dataset")
            try:
                frame_boundary = input("What width/height should the sprites be limited to?")
                width_boundary, height_boundary = int(frame_boundary.split()[0]), int(frame_boundary.split()[1])
                verify(width_boundary, height_boundary)
            except TypeError:
                print("")
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
        elif action.upper() == "QUIT":
            exit(0)


if __name__ == "__main__":
    main()
