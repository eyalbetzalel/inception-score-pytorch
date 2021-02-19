import argparse
from inception_score import inception_score
import torch
from torchvision import datasets, transforms


# args :

parser = argparse.ArgumentParser()

parser.add_argument(
    "--path",
    type=str,
    help="Path to ImageGPT results",
    default=r"/home/dsi/eyalbetzalel/pytorch-generative-v6/tmp/run/ep_0_ch_128_psb_2_resb_4_atval_64_attk_8/",
)

args = parser.parse_args()

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])


dataset = datasets.ImageFolder(args.path, transform=transform)

mean_split_scores, std_split_scores = inception_score(dataset, cuda=True, batch_size=32, resize=True, splits=1)

print("Calculating Inception Score...")
print(mean_split_scores)
print(std_split_scores)


"""Computes the inception score of the generated images imgs

imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
cuda -- whether or not to run on GPU
batch_size -- batch size for feeding into Inception v3
splits -- number of splits
"""