import glob
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class img_data(Dataset):

    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)

        # self.files_A = sorted(glob.glob(os.path.join(root,"%s/ground_random" % mode)+"/*.*"))
        self.files_A = sorted(glob.glob(os.path.join(root,"%s/groundtruth" % mode)+"/*.*"))
        # self.files_B = sorted(glob.glob(os.path.join(root,"%s/noise_random" % mode)+"/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root,"%s/noise" % mode)+"/*.*"))

    def __getitem__(self, index):

        image_A_path = self.files_A[index % len(self.files_A)]
        image_B_path = self.files_B[index % len(self.files_B)]

        image_A = Image.open(image_A_path)
        image_B = Image.open(image_B_path)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)

        return {"groundtruth": item_A,
                "noise": item_B,
                "img_path_ground": image_A_path,
                "img_path_noise": image_B_path}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
