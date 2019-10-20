import os
import pickle
from PIL import Image
import torchvision.transforms as transforms


class Data_loader():

    def __init__(self, path, train, transform):
        self.path = path
        self.train = train
        self.transform = transform
        self.load_image()

    def _load_image(self):
        self.characters = []
        for (root, dir, files) in os.walk(self.path):
            images = []
            for file in files:
                if file.endswith(".png"):
                    file = root + "\\" + file
                    image = Image.open(file)
                    image = self.transform(image)
                    images.append(image)
            if len(images) != 0:
                self.characters.append(images)

    def load_image(self):
        file_name = self.path + [r"\val", r"\train"][self.train] + ".pickle"
        if self._exsit_image_pickle() and os.path.getsize(file_name) > 0:
            f = open(file_name, "rb")
            self.characters = pickle.load(f)
        else:
            f = open(file_name, "wb")
            self._load_image()
            pickle.dump(self.characters, f)

    def _exsit_image_pickle(self):
        file_name = self.path + [r"\val", r"\train"][self.train] + ".pickle"
        return os.path.exists(file_name)


if __name__ == "__main__":

    path = r"C:\Users\Miao_\Desktop\my_maml\omniglot"
    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    Ominiglot = Data_loader(path=path, train=True, transform=transform)
    characters = Ominiglot.characters
    print(len(characters))
    print(len(characters[0]))
    for idx,i in enumerate(characters):
        if len(i) == 0:
            print(idx)
