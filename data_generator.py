from data_loader import Data_loader
import numpy as np
from numpy.random import choice, shuffle
from torchvision import transforms
import torch


class Data_gernerator():
    def __init__(self, **config):
        self.path = config["path"]
        self.num_epoches = config["num_epoches"]
        self.task_batch = config["task_batch"]
        self.support_num = config["support_num"]
        self.query_num = config["query_num"]
        self.nways = config["nways"]

        self.characters = self.load()
        self.generator = self.sample_task()

    def load(self):
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        Ominiglot = Data_loader(path=self.path,
                                train=True,
                                transform=transform)
        characters = Ominiglot.characters
        return characters

    def sample_task(self):
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        count = 0
        while count < self.num_epoches:

            batch_data = []
            for i in range(self.task_batch):
                task_data = []
                classes = choice(len(self.characters), self.nways)
                shuffle(classes)
                for j in range(self.nways):
                    sel_imgs = choice(len(self.characters[classes[j]]), (self.query_num + self.support_num))
                    task_data.append([self.characters[classes[j]][_] for _ in sel_imgs])

                s = [one_class[0] for one_class in task_data]
                s_y = [_ for _ in range(self.nways)]
                s_list = list(zip(s, s_y))
                shuffle(s_list)
                s, s_y = zip(*s_list)
                s = torch.stack(s).to(device=device)
                s_y = torch.tensor(s_y).to(device=device)

                q = [_ for one_class in task_data for _ in one_class[1:]]
                q_y = [_ // self.query_num for _ in range(self.query_num * self.nways)]
                q_list = list(zip(q, q_y))
                shuffle(q_list)
                q, q_y = zip(*q_list)
                q = torch.stack(q).to(device=device)
                q_y = torch.tensor(q_y).to(device=device)

                batch_data.append((s, s_y, q, q_y))

            yield batch_data


if __name__ == "__main__":
    para= {}
    para["path"]= r"C:\Users\Miao_\Desktop\my_maml\omniglot"
    para["num_epoches"]= 100000
    para["task_batch"]= 32
    para["support_num"]= 1
    para["query_num"]= 15
    para["nways"]= 5

    a= Data_gernerator(**para)

    for i in a.generator:
        print(len(i))
        print(len(i[0]))
        print(len(i[0][0]))
        break
