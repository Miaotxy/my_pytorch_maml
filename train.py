import torch
import numpy as np
from numpy.random import choice as choice
from maml import Maml
from data_generator import Data_gernerator
from numpy.random import shuffle
from torch.autograd import grad
from util import make_functional


if __name__ == "__main__":

    config = {}
    config["path"] = r"C:\Users\Miao_\Desktop\my_maml\omniglot"
    config["num_epoches"] = 100000
    config["task_batch"] = 2
    config["support_num"] = 1
    config["query_num"] = 15
    config["nways"] = 5

    in_lr = 0.01
    meta_lr = 0.001

    data_gen = Data_gernerator(**config)

    def train():

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = Maml().to(device=device)
        
        parameters = list(model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=meta_lr)
        criterion = torch.nn.CrossEntropyLoss()

        for idx, batch_data in enumerate(data_gen.generator):

            loss = 0
            acc = 0

            for task_data in batch_data:

                s, s_y, q, q_y = task_data

                s_res = model(s)
                s_loss = criterion(s_res, s_y)
                s_grad = grad(s_loss, model.parameters(), create_graph=True)

                fast_weights = list(map(lambda p: p[1] - in_lr * p[0], zip(s_grad, parameters)))
                f_model = make_functional(model)

                q_res = f_model(q, params=fast_weights)
                q_loss = criterion(q_res, q_y)
                loss = q_loss if loss == 0 else loss + q_loss
                q_acc = (torch.argmax(q_res, dim=1) == q_y).sum().item() / len(q_y)
                acc = q_acc if acc == 0 else acc + q_acc

            if idx % 2:
                print("acc:{}   loss:{}".format(acc / config["task_batch"], loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    train()
