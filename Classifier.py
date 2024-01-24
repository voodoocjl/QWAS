import json
from math import log2, ceil
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from Network import Attention, RNN, normalize
from FusionModel import cir_to_matrix

torch.cuda.is_available = lambda : False


def get_label(energy, mean = None):
    label = energy.clone()
    # if mean and (mean < float('inf')):
    #     energy_mean = mean
    # else:
    energy_mean = energy.mean()
    for i in range(energy.shape[0]):
        label[i] = energy[i] > energy_mean
    return label

def insert_job(change_code, job):
        if type(job[0]) == type([]):
            qubit = [sub[0] for sub in job]
        else:
            qubit = [job[0]]
            job = [job]
        if change_code != None:            
            for change in change_code:
                if change[0] not in qubit:
                    job.append(change)
        return job


class Classifier:
    def __init__(self, samples, arch_code, node_id):
        assert type(samples) == type({})        

        self.samples          = samples
        self.arch_code        = arch_code   #[qubits, layers]        
        self.input_dim_2d     = 21
        self.training_counter = 0
        self.node_layer       = ceil(log2(node_id + 2) - 1)
        # self.model            = Linear(self.input_dim_2d, 2)
        # self.model            = Mlp(self.input_dim_2d, 6, 2)        
        self.model            = RNN(arch_code[0], 16, 2)
        if torch.cuda.is_available():
            self.model.cuda()
        self.loss_fn          = nn.MSELoss()
        self.l_rate           = 0.001
        self.optimizer        = optim.Adam(self.model.parameters(), lr=self.l_rate, betas=(0.9, 0.999), eps=1e-08)
        self.epochs           = []
        self.training_accuracy = [0]
        self.boundary         = -1
        self.nets             = None
        self.maeinv           = None
        self.labels           = None
        self.mean             = 0        
        self.period           = 10
        


    def update_samples(self, latest_samples, arch):
        assert type(latest_samples) == type(self.samples)
        sampled_nets = []
        nets_maeinv  = []
        for k, v in latest_samples.items():
            net = json.loads(k)            
            sampled_nets.append(net)
            nets_maeinv.append(v)
        self.nets = torch.from_numpy(np.asarray(sampled_nets, dtype=np.float32))
        self.nets = normalize(self.nets)       
        self.maeinv = torch.from_numpy(np.asarray(nets_maeinv, dtype=np.float32).reshape(-1, 1))
        self.labels = get_label(self.maeinv)
        self.samples = latest_samples
        if torch.cuda.is_available():
            self.nets = self.nets.cuda()
            self.maeinv = self.maeinv.cuda()
            self.labels = self.labels.cuda()


    def train(self):
        if self.training_counter == 0:
            self.epochs = 3000
        else:
            self.epochs = 1000
        self.training_counter += 1
        # in a rare case, one branch has no networks
        if len(self.nets) == 0:
            return
        # linear, mlp
        nets = self.nets
        labels = 2 * self.labels - 1
        maeinv = self.maeinv
        train_data = TensorDataset(nets, maeinv, labels)
        train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
        for epoch in range(self.epochs):
            for x, y, z in train_loader:
                # clear grads
                self.optimizer.zero_grad()
                # forward to get predicted values
                outputs = self.model(x)                
                loss_mae = self.loss_fn(outputs[:, 0], y.reshape(-1))
                loss_t = self.loss_fn(outputs[:, -1], z.reshape(-1))
                # loss = loss_mae + loss_t
                loss = loss_t
                loss.backward()  # back props
                nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()  # update the parameters

        # training accuracy
        pred = self.model(nets).cpu()
        
        pred_label = (pred[:, -1] > 0.5).float()
        true_label = self.labels.reshape(-1).cpu()
        acc = accuracy_score(true_label.numpy(), pred_label.numpy())
        self.training_accuracy.append(acc)    

    
    def predict(self, remaining, arch):
        assert type(remaining) == type({})
        remaining_archs = []
        for k, v in remaining.items():
            net = json.loads(k)
            if arch['phase'] == 0:
                net = insert_job(arch['single'], net) 
                net = cir_to_matrix(net, arch['enta'], self.arch_code)                
            else:
                net = insert_job(arch['enta'], net)
                net = cir_to_matrix(arch['single'], net, self.arch_code)           
            remaining_archs.append(net)
        remaining_archs = torch.from_numpy(np.asarray(remaining_archs, dtype=np.float32))
        remaining_archs = normalize(remaining_archs)
                
        if torch.cuda.is_available():
            remaining_archs = remaining_archs.cuda()
        
        outputs = self.model(remaining_archs)        
        xbar = outputs[:, 0].mean().tolist()

        if torch.cuda.is_available():
            remaining_archs = remaining_archs.cpu()
            outputs         = outputs.cpu()
        result = {}
        for k in range(0, len(remaining_archs)):            
            arch_str = list(remaining.keys())[k]
            result[arch_str] = outputs[k].tolist()
        assert len(result) == len(remaining)
        return result, xbar


    def split_predictions(self, remaining, arch, method = None):
        assert type(remaining) == type({})
        samples_badness = {}
        samples_goodies = {}
        xbar = 0
        if len(remaining) == 0:
            return samples_goodies, samples_badness, 0
        if method == None:            
            predictions, xbar = self.predict(remaining, arch)  # arch_str -> pred_test_mae
            for k, v in predictions.items():
                # if v < self.sample_mean():
                # split by label
                if v[-1] < 0:
                    samples_badness[k] = v[0]
                else:
                    samples_goodies[k] = v[0]            
        else:
            predictions = np.mean(list(remaining.values()))
            for k, v in remaining.items():
                if v > predictions:
                    samples_badness[k] = v
                else:
                    samples_goodies[k] = v

        assert len(samples_badness) + len(samples_goodies) == len(remaining)
        return samples_goodies, samples_badness, xbar

    """
    def predict_mean(self):
        if len(self.nets) == 0:
            return 0
        # can we use the actual maeinv?
        outputs = self.model(self.nets)
        pred_np = None
        if torch.cuda.is_available():
            pred_np = outputs.detach().cpu().numpy()
        else:
            pred_np = outputs.detach().numpy()
        return np.mean(pred_np)
    """

    def sample_mean(self):
        if len(self.nets) == 0:
            return 0
        outputs = self.maeinv
        true_np = None
        if torch.cuda.is_available():
            true_np = outputs.cpu().numpy()
        else:
            true_np = outputs.numpy()
        return np.mean(true_np)


    def split_data(self, f1 = None):
        samples_badness = {}
        samples_goodies = {}
        if len(self.nets) == 0:
            return samples_goodies, samples_badness
        self.train()
        outputs = self.model(self.nets)[:, -1].reshape(-1, 1)        
        if torch.cuda.is_available():
            self.nets = self.nets.cpu()
            outputs   = outputs.cpu()
        predictions = {}
        for k in range(0, len(self.nets)):
            # arch = self.nets[k].detach().numpy().astype(np.int32)
            # arch_str = json.dumps(arch.tolist())
            arch_str = list(self.samples)[k]
            predictions[arch_str] = outputs[k].detach().numpy().tolist()[0]  # arch_str -> pred_test_mae
        assert len(predictions) == len(self.nets)
        # avg_maeinv = self.sample_mean()
        # self.boundary = avg_maeinv
        threshold = 0
        if self.arch_code[0] != 4:
            threshold = 0.5
        for k, v in predictions.items():
            # if v < self.sample_mean():
            if v < threshold:
                samples_badness[k] = self.samples[k]  # (val_loss, test_mae)
            else:
                samples_goodies[k] = self.samples[k]  # (val_loss, test_mae)
        assert len(samples_badness) + len(samples_goodies) == len(self.samples)
        return samples_goodies, samples_badness
