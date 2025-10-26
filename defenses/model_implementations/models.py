
# This code is adapted from pad4amd by deqangss
# Repository: https://github.com/deqangss/pad4amd


import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.utils import evaluate_predictions


class BaseModel(nn.Module):
    def __init__(self, model_name, device=None):
        super().__init__()
        self.model_name = model_name
        self.device = device if device else torch.device("cpu")
        self.to(self.device)
    
    
    def predict(self, test_loader, indicator_masking=True):

        y_cent, x_prob, y_true = self.inference(test_loader)
        y_pred = y_cent.argmax(1).cpu().numpy()
        y_true = y_true.cpu().numpy()
        indicator_flag = self.indicator(x_prob, y_pred).cpu().numpy()


        if indicator_masking:
            y_pred, y_true = y_pred[indicator_flag], y_true[indicator_flag]
            excluded_samples = (len(y_cent) - len(y_true)) / len(y_cent)
            print(f"\n**************************************\nModel: {self.model_name}")
            print(f"Deferred option: \n----------")
            print(f"Excluded samples: {excluded_samples * 100:.2f}%")
        else:
            y_pred[~indicator_flag] = 1
            print(f"\n**************************************\nModel: {self.model_name}")
            print(f"Conservative option: \n----------")
        evaluate_predictions(y_true, y_pred)


    def inference(self, test_data):
        y_cent, x_prob, gt_labels = [], [], []
        self.eval()
        with torch.no_grad():
            for x, y in test_data:
                x, y = x.double().to(self.device ), y.long().to(self.device )
                logits, logits_g = self.forward(x)
                y_cent.append(F.softmax(logits, dim=-1))
                x_prob.append(logits_g)
                gt_labels.append(y)
        return torch.cat(y_cent, dim=0), torch.cat(x_prob, dim=0), torch.cat(gt_labels, dim=0)


    def get_tau_sample_wise(self, y_pred=None):
        return self.tau

    def indicator(self, x_prob, y_pred=None):
        return x_prob <= self.tau


    def get_threshold(self, validation_data_producer, ratio=None):
        """
        get the threshold for adversary detection
        :@param validation_data_producer: Object, an iterator for producing validation dataset
        """
        self.eval()
        ratio = ratio if ratio is not None else self.ratio
        assert 0 <= ratio <= 1
        probabilities = []
        with torch.no_grad():
            for x_val, y_val in validation_data_producer:
                x_val, y_val = x_val.double().to(self.device ), y_val.long().to(self.device )
                _1, x_prob = self.forward(x_val)
                probabilities.append(x_prob)
            s, _ = torch.sort(torch.cat(probabilities, dim=0))
            i = int((s.shape[0] - 1) * ratio)
            assert i >= 0
            self.tau[0] = s[i]

    def load(self, root_path):
        model_path = os.path.join(root_path, f'{self.model_name}.pth')
        ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            self.load_state_dict(ckpt['model'])
        else:
            self.load_state_dict(ckpt)


class MalwareDetectionDNN(BaseModel):
    def __init__(self, model_name, input_size=10000, n_classes=2, device=None):
        super().__init__(model_name, device)
        self.input_size = input_size
        self.n_classes = n_classes
        self.nn_model_layer_0 = nn.Linear(self.input_size, 200)
        self.nn_model_layer_1 = nn.Linear(200, 200)
        self.nn_model_layer_2 = nn.Linear(200, self.n_classes)

    def forward(self, x, return_all_layers=False):
        x = x.to(self.device)
        layer_0_out = F.relu(self.nn_model_layer_0(x))
        layer_1_out = F.relu(self.nn_model_layer_1(layer_0_out))
        output = self.nn_model_layer_2(layer_1_out)
        
        if return_all_layers:
            return output, layer_1_out, layer_0_out
        
        return output

    def predict(self, test_loader):
        self.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.double().to(self.device ), y.long().to(self.device )
                logits = self.forward(x)
                y_pred.extend(logits.argmax(1).cpu().numpy())
                y_true.extend(y.cpu().numpy())
        print(f"\n**************************************\nModel: {self.model_name}\n----------")
        evaluate_predictions(y_true, y_pred)





class TorchAlarm(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(input_size, 112),
            torch.nn.ReLU(),
            torch.nn.Linear(112, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 77),
            torch.nn.ReLU(),
            torch.nn.Linear(77, 1),
        ])

    def __call__(self, x, training=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        for layer in self.layers:
            x = layer(x)
        return x




class AMalwareDetectionDLA(BaseModel):  # Change inheritance to BaseModel
    def __init__(self, md_nn_model, ratio=0.95, device='cpu', model_name='DLA'):
        super().__init__(model_name, device)  # Call the BaseModel constructor       
        self.input_size = 10000
        self.n_classes = 2
        self.ratio = ratio

        self.md_nn_model = md_nn_model
        self.is_fitting_md_model = False  # the model is trained by default

        self.alarm_nn_model = TorchAlarm(input_size=400)

        self.tau = nn.Parameter(torch.zeros([1, ], device=self.device), requires_grad=False)

    def forward(self, x):
        logits, penultimate, layer_0_out = self.md_nn_model.forward(x, return_all_layers=True)
        hidden_activations = torch.cat([layer_0_out, penultimate], dim=-1)
        x_prob = self.alarm_nn_model(hidden_activations).reshape(-1)
        return logits, x_prob
    


class AMalwareDetectionDNNPlus(BaseModel):
    def __init__(self, md_nn_model, ratio=0.95, device='cpu', model_name='DNNPlus'):
        super().__init__(model_name, device) 
        self.input_size = 10000
        self.n_classes = 2
        self.ratio = ratio


        self.md_nn_model = md_nn_model
        self.is_fitting_md_model = False  # the model is trained by default

        self.amd_nn_plus = MalwareDetectionDNN(self.model_name,
                                               self.input_size,
                                               self.n_classes + 1,
                                               self.device)

        self.tau = nn.Parameter(torch.zeros([1, ], device=self.device), requires_grad=False)


    def forward(self, x):
        logits = self.amd_nn_plus(x)
        logits -= torch.amax(logits, dim=-1, keepdim=True).detach()  # increasing the stability, which might be helpful
        return logits, torch.softmax(logits, dim=-1)[:, -1]


class AdvMalwareDetectorICNN(BaseModel):
    def __init__(self, md_nn_model, ratio=0.95, device='cpu', model_name='ICNN' ):
        super().__init__(model_name, device) 
        self.md_nn_model = md_nn_model
        self.input_size = 10000
        self.n_classes = 2
        self.ratio = ratio
        self.dense_hidden_units = [200, 200]


        if hasattr(self.md_nn_model, 'smooth'):
            if not self.md_nn_model.smooth:  # non-smooth, exchange it
                for name, child in self.md_nn_model.named_children():
                    if isinstance(child, nn.ReLU):
                        self.md_nn_model._modules['relu'] = nn.SELU()
        else:
            for name, child in self.md_nn_model.named_children():
                if isinstance(child, nn.ReLU):
                    self.md_nn_model._modules['relu'] = nn.SELU()
        self.md_nn_model = self.md_nn_model.to(self.device)

        # input convex neural network
        self.non_neg_dense_layers = []
        for i in range(len(self.dense_hidden_units[0:-1])):
            self.non_neg_dense_layers.append(nn.Linear(self.dense_hidden_units[i],  # start from idx=1
                                                       self.dense_hidden_units[i + 1],
                                                       bias=False))
        self.non_neg_dense_layers.append(nn.Linear(self.dense_hidden_units[-1], 1, bias=False))
        # registration
        for idx_i, dense_layer in enumerate(self.non_neg_dense_layers):
            self.add_module('non_neg_layer_{}'.format(idx_i), dense_layer)

        self.dense_layers = []
        self.dense_layers.append(nn.Linear(self.input_size, self.dense_hidden_units[0]))
        for i in range(len(self.dense_hidden_units[1:])):
            self.dense_layers.append(nn.Linear(self.input_size, self.dense_hidden_units[i]))
        self.dense_layers.append(nn.Linear(self.input_size, 1))
        # registration
        for idx_i, dense_layer in enumerate(self.dense_layers):
            self.add_module('layer_{}'.format(idx_i), dense_layer)

        self.tau = nn.Parameter(torch.zeros([1, ], device=self.device), requires_grad=False)


    def forward_f(self, x):
        return self.md_nn_model(x)

    def forward_g(self, x):
        prev_x = None
        for i, dense_layer in enumerate(self.dense_layers):
            x_add = []
            x1 = dense_layer(x)
            x_add.append(x1)
            if prev_x is not None:
                x2 = self.non_neg_dense_layers[i - 1](prev_x)
                x_add.append(x2)
            prev_x = torch.sum(torch.stack(x_add, dim=0), dim=0)
            if i < len(self.dense_layers):
                prev_x = F.selu(prev_x)
        return prev_x.reshape(-1)

    def forward(self, x):
        return self.forward_f(x), self.forward_g(x)




class KernelDensityEstimation(BaseModel):
    def __init__(self, model, n_centers=1000, bandwidth=16., n_classes=2, ratio=0.95, model_name='KDE', device='cpu'):
        super().__init__(model_name, device)
        self.model = model.to(device)
        self.n_centers = n_centers
        self.bandwidth = bandwidth
        self.n_classes = n_classes
        self.ratio = ratio
        self.gaussian_means = None
        self.tau = nn.Parameter(torch.zeros(self.n_classes, device=device), requires_grad=False)
    
    def forward(self, x):
        logits, penultimate, _ = self.model.forward(x, return_all_layers=True)
        x_prob = self.forward_g(penultimate, logits.argmax(1).detach())
        return logits, x_prob
    
    def forward_f(self, x):
        logits, penultimate, _ = self.model.forward(x, return_all_layers=True)
        return logits, penultimate

    def forward_g(self, x_hidden, y_pred):
        dist = [torch.sum((means.unsqueeze(0) - x_hidden.unsqueeze(1))**2, dim=-1) for means in self.gaussian_means]
        kd = torch.stack([torch.mean(torch.exp(-d / self.bandwidth ** 2), dim=-1) for d in dist], dim=1)
        return -1 * kd[torch.arange(x_hidden.size(0)), y_pred]

    def get_threshold(self, validation_data, ratio=None):
        ratio = ratio if ratio is not None else self.ratio
        assert 0 <= ratio <= 1
        self.eval()
        probabilities, gt_labels = [], []
        with torch.no_grad():
            for x_val, y_val in validation_data:
                x_val, y_val = x_val.double().to(self.device ), y_val.long().to(self.device )
                _, x_prob = self.forward(x_val)
                probabilities.append(x_prob)
                gt_labels.append(y_val)
            prob = torch.cat(probabilities, dim=0)
            gt_labels = torch.cat(gt_labels)
            for i in range(self.n_classes):
                sorted_probs, _ = torch.sort(prob[gt_labels == i])
                self.tau[i] = sorted_probs[int((sorted_probs.shape[0] - 1) * ratio)]
    

    def get_tau_sample_wise(self, y_pred):
        return self.tau[y_pred]
    
    def indicator(self, x_prob, y_pred):
        return x_prob <= self.get_tau_sample_wise(y_pred) 

    def load(self, root_path):
        model_path = os.path.join(root_path, f'{self.model_name}.pth')
        ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
        self.gaussian_means = ckpt['gaussian_means']
        self.tau = ckpt['tau']
        self.model.load_state_dict(ckpt['base_model'])






