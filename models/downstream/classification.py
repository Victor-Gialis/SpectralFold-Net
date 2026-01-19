import torch
import torch.nn as nn

from sklearn.preprocessing import LabelBinarizer

class LinearClassificationHead(nn.Module):
    def __init__(self, 
                 input_dim=512, 
                 n_classes=4, 
                 device:torch.device="cpu"):
        
        super().__init__()
        self.fc = nn.Linear(input_dim, n_classes)

        self.loss_function = torch.nn.CrossEntropyLoss()
        self.lb = LabelBinarizer()
        self.device = device

    def fit_labelizer(self,labels):
        self.lb.fit(labels)

    def transform_labels(self, labels):
        targets = torch.tensor(self.lb.transform(labels))
        targets = targets.float()
        targets = targets.to(self.device)
        return targets

    def forward(self, x):
        return self.fc(x)
    
    def compute_loss(self, outputs, inputs):
        y_label = inputs['y_label']

        targets = self.transform_labels(y_label)
        predictions = outputs

        return self.loss_function(predictions, targets)

    
class MLPClassificationHead(nn.Module):
    def __init__(self, 
                 in_dim=512, 
                 n_classes=4, 
                 dropout:float=0.5,
                 device:torch.device="cpu"):
        
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim//2),
            nn.ReLU(),
            nn.Linear(in_dim//2, n_classes),
            nn.Dropout(dropout)
        )

        self.loss_function = torch.nn.CrossEntropyLoss()
        self.lb = LabelBinarizer()
        self.device = device

    def fit_labelizer(self,labels):
        self.lb.fit(labels)
    
    def transform_labels(self, labels):
        targets = torch.tensor(self.lb.transform(labels))
        targets = targets.float()
        targets = targets.to(self.device)
        return targets

    def forward(self, x):
        return self.fc(x)
    
    def compute_loss(self, outputs, inputs):
        y_label = inputs['y_label']

        targets = self.transform_labels(y_label)
        predictions = outputs

        return self.loss_function(predictions, targets)