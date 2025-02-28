import torch
import torch.nn as nn
import pytorch_lightning as pl
#from torch.optim.lr_scheduler import ReduceLROnPlateau
#from torchmetrics.classification import BinaryAccuracy
#import torchmetrics

class MLP_binary(pl.LightningModule):
    def __init__(self):
        super(MLP_binary, self).__init__()
        self.fc1 = nn.Linear(1280, 700)
        self.bn1 = nn.BatchNorm1d(700)
        self.dropout1 = nn.Dropout(0.3659)
        
        self.fc2 = nn.Linear(700, 350)
        self.bn2 = nn.BatchNorm1d(350)
        self.dropout2 = nn.Dropout(0.3659)
        
        self.fc3 = nn.Linear(350, 1)
        
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # Skip BatchNorm if batch size is 1
        if x.shape[0] > 1:
            x = self.bn1(x)
        x = self.dropout1(x)

        x = torch.relu(self.fc2(x))
        if x.shape[0] > 1:
            x = self.bn2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x

class MLP_multi_output(pl.LightningModule):
    def __init__(self, num_classes_level1, num_classes_level2, num_classes_level3, num_classes_level4):
        super(MLP_multi_output, self).__init__()

        self.fc1 = nn.Linear(1280, 750)
        self.bn1 = nn.BatchNorm1d(750)
        self.dropout1 = nn.Dropout(0.32)
        
        self.fc2 = nn.Linear(750, 400)
        self.bn2 = nn.BatchNorm1d(400)
        self.dropout2 = nn.Dropout(0.32)
        
        self.fc_level1 = nn.Linear(400, num_classes_level1)
        self.fc_level2 = nn.Linear(400, num_classes_level2)
        self.fc_level3 = nn.Linear(400, num_classes_level3)
        self.fc_level4 = nn.Linear(400, num_classes_level4)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)

        x = torch.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)

        level1_out = self.fc_level1(x)
        level2_out = self.fc_level2(x)
        level3_out = self.fc_level3(x)
        level4_out = self.fc_level4(x)

        return level1_out, level2_out, level3_out, level4_out
    
class MLP_family(pl.LightningModule):
    def __init__(self, num_classes_level3):
        super(MLP_family, self).__init__()

        self.fc1 = nn.Linear(1280, 720)
        self.bn1 = nn.BatchNorm1d(720)
        self.dropout1 = nn.Dropout(0.23)
        
        self.fc2 = nn.Linear(720, 360)
        self.bn2 = nn.BatchNorm1d(360)
        self.dropout2 = nn.Dropout(0.23)
        
        self.fc_level3 = nn.Linear(360, num_classes_level3)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)

        x = torch.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)

        level3_out = self.fc_level3(x)
        return level3_out
    
class MLP_subfamily(pl.LightningModule):
    def __init__(self, num_classes_level4):
        super(MLP_subfamily, self).__init__()

        self.fc1 = nn.Linear(1280, 800)
        self.bn1 = nn.BatchNorm1d(800)
        self.dropout1 = nn.Dropout(0.33)
        
        self.fc2 = nn.Linear(800, 400)
        self.bn2 = nn.BatchNorm1d(400)
        self.dropout2 = nn.Dropout(0.33)
        
        self.fc_level4 = nn.Linear(400, num_classes_level4)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)

        x = torch.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)

        level4_out = self.fc_level4(x)
        return level4_out
    
class MLP_substrate(pl.LightningModule):
    def __init__(self, num_classes_level1,):
        super(MLP_substrate, self).__init__()

        self.fc1 = nn.Linear(1280, 720)
        self.bn1 = nn.BatchNorm1d(720)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(720, 360)
        self.bn2 = nn.BatchNorm1d(360)
        self.dropout2 = nn.Dropout(0.15)
        
        self.fc_level1 = nn.Linear(360, num_classes_level1)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)

        x = torch.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)

        level1_out = self.fc_level1(x)
        return level1_out

    def training_step(self, batch, batch_idx):
        x, y1 = batch
        y_hat1 = self(x)

        loss = self.criterion(y_hat1, y1)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)

        return loss