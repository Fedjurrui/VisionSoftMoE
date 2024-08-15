from typing import Any

import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.models.softmoe import ViTSoftMoE

class LightningVitSoftMoE(pl.LightningModule):
    def __init__(self,image_size, patch_size,
                 num_classes,
                 dim, depth, heads,
                 num_experts, num_slots, num_tokens,
                 channels=3, dim_head=64,
                 learning_rate=0.001,
                 warmup_steps=10000):
        super(LightningVitSoftMoE,self).__init__()
        self.save_hyperparameters()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.num_experts = num_experts
        self.num_slots = num_slots
        self.num_tokens = num_tokens
        self.channels = channels
        self.dim_head = dim_head
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.model = ViTSoftMoE(image_size=self.image_size,patch_size=self.patch_size,
                                num_classes=self.num_classes,dim=self.dim,depth=self.depth,
                                heads=self.heads,num_experts=self.num_experts,num_slots=self.num_slots,
                                num_tokens=self.num_tokens,channels=self.channels,dim_head=self.dim_head)
        self.automatic_optimization = False
        self.loss = nn.CrossEntropyLoss()
    def forward(self,x):
        return self.model(x)
    def training_step(self,batch,batch_idx):
        opt = self.optimizers()
        #sch = self.lr_schedulers()
        x,y = batch
        logits = self(x)
        opt.zero_grad()
        loss = self.loss(logits,y)
        self.manual_backward(loss,retain_graph=True)
        opt.step()
        #sch.step()

        self.log("train_loss",loss)
        return loss
    def validation_step(self,batch,batch_idx):
        # log val loss and calculate accuracy
        x,y = batch
        logits = self(x)
        loss = self.loss(logits,y)
        preds = torch.argmax(logits,dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss",loss,prog_bar=False)
        self.log("val_acc",acc)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=self.learning_rate)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=self.warmup_steps,T_mult=2)
        return [optimizer]#,[scheduler]




