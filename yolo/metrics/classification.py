class MSELoss(nn.Module):
    def __init__(
        self, 
        weight=None,  
        size_average=None, 
        ignore_index=-100, 
        reduce=None, 
        reduction='mean'
    ):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, inputs, targets, mask):
        pos_id = (mask==1.0).float()
        neg_id = (mask==0.0).float()
        pos_loss = pos_id * (inputs - targets) ** 2
        neg_loss = neg_id * (inputs) ** 2
        
        if self.reduction == 'mean':
            pos_loss = torch.mean(torch.sum(pos_loss, 1))
            neg_loss = torch.mean(torch.sum(neg_loss, 1))
            return pos_loss, neg_loss
        else:
            return pos_loss, neg_loss