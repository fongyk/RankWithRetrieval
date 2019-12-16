class APLoss(nn.Module):
    '''
    reference:
    https://github.com/almazan/deep-image-retrieval/blob/master/dirtorch/loss.py
    Learning with Average Precision: Training Image Retrieval with a Listwise Loss.
    '''
    def __init__(self, bin_num=20, range_left=0.0, range_right=1.0):
        super(APLoss, self).__init__()
        self.bin_num = bin_num
        self.min = range_left
        self.max = range_right
        self.delta = (self.max - self.min) / (self.bin_num - 1)
        k = 1.0 / self.delta

        quantizer = nn.Conv1d(1, 2*self.bin_num, kernel_size=1, bias=True)
        quantizer.weight = nn.Parameter(quantizer.weight.detach(), requires_grad=False)
        quantizer.bias = nn.Parameter(quantizer.bias.detach(), requires_grad=False)
        ## left bin
        quantizer.weight[:self.bin_num] = k
        quantizer.bias[:self.bin_num] = torch.from_numpy(np.arange(2-self.bin_num, 2) - k*self.min)
        ## right bin
        quantizer.weight[self.bin_num:] = -k
        quantizer.bias[self.bin_num:] = torch.from_numpy(np.arange(self.bin_num, 0, -1) + k*self.min)
        ## edge bin
        quantizer.weight[0] = quantizer.weight[-1] = 0
        quantizer.bias[0] = quantizer.bias[-1] = 1

        self.quantizer = quantizer

    def forward(self, similarity, label):
        '''
        batch size: N (a large N is prefered to involve more positive samples.)
        similarity: N * N
        label: N
        '''
        N = similarity.size(0)
        similarity = similarity.unsqueeze(1) # N * 1 * N
        label = label.view(1, -1)
        label = (label == label.transpose(0, 1)).float() # N * N

        quan_sim = self.quantizer(similarity) # N * 2M * N, M = bin_num
        quan_sim = torch.min(quan_sim[:, :self.bin_num], quan_sim[:, self.bin_num:]).clamp(min=0) # N * M * N

        quan_sum = quan_sim.sum(dim=-1) # N * M
        quan_target = (quan_sim * label.view(N, 1, N)).sum(dim=-1) # N * M
        prec = quan_target.cumsum(dim=-1) / (1e-8 + quan_sum.cumsum(dim=-1)) # N * M
        delta_rec = quan_target / label.sum(dim=-1, keepdims=True) # N * M
        ap = (prec * delta_rec).sum(dim=-1) # N

        return 1 - ap.mean()
