import torch.nn as nn


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
    
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)

class OutputHead(nn.Module):
    def __init__(self, in_dim, n_hid=200, out_dim=1):
        super(OutputHead, self).__init__()
        
        self.out = nn.Sequential()
        self.out.add_module('Flatten', Flatten())
        #self.out.add_module('Dropout', nn.Dropout(p=0.3))
        self.out.add_module('LinearClassifier', nn.Linear(in_dim, n_hid))
        # self.out.add_module('ReLU', nn.ReLU(inplace=True))
        
        #self.out.add_module('Dropout', nn.Dropout(p=0.3))
        self.out.add_module('LinearClassifier2', nn.Linear(n_hid, out_dim))
        self.initilize()

        self.softmax = nn.Softmax(dim=0)

    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.0)

    def forward(self, x):
        out = self.out(x)
        out = nn.functional.normalize(out, dim=1)
        return out
        # return self.softmax(out) # For DeepHit
    

class DualHeadedProjection(nn.Module):
    def __init__(self, in_dim, n_hid=200, proj_dim=128):
        super(DualHeadedProjection, self).__init__()

        self.flatten = Flatten()
        self.emb = nn.Linear(in_dim, n_hid)
        self.hazard = nn.Linear(n_hid, 1)
        self.proj = nn.Linear(n_hid, proj_dim)

    def forward(self, x):
        x = self.emb(self.flatten(x))
        h = self.hazard(x)
        proj = self.proj(x)

        return h, proj

class FullModel(nn.Module):
    def __init__(self, image_embedding_model, head=None):
        super(FullModel, self).__init__()
        self.image_embedding_model = image_embedding_model
        self.head = head

    def forward(self, input_image_variable, return_type='hazard_ratio'):
        assert return_type in ['hazard_ratio', 'features'], f'Expected param "return_type" to be one of [hazard_ratio, features], got {return_type}'

        image_embedding = self.image_embedding_model(input_image_variable)

        if return_type == 'features':
            logit_res = image_embedding
        elif return_type == 'hazard_ratio':
            logit_res = self.head(image_embedding)

        return logit_res
