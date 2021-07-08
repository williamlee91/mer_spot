import torch.nn as nn
import torch
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


############### Postbackbone ##############
class BaseFeatureNet(nn.Module):
    '''
    Calculate basic feature
    PreBackbobn -> Backbone
    CAS(ME)^2:
    input: [batch_size, 2048, 64]
    output: [batch_size, 512, 16]
    SAMM:
    input: [batch_size, 2048, 256]
    output: [batch_size, 512, 64]
    '''
    def __init__(self, cfg):
        super(BaseFeatureNet, self).__init__()
        self.dataset = cfg.DATASET.DATASET_NAME
        self.conv1 = nn.Conv1d(in_channels=cfg.MODEL.IN_FEAT_DIM,
                               out_channels=cfg.MODEL.BASE_FEAT_DIM,
                               kernel_size=9, stride=1, padding=4, bias=True)
        self.conv2 = nn.Conv1d(in_channels=cfg.MODEL.BASE_FEAT_DIM,
                               out_channels=cfg.MODEL.BASE_FEAT_DIM,
                               kernel_size=9, stride=1, padding=4, bias=True)      
        self.max_pooling = nn.MaxPool1d(kernel_size=2, stride=2)
        self.mish = Mish()
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.mish(self.conv1(x))
        # feat = self.relu(self.conv1(x))
        feat = self.max_pooling(feat)
        feat = self.mish(self.conv2(feat))
        # feat = self.relu(self.conv2(feat))
        feat = self.max_pooling(feat)
        return feat


############### Neck ##############
class FeatNet(nn.Module):
    '''
    Main network
    Backbone -> Neck
    CAS(ME)^2:
    input: base feature, [batch_size, 512, 16]
    output: MAL1, MAL2, MAL3, MAL4
    MAL1: [batch_size, 512, 16]
    MAL2: [batch_size, 512, 8]
    MAL3: [batch_size, 1024, 4]
    MAL4: [batch_size, 1024, 2]
    SAMM:
    input: base feature, [batch_size, 512, 128]
    output: MAL1, MAL2, MAL3, MAL4, MAL5, MAL6, MAL7
    MAL1: [batch_size, 1024, 32]
    MAL2: [batch_size, 1024, 16]
    MAL3: [batch_size, 1024, 8]
    MAL4: [batch_size, 1024, 4]
    MAL5: [batch_size, 1024, 2]
    '''
    def __init__(self, cfg):
        super(FeatNet, self).__init__()
        self.base_feature_net = BaseFeatureNet(cfg)
        self.convs = nn.ModuleList()
        for layer in range(cfg.MODEL.NUM_LAYERS):
            # stride = 1 if layer == 0 else 2
            in_channel = cfg.MODEL.BASE_FEAT_DIM if layer == 0 else cfg.MODEL.LAYER_DIMS[layer - 1]
            out_channel = cfg.MODEL.LAYER_DIMS[layer]
            conv = nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=cfg.MODEL.LAYER_STRIDES[layer], padding=1)
            self.convs.append(conv)
        # self.relu = nn.ReLU(inplace=True)
        self.mish = Mish()

    def forward(self, x):
        results = []
        feat = self.base_feature_net(x)
        for conv in self.convs:
            feat = self.mish(conv(feat))
            # feat = self.relu(conv(feat))
            results.append(feat)

        return tuple(results)


# Postbackbone -> Neck
class GlobalLocalBlock(nn.Module):
    def __init__(self, cfg):
        super(GlobalLocalBlock, self).__init__()

        self.dim_in = cfg.MODEL.BASE_FEAT_DIM
        self.dim_out = cfg.MODEL.BASE_FEAT_DIM
        self.ws = cfg.DATASET.WINDOW_SIZE
        self.drop_threshold = cfg.MODEL.DROP_THRESHOLD
        self.ss = cfg.DATASET.SAMPLE_STRIDE
        self.mish = Mish()
        
        self.down = nn.Conv1d(self.dim_in, self.dim_out//2, kernel_size=1, stride=1)
        
        self.theta = nn.Conv1d(self.dim_in//2, self.dim_out//2, kernel_size=1, stride=1)
        self.phi = nn.Conv1d(self.dim_in//2, self.dim_out//2, kernel_size=1, stride=1)
        self.g = nn.Conv1d(self.dim_in//2, self.dim_out//2, kernel_size=1, stride=1)
        
        #Fuse
        self.lcoal_global = nn.Conv1d(self.dim_out//2, self.dim_out//2, kernel_size=1, stride=1)

        # MLP
        self.drop = nn.Dropout(p=0.3)
        self.conv1 = nn.Conv1d(self.dim_out//2, 4*self.dim_out//2, kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(4*self.dim_out//2, self.dim_out//2, kernel_size=1, stride=1)

        self.up = nn.Conv1d(self.dim_out//2, self.dim_out, kernel_size=1, stride=1)
    
    def forward(self, x):
        
        x= self.down(x)
        residual = x

        batch_size = x.shape[0]
        channels = x.shape[1]
        ori_length = x.shape[2]
        
        length_temp = self.ws //(self.ss*ori_length)

        theta = self.theta(x)
        phi = self.phi(x)
        g = self.g(x)

        all_tmp = torch.zeros([channels, batch_size, length_temp, ori_length]).cuda()
        all_temp_g = all_tmp
        for j in range(theta.size(1)):
            # Sometimes, temp1: BS* T * Channels
            # temp2: BS* (T+1) * Channels
            temp = torch.zeros([batch_size, length_temp, ori_length]).cuda() # BS*T*mean_channel
            temp_g = temp
            if j < length_temp//2:
                temp[:,length_temp//2-j:,:] = theta[:,:j+length_temp//2,:]
                temp_g[:,length_temp//2-j:,:] = g[:,:j+length_temp//2,:]
            elif length_temp//2 <= j <= theta.size(1)-length_temp//2:
                temp = theta[:,j-length_temp//2:j+length_temp//2,:]
                temp_g= g[:,j-length_temp//2:j+length_temp//2,:]
            else:
                temp[:,:length_temp-(j%length_temp-length_temp//2),:] = theta[:,j-length_temp//2:,:]
                temp_g[:,:length_temp-(j%length_temp-length_temp//2),:] = g[:,j-length_temp//2:,:]
            
            all_tmp[j:j+1,:,:,:]= temp
            all_temp_g[j:j+1,:,:,:] = temp_g

        all_tmp_phi = phi.unsqueeze(dim=2)
        local_theta_phi = torch.matmul(all_tmp_phi, all_tmp.permute(1,0,3,2))

        local_theta_phi_sc = local_theta_phi * (channels**-.5)
        local_p = F.softmax(local_theta_phi_sc, dim=-1)  
        local_p = local_p.expand(-1, -1, ori_length, -1)
        all_temp_g = all_temp_g.permute(1,0,3,2)
        all_temp_g = torch.where(all_temp_g < torch.tensor(self.drop_threshold).float().cuda(), torch.tensor(0).float().cuda(), all_temp_g)
        local_temp = torch.sum(self.drop(local_p) * all_temp_g, dim=-1) 
        out_temp = local_temp
    
        # global temporal encoder
        # e.g. (BS, 512, 16) * (BS, 16, 512) => (BS, 1024, 1024)
        # global_theta_phi = torch.bmm(phi, torch.transpose(theta,2,1))
        # global_theta_phi_sc = global_theta_phi * (channels**-.5)
        # global_p = F.softmax(global_theta_phi_sc, dim=-1)
        # global_temp = torch.bmm(self.drop(global_p), g)

        # out_temp = torch.cat((local_temp, global_temp), dim=1)

        # MLP
        local_global = self.lcoal_global(out_temp)
        out_temp_ln = F.layer_norm(self.drop(local_global)+residual,[channels, ori_length])
        
        out_mlp_conv1 = self.conv1(out_temp_ln)
        out_mlp_act = self.mish(self.drop(out_mlp_conv1))
        out_mlp_conv2 = self.conv2(out_mlp_act)
        out = F.layer_norm(self.drop(out_mlp_conv2) + out_temp_ln, [channels, ori_length])

        out = self.up(out)

        return out


############### Postneck ##############
class ReduceChannel(nn.Module):
    '''
    From FeatNet
    Neck -> Postneck
    CAS(ME)^2:
    input: from FeatNet
           MAL1: [batch_size, 512, 16]
           MAL2: [batch_size, 512, 8]
           MAL3: [batch_size, 1024, 4]
           MAL4: [batch_size, 1024, 2]
    output: All Level-features'Channels Reduced into 512
    SAMM:
    input: from FeatNet
           MAL1: [batch_size, 512, 128]
           MAL2: [batch_size, 512, 64]
           MAL3: [batch_size, 1024, 32]
           MAL4: [batch_size, 1024, 16]
           MAL5: [batch_size, 1024, 8]
           MAL6: [batch_size, 1024, 4]
           MAL7: [batch_size, 1024, 2]
    output: All Level-features'Channels Reduced into 512
    '''
    def __init__(self, cfg):
        super(ReduceChannel, self).__init__()
        self.convs = nn.ModuleList()
        for layer in range(cfg.MODEL.NUM_LAYERS):
            conv = nn.Conv1d(cfg.MODEL.LAYER_DIMS[layer], cfg.MODEL.REDU_CHA_DIM, kernel_size=1)
            self.convs.append(conv)
        # self.relu = nn.ReLU(inplace=True)
        self.mish = Mish()

    def forward(self, feat_list):
        assert len(feat_list) == len(self.convs)
        results = []
        for conv, feat in zip(self.convs, feat_list):
           results.append(self.mish(conv(feat)))
           # results.append(self.relu(conv(feat)))
        return tuple(results)


############### Head ##############
class PredHeadBranch(nn.Module):
    '''
    From ReduceChannel Module
    CAS(ME)^2:
    input: [batch_size, 512, (16,8,4,2)]
    output: Channels reduced into 256
    SAMM:
    input: [batch_size, 512, (128,64,32,16,8,4,2)]
    output: Channels reduced into 256
    '''
    def __init__(self, cfg):
        super(PredHeadBranch, self).__init__()
        self.head_stack_layers = cfg.MODEL.HEAD_LAYERS  # 2
        self._init_head(cfg)

    def _init_head(self, cfg):
        self.convs = nn.ModuleList()
        for layer in range(self.head_stack_layers):
            in_channel = cfg.MODEL.REDU_CHA_DIM if layer == 0 else cfg.MODEL.HEAD_DIM
            out_channel = cfg.MODEL.HEAD_DIM
            conv = nn.Conv1d(in_channel, out_channel, kernel_size=3, padding=1)
            self.convs.append(conv)
        self.mish = Mish()
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = x
        for conv in self.convs:
            feat = self.mish(conv(feat))
            # feat = self.relu(conv(feat))
        return feat


############### Details of Prediction ##############
class PredHead(nn.Module):
    '''
    CAS(ME)^2:
    input: [batch_size, 512, (16,8,4,2)]
    input_tmp: to PredHeadBranch Module
    output: Channels reduced into number of classes or boundaries
    SAMM:
    input: [batch_size, 512, (128,64,32,16,8,4,2)]
    input_tmp: to PredHeadBranch Module
    output: Channels reduced into number of classes or boundaries
    '''
    def __init__(self, cfg):
        super(PredHead, self).__init__()
        self.head_branches = nn.ModuleList()
        self.lgf= GlobalLocalBlock(cfg)
        self.inhibition = cfg.MODEL.INHIBITION_INTERVAL
        for _ in range(4):
            self.head_branches.append(PredHeadBranch(cfg))
        num_class = cfg.DATASET.NUM_CLASSES  # 2
        num_box = len(cfg.MODEL.ASPECT_RATIOS)  # 5

        # [batch_size, 256, (16,8,4,2)] -> [batch_size, _, (16,8,4,2)]
        af_cls = nn.Conv1d(cfg.MODEL.HEAD_DIM, num_class, kernel_size=3, padding=1)
        af_reg = nn.Conv1d(cfg.MODEL.HEAD_DIM, 2, kernel_size=3, padding=1)
        ab_cls = nn.Conv1d(cfg.MODEL.HEAD_DIM, num_box * num_class, kernel_size=3, padding=1)
        ab_reg = nn.Conv1d(cfg.MODEL.HEAD_DIM, num_box * 2, kernel_size=3, padding=1)
        self.pred_heads = nn.ModuleList([af_cls, af_reg, ab_cls, ab_reg])

    def forward(self, x):
        preds = []
        if x.size(-1) in self.inhibition:
            lgf_out = self.lgf(x)
        else:
            lgf_out = x
        for pred_branch, pred_head in zip(self.head_branches, self.pred_heads):
            feat = pred_branch(lgf_out)
            preds.append(pred_head(feat))

        return tuple(preds)


############### Prediction ##############
class LocNet(nn.Module):
    '''
    Predict action boundary, based on features from different FPN levels
    '''
    def __init__(self, cfg):
        super(LocNet, self).__init__()
        # self.features = FeatNet(cfg)
        self.reduce_channels = ReduceChannel(cfg)
        self.pred = PredHead(cfg)
        self.num_class = cfg.DATASET.NUM_CLASSES
        self.ab_pred_value = cfg.DATASET.NUM_CLASSES + 2

    def _layer_cal(self, feat_list):
        af_cls = list()
        af_reg = list()
        ab_pred = list()

        for feat in feat_list:
            cls_af, reg_af, cls_ab, reg_ab = self.pred(feat)
            af_cls.append(cls_af.permute(0, 2, 1).contiguous())
            af_reg.append(reg_af.permute(0, 2, 1).contiguous())
            ab_pred.append(self.tensor_view(cls_ab, reg_ab))

        af_cls = torch.cat(af_cls, dim=1)  # bs, sum(t_i), n_class+1
        af_reg = torch.cat(af_reg, dim=1)  # bs, sum(t_i), 2
        af_reg = F.relu(af_reg)

        return (af_cls, af_reg), tuple(ab_pred)

    def tensor_view(self, cls, reg):
        '''
        view the tensor for [batch, 120, depth] to [batch, (depth*5), 24]
        make the prediction (24 values) for each anchor box at the last dimension
        '''
        bs, c, t = cls.size()
        cls = cls.view(bs, -1, self.num_class, t).permute(0, 3, 1, 2).contiguous()
        reg = reg.view(bs, -1, 2, t).permute(0, 3, 1, 2).contiguous()
        data = torch.cat((cls, reg), dim=-1)
        data = data.view(bs, -1, self.ab_pred_value)
        return data

    def forward(self, features_list):
        features_list = self.reduce_channels(features_list)

        return self._layer_cal(features_list)


############### All processing ##############
class FuseModel(nn.Module):
    def __init__(self, cfg):
        super(FuseModel, self).__init__()
        self.features = FeatNet(cfg)
        self.loc_net = LocNet(cfg)

    def forward(self, x):
        features = self.features(x)
        out_af, out_ab = self.loc_net(features)
        return out_af, out_ab

if __name__ == '__main__':
    import sys
    sys.path.append('/home/yww/1_spot/MSA-Net/lib')
    from config import cfg, update_config
    cfg_file = '/home/yww/1_spot/MSA-Net/experiments/A2Net_thumos.yaml'
    update_config(cfg_file)

    model = FuseModel(cfg).cuda()
    data = torch.randn((8, 2048, 64)).cuda()
    output = model(data)
