import torch
import torch.nn as nn
import math

from lib.pointops.functions import pointops


class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16, pos_enc='relative'):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.pos_enc = pos_enc.lower()

        assert self.pos_enc in ['relative', 'absolute', 'none', 'magnitude'], "pos_enc must be -> 'relative', 'absolute', 'magnitude' or 'none'"

        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)

        if self.pos_enc != 'none':
            self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes))

        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
                                    nn.Linear(mid_planes, mid_planes // share_planes),
                                    nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
                                    nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)  # (n, c)

        if self.pos_enc == 'relative':
            x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)  # (n, nsample, 3+c)
            x_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)  # (n, nsample, c)
            p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        elif self.pos_enc == 'magnitude':
            x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)
            x_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)
            p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
            p_r = torch.abs(p_r) # directional vector
        elif self.pos_enc == 'absolute':
            x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=False)
            x_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)
            # We group the coordinates p directly as features to get absolute neighbor coordinates p_j
            p_r = pointops.queryandgroup(self.nsample, p, p, p, None, o, o, use_xyz=False) # (n, nsample, 3)
        else: # 'none'
            x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=False)
            x_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)

        if self.pos_enc != 'none':
            for i, layer in enumerate(self.linear_p): p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)    # (n, nsample, c)
            w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes, self.mid_planes).sum(2)  # (n, nsample, c)
        else:
            w = x_k - x_q.unsqueeze(1)

        for i, layer in enumerate(self.linear_w): w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
        w = self.softmax(w)  # (n, nsample, c)
        n, nsample, c = x_v.shape; s = self.share_planes

        if self.pos_enc != 'none':
            x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
        else:
            x = (x_v.view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)

        return x


class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3+in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i-1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            idx = pointops.furthestsampling(p, o, n_o)  # (m)
            n_p = p[idx.long(), :]  # (m, 3)
            x = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True)  # (m, 3+c, nsample)
            x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous()))  # (m, c, nsample)
            x = self.pool(x).squeeze(-1)  # (m, c)
            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))  # (n, c)
        return [p, x, o]


class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(nn.Linear(2*in_planes, in_planes), nn.BatchNorm1d(in_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))

    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i-1], o[i], o[i] - o[i-1]
                x_b = x[s_i:e_i, :]
                x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
        else:
            p1, x1, o1 = pxo1; p2, x2, o2 = pxo2
            x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
        return x


class PointTransformerLayerMLP(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_planes, out_planes),
            nn.BatchNorm1d(out_planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, pxo):
        p, x, o = pxo
        return self.mlp(x)


class PointTransformerLayerMLPPooling(nn.Module):
    def __init__(self, in_planes, out_planes, nsample=16):
        super().__init__()
        self.nsample = nsample
        self.mlp = nn.Sequential(
            nn.Linear(in_planes, out_planes),
            nn.BatchNorm1d(out_planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, pxo):
        p, x, o = pxo
        x = self.mlp(x)

        x_grouped = pointops.queryandgroup(self.nsample, p, p, x, None, o, o, use_xyz=False) # (n, nsample, c)
        x_pooled = x_grouped.max(dim=1)[0] # (n, c)

        return x_pooled


class PointTransformerLayerScalar(nn.Module):
    def __init__(self, in_planes, out_planes, nsample=16):
        super().__init__()
        self.out_planes = out_planes
        self.nsample = nsample

        self.linear_q = nn.Linear(in_planes, out_planes)
        self.linear_k = nn.Linear(in_planes, out_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo):
        p, x, o = pxo
        x_q = self.linear_q(x) # (n, c)
        x_k = self.linear_k(x) # (n, c)
        x_v = self.linear_v(x) # (n, c)

        # Group K and V based on query points
        x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=False) # (n, nsample, c)
        x_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False) # (n, nsample, c)

        # Inject positional encoding into the attention logits if required
        p_r = pointops.queryandgroup(self.nsample, p, p, p, None, o, o, use_xyz=True)[:, :, :3]
        for i, layer in enumerate(self.linear_p):
            p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)

        x_k = x_k + p_r
        x_v = x_v + p_r

        # Calculate standard dot product: Q * K^T
        # We unsqueeze Q to (n, 1, c) to broadcast multiply with K (n, nsample, c), then sum over channels
        attn = (x_q.unsqueeze(1) * x_k).sum(dim=-1, keepdim=True) / math.sqrt(self.out_planes) # (n, nsample, 1)

        # Apply softmax over the nsample dimension
        attn_weights = self.softmax(attn) # (n, nsample, 1)

        # Aggregate V using scalar attention weights
        x_out = (x_v * attn_weights).sum(dim=1) # (n, c)

        return x_out



class PointTransformerBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16, pos_enc='relative', attn_type='vector'):
        super(PointTransformerBlock, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)

        # layer based on ablation type
        if attn_type == 'vector':
            self.transformer2 = PointTransformerLayer(planes, planes, share_planes, nsample, pos_enc)
        elif attn_type == 'scalar':
            self.transformer2 = PointTransformerLayerScalar(planes, planes, nsample)
        elif attn_type == 'mlp_pooling':
            self.transformer2 = PointTransformerLayerMLPPooling(planes, planes, nsample)
        elif attn_type == 'mlp':
            self.transformer2 = PointTransformerLayerMLP(planes, planes)
        else:
            raise ValueError(f"Unknown attn_type: {attn_type}")

        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer2([p, x, o])))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]


class PointTransformerSeg(nn.Module):
    def __init__(self, block, blocks, c=6, k=13):
        super().__init__()
        self.c = c
        self.in_planes, planes = c, [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]
        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])  # N/1
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])  # N/4
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])  # N/16
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])  # N/64
        self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes, stride=stride[4], nsample=nsample[4])  # N/256
        self.dec5 = self._make_dec(block, planes[4], 2, share_planes, nsample=nsample[4], is_head=True)  # transform p5
        self.dec4 = self._make_dec(block, planes[3], 2, share_planes, nsample=nsample[3])  # fusion p5 and p4
        self.dec3 = self._make_dec(block, planes[2], 2, share_planes, nsample=nsample[2])  # fusion p4 and p3
        self.dec2 = self._make_dec(block, planes[1], 2, share_planes, nsample=nsample[1])  # fusion p3 and p2
        self.dec1 = self._make_dec(block, planes[0], 2, share_planes, nsample=nsample[0])  # fusion p2 and p1
        self.cls = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], k))

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def forward(self, pxo):
        p0, x0, o0 = pxo  # (n, 3), (n, c), (b)
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        x = self.cls(x1)
        return x


def pointtransformer_seg_repro(**kwargs):
    model = PointTransformerSeg(PointTransformerBlock, [2, 3, 4, 6, 3], **kwargs)
    return model


class PointTransformerCls(nn.Module):
    def __init__(self, block, blocks, c=6, k=40, num_neighbors_k=8, pos_enc='relative', attn_type='vector'):
        super().__init__()
        self.c = c
        # The encoder feature dimensions progressively scale up
        self.pos_enc = pos_enc
        self.attn_type = attn_type
        self.in_planes, planes = c, [32, 64, 128, 256, 512]
        share_planes = 8
        stride = [1, 4, 4, 4, 4]
        nsample = [num_neighbors_k] * 5 # [8, 16, 16, 16, 16]

        # Encoder stages (identical to segmentation)
        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])  # N/1
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])  # N/4
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])  # N/16
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])  # N/64
        self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes, stride=stride[4], nsample=nsample[4])  # N/256

        # Classification head: MLP applied after Global Average Pooling
        self.cls = nn.Sequential(
            nn.Linear(planes[4], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, k)
        )

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample, pos_enc=self.pos_enc, attn_type=self.attn_type))
        return nn.Sequential(*layers)

    def forward(self, pxo):
        p0, x0, o0 = pxo  # (n, 3), (n, c), (b)
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)

        # Pass through the encoder stages
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])

        # Global Average Pooling (handle continuous tensor grouped by batch offsets)
        x_pool = []
        for i in range(o5.shape[0]):
            s_i = 0 if i == 0 else o5[i-1]
            e_i = o5[i]
            x_pool.append(x5[s_i:e_i].mean(0))
        x_pool = torch.stack(x_pool, dim=0)  # Yields (b, 512)

        # Output classification logits
        x = self.cls(x_pool)
        return x


def pointtransformer_cls(num_neighbors_k=8, pos_enc='relative', attn_type='vector', **kwargs):
    # Uses the same residual block configuration as the segmentation model [2, 3, 4, 6, 3]
    print(f"Selected number of neighbors k: {num_neighbors_k}")
    print(f"Selected positional encoding strategy: {pos_enc}")
    print(f"Selected layer attention type: {attn_type}")
    model = PointTransformerCls(PointTransformerBlock, [2, 3, 4, 6, 3], num_neighbors_k=num_neighbors_k, pos_enc=pos_enc, attn_type=attn_type, **kwargs)
    return model