import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import torchvision

# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            # 交叉熵
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [%s] is not found' % self.gan_type)

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp,
            grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss

####################
# Perceptual Network
####################
class ContextualLoss(nn.Module):
    def __init__(self, sigma=0.1, b=1.0, epsilon=1e-5, similarity='cos'):
        super(ContextualLoss, self).__init__()
        self.sigma = sigma
        self.similarity = similarity
        self.b = b
        self.e = epsilon

    def cos_similarity(self, image_features, target_features):
        # N, V, C
        # torch.view: return a new tensor with the same data at
        # the self tensor but of a different shape
        # tensor for pytorch: [batchsize, c, h, w]
        if_vec = image_features.view((image_features.size()[0], image_features.size()[1], -1)).permute(0, 2, 1)
        tf_vec = target_features.view((target_features.size()[0], target_features.size()[1], -1)).permute(0, 2, 1)
        # 转置后：(batchsize, n, channel)
        # Centre by T
        # tf_mean:uy
        # tf_mean是平均特征图，所以是dim=1
        tf_mean = torch.mean(tf_vec, dim=1, keepdim=True)
        ifc_vec = if_vec - tf_mean
        tfc_vec = tf_vec - tf_mean
        # L2-norm normalization
        # 每个特征图有多个通道。单层特征图的L2归一化，是特征图中的每个元素除以这个特征图的 L2范数，所以是dim=2
        ifc_vec_l2 = torch.div(ifc_vec, torch.sqrt(torch.sum(ifc_vec * ifc_vec, dim=2, keepdim=True)))
        tfc_vec_l2 = torch.div(tfc_vec, torch.sqrt(torch.sum(tfc_vec * tfc_vec, dim=2, keepdim=True)))
        # cross dot
        # 由余弦公式，这里是内积。每个特征图呈一行或一列。
        # 得到的feature_cos_similarity martix是一个矩阵,大小为[batchsize, nfeature, nfeature]
        # dim=1，列是j, dim=2，行是i
        # bmm: perfomrs a batch matrix-matrix product of matrics
        # xi指的是同一个像素，不同通道的特征
        feature_cos_similarity_matrix = 1 - torch.bmm(ifc_vec_l2, tfc_vec_l2.permute(0, 2, 1))
        return feature_cos_similarity_matrix

    def L2_similarity(self, image_features, target_features):
        pass

    # the second equaiton for CXij
    def relative_distances(self, feature_similarity_matrix):
        # 矩阵相乘，前一个是generated image(i)，后面的是ground truthI(j)
        # ij排列
        # 11 12 13
        # 21 22 23
        # 31 32 33
        relative_dist = feature_similarity_matrix / (torch.min(feature_similarity_matrix, dim=2, keepdim=True)[0] + self.e)
        return relative_dist

    # the later three equations for CXij
    def weighted_average_distances(self, relative_distances_matrix):
        weights_before_normalization = torch.exp((self.b - relative_distances_matrix) / self.sigma)
        weights_sum = torch.sum(weights_before_normalization, dim=2, keepdim=True)
        weights_normalized = torch.div(weights_before_normalization, weights_sum)
        return weights_normalized

    def CX(self, feature_similarity_matrix):
        feature_similarity_matrix_norm = self.relative_distances(feature_similarity_matrix)
        CX_i_j = self.weighted_average_distances(self.relative_distances(feature_similarity_matrix))
        CX_j_i = CX_i_j.permute(0, 2, 1)
        max_i_on_j = torch.max(CX_j_i, dim=1)[0]

        # max_i_on_j = 0.5*(max_i_on_j + torch.diag(CX_j_i[0]))

        CS = torch.mean(max_i_on_j, dim=1)
        CX = - torch.log(CS)
        CX_loss = torch.mean(CX)
        # return CX_loss,feature_similarity_matrix,feature_similarity_matrix_norm,CX_j_i
        return CX_loss

    def forward(self, image_features, target_features):
        if self.similarity == 'cos':
            feature_similarity_matrix = self.cos_similarity(image_features, target_features)
        elif self.similarity == 'l2':
            feature_similarity_matrix = self.L2_similarity(image_features, target_features)
        else:
            feature_similarity_matrix = self.cos_similarity(image_features, target_features)
        return self.CX(feature_similarity_matrix)

#
class CosSimilarLoss(nn.Module):
    def __init__(self, similarity='cos'):
        super(CosSimilarLoss, self).__init__()
        self.similarity = similarity

    def cos_similarity(self, image_features, target_features):
        # N, V, C
        if_vec = image_features.view((image_features.size()[0], image_features.size()[1], -1)).permute(0, 2, 1)
        tf_vec = target_features.view((target_features.size()[0], target_features.size()[1], -1)).permute(0, 2, 1)
        # Centre by T
        tf_mean = torch.mean(tf_vec, dim=1, keepdim=True)
        ifc_vec = if_vec - tf_mean
        tfc_vec = tf_vec - tf_mean
        # L2-norm normalization
        ifc_vec_l2 = torch.div(ifc_vec, torch.sqrt(torch.sum(ifc_vec * ifc_vec, dim=2, keepdim=True)))
        tfc_vec_l2 = torch.div(tfc_vec, torch.sqrt(torch.sum(tfc_vec * tfc_vec, dim=2, keepdim=True)))
        # cross dot
        feature_cos_similarity_matrix = 1 - torch.bmm(ifc_vec_l2, tfc_vec_l2.permute(0, 2, 1))
        return feature_cos_similarity_matrix

    def L2_similarity(self, image_features, target_features):
        pass

    def CX(self, feature_similarity_matrix):
        feature_similarity_matrix = torch.abs(feature_similarity_matrix)
        C,H,W = feature_similarity_matrix.shape
        CX_total = 0
        for i in range(C):
            CX_loss_diag = torch.diag(feature_similarity_matrix[i])
            CX_loss = torch.mean(CX_loss_diag)
            CX_total = CX_total+CX_loss
        CX_loss_final = CX_total/C
        return CX_loss_final

    def forward(self, image_features, target_features):
        if self.similarity == 'cos':
            feature_similarity_matrix = self.cos_similarity(image_features, target_features)
        elif self.similarity == 'l2':
            feature_similarity_matrix = self.L2_similarity(image_features, target_features)
        else:
            feature_similarity_matrix = self.cos_similarity(image_features, target_features)

        # return self.CX(feature_similarity_matrix),feature_similarity_matrix
        return self.CX(feature_similarity_matrix)

class EucSimilarLoss(nn.Module):
    def __init__(self, similarity='l2'):
        super(EucSimilarLoss, self).__init__()
        self.similarity = similarity

    def cos_similarity(self, image_features, target_features):
        # N, V, C
        if_vec = image_features.view((image_features.size()[0], image_features.size()[1], -1)).permute(0, 2, 1)
        tf_vec = target_features.view((target_features.size()[0], target_features.size()[1], -1)).permute(0, 2, 1)
        # Centre by T
        tf_mean = torch.mean(tf_vec, dim=1, keepdim=True)
        ifc_vec = if_vec - tf_mean
        tfc_vec = tf_vec - tf_mean
        # L2-norm normalization
        ifc_vec_l2 = torch.div(ifc_vec, torch.sqrt(torch.sum(ifc_vec * ifc_vec, dim=2, keepdim=True)))
        tfc_vec_l2 = torch.div(tfc_vec, torch.sqrt(torch.sum(tfc_vec * tfc_vec, dim=2, keepdim=True)))
        # cross dot
        feature_cos_similarity_matrix = 1 - torch.bmm(ifc_vec_l2, tfc_vec_l2.permute(0, 2, 1))
        return feature_cos_similarity_matrix

    def L2_similarity(self, image_features, target_features):
        # print(image_features[0].shape)
        cri = nn.L1Loss()
        # for i in range(16):
        l2_floss = cri(image_features, target_features)
        # print(l2_floss)
        return l2_floss

    def CX(self, feature_similarity_matrix):
        feature_similarity_matrix = torch.abs(feature_similarity_matrix)
        C,H,W = feature_similarity_matrix.shape
        CX_total = 0
        for i in range(C):
            CX_loss_diag = torch.diag(feature_similarity_matrix[i])
            CX_loss = torch.mean(CX_loss_diag)
            CX_total = CX_total+CX_loss
        CX_loss_final = CX_total/C
        return CX_loss_final

    def forward(self, image_features, target_features):
        if self.similarity == 'cos':
            feature_similarity_matrix = self.cos_similarity(image_features, target_features)
        elif self.similarity == 'l2':
            feature_similarity_matrix = self.L2_similarity(image_features, target_features)
        else:
            feature_similarity_matrix = self.cos_similarity(image_features, target_features)

        # return self.CX(feature_similarity_matrix),feature_similarity_matrix
        # return self.CX(feature_similarity_matrix)
        return feature_similarity_matrix
#
class SelfCosSimilarLoss(nn.Module):
    def __init__(self, similarity='cos'):
        super(SelfCosSimilarLoss, self).__init__()
        self.similarity = similarity

    def cos_similarity(self, image_features, target_features):
        # N, V, C
        if_vec = image_features.view((image_features.size()[0], image_features.size()[1], -1)).permute(0, 2, 1)
        tf_vec = target_features.view((target_features.size()[0], target_features.size()[1], -1)).permute(0, 2, 1)
        # Centre by T
        tf_mean = torch.mean(tf_vec, dim=1, keepdim=True)
        ifc_vec = if_vec - tf_mean
        tfc_vec = tf_vec - tf_mean
        # L2-norm normalization
        ifc_vec_l2 = torch.div(ifc_vec, torch.sqrt(torch.sum(ifc_vec * ifc_vec, dim=2, keepdim=True)))
        tfc_vec_l2 = torch.div(tfc_vec, torch.sqrt(torch.sum(tfc_vec * tfc_vec, dim=2, keepdim=True)))
        # cross dot
        feature_cos_similarity_matrix = 1 - torch.bmm(ifc_vec_l2, tfc_vec_l2.permute(0, 2, 1))
        return feature_cos_similarity_matrix

    def L2_similarity(self, image_features, target_features):
        pass

    def CX(self, feature_similarity_matrix, selffeature_similarity_matrix):

        feature_similarity_matrix_err = torch.abs(feature_similarity_matrix - selffeature_similarity_matrix)

        C,H,W = feature_similarity_matrix_err.shape
        CX_total = 0
        for i in range(C):
            CX_loss_diag = torch.diag(feature_similarity_matrix_err[i])
            CX_loss = torch.mean(CX_loss_diag)
            CX_total = CX_total+CX_loss
        CX_loss_final = CX_total/C

        testv1 = torch.sum(feature_similarity_matrix_err)
        testv2 = testv1/C/H/100
        # print(testv2)
        # print(CX_loss_final)

        # max_i_on_j = torch.max(feature_similarity_matrix_err, dim=1)[0]
        # CS = torch.mean(max_i_on_j, dim=1)
        # CS_loss = torch.mean(CS)

        CS_loss = 0.02*testv2
        # CS_loss = testv2

        # print(CS_loss)
        # print("-----------------")

        CX_loss_total = CS_loss+CX_loss_final

        return CX_loss_total

    def forward(self, image_features, target_features):
        if self.similarity == 'cos':
            feature_similarity_matrix = self.cos_similarity(image_features, target_features)
            selffeature_similarity_matrix = self.cos_similarity(target_features, target_features)

        elif self.similarity == 'l2':
            feature_similarity_matrix = self.L2_similarity(image_features, target_features)
        else:
            feature_similarity_matrix = self.cos_similarity(image_features, target_features)
            selffeature_similarity_matrix = self.cos_similarity(target_features, target_features)

        return self.CX(feature_similarity_matrix, selffeature_similarity_matrix)

# Assume input range is [0, 1]
class VGGContextualLoss(nn.Module):
    def __init__(self,
                 feature_layer=17,
                 use_bn=False,
                 use_input_norm=True,
                 sigma=0.1, b=1.0, epsilon=1e-5, similarity='cos',cxloss_type='cx',
                 device=torch.device('cpu')):
        super(VGGContextualLoss, self).__init__()
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        if cxloss_type == 'cx':
            self.CXLoss = ContextualLoss(sigma=sigma, b=b, epsilon=epsilon, similarity=similarity)
            print("cx_type is cx ")
        elif cxloss_type == 'cos':
            self.CXLoss = CosSimilarLoss(similarity=similarity)
            print("cx_type is cos ")
        elif cxloss_type == 'selfcos':
            self.CXLoss = SelfCosSimilarLoss(similarity=similarity)
            print("cx_type is selfcx ")
        elif cxloss_type == 'euc':
            self.CXLoss = EucSimilarLoss(similarity='l2')

        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:17])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, im1, im2):
        if self.use_input_norm:
            im1 = (im1 - self.mean) / self.std
            im2 = (im2 - self.mean) / self.std

        output1_features = self.features(im1)
        output2_features = self.features(im2)
        cx_loss = self.CXLoss(output1_features, output2_features)
        return cx_loss

# blur loss
class Blur(nn.Module):
    def __init__(self, l=15, kernel=None, use_input_norm=True,device=torch.device('cpu')):
        super(Blur, self).__init__()
        self.l = l
        self.pad = nn.ReflectionPad2d(l // 2)
        self.kernel = Variable(torch.FloatTensor(kernel).view((1, 1, self.l, self.l)))

        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

    def cuda(self, device=None):
        self.kernel = self.kernel.cuda()

    def forward(self, input):
        if self.use_input_norm:
            input = (input - self.mean) / self.std    	
        B, C, H, W = input.size()
        pad = self.pad(input)
        H_p, W_p = pad.size()[-2:]
        input_CBHW = pad.view((C * B, 1, H_p, W_p))
        self.kernel = self.kernel.cuda()
        output = F.conv2d(input_CBHW, self.kernel).view(B, C, H, W)
        return output

def isotropic_gaussian_kernel(l, sigma, tensor=True,device=torch.device('cpu')):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    kernel_out = torch.FloatTensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)
    # output = output.cuda()
    return kernel_out


####################
#D Perceptual Network
####################


# Assume input range is [0, 1]
class DnetFeatureExtractor(nn.Module):
    def __init__(self,
                 model,
                 feature_layer=4,
                 use_input_norm=True,
                 device=torch.device('cpu')):
        super(DnetFeatureExtractor, self).__init__()
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.module.features.children())[:4])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output

def define_Dfeature(opt,model, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # pytorch pretrained VGG19-54, before ReLU.

    netDfeature = DnetFeatureExtractor(model , feature_layer=4, \
        use_input_norm=True, device=device)
    if gpu_ids:
        netDfeature = nn.DataParallel(netDfeature)
    netDfeature.eval()  # No need to train
    return netDfeature

class SensMarginRankingLoss(nn.Module):
    def __init__(self,
                 margine=0.5,size_average = True,
                 device=torch.device('cpu')):

        super(SensMarginRankingLoss, self).__init__()


    def forward(self, input1, input2, y, margin, sensweight, size_average):

        _output = input1.clone()
        _output.add_(-1, input2)
        _output.mul_(-1).mul_(y)
        _output.add_(margin)
        _output.clamp_(min=0)
        
        # output = _output.sum()

        # if size_average:
        #     output = output / y.size(0)

        return output