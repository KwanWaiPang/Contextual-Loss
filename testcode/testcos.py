
from PIL import Image
from . models.modules.loss import VGGContextualLoss



cxloss = VGGContextualLoss(opt, use_bn=False, cxloss_type="cos")
cxloss.cuda()

pic_pil = Image.open('../../data/val/classpatch/building_469.png').convert('RGB')

pic_var = PIL2VAR(pic_pil, volatile=True)
####
hr_img = pic_var.cuda()

# pic_pil2 = Image.open('result/SRoutput/SRbaby_GT.bmp').convert('RGB')
# pic_var2 = PIL2VAR(pic_pil2, volatile=True)
####
# sr_var = pic_var2.cuda()
# sr_cx_loss = cxloss(sr_var, hr_img)  # CXloss
# print(sr_cx_loss)
pic_pil2 = Image.open('../../data/val/classpatch/building_469.png').convert('RGB')
# pic_pil2 = Image.open('result/test_samples/SRGANCX.png').convert('RGB')
pic_var2 = PIL2VAR(pic_pil2, volatile=True)
####
sr_var = pic_var2.cuda()

sr_cx_loss,feature,featurenorm,featurelog = cxloss(sr_var, hr_img)  # CXloss
feature1 = feature.cpu()
featurenorm1 = featurenorm.cpu()
featurelog1 = featurelog.cpu()
feature1 = feature1.data.numpy()
featurenorm1 = featurenorm1.data.numpy()
featurelog1 = featurelog1.data.numpy()

print(sr_cx_loss)
print(feature1)

dataNew ='GT2GT.mat'
# scio.savemat(dataNew, {'feature':feature})
scio.savemat(dataNew, {'feature':feature1,'norm':featurenorm1,'featurelog':featurelog1})