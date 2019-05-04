def create_model(opt):
    model = opt['model']

    if model == 'sr':
        from .SR_model import SRModel as M
    elif model == 'srcos':
        from .SR_cos_model import SRModel as M
    elif model == 'srgan':
        from .SRGAN_model import SRGANModel as M
    elif model == 'srgancx':
        from .SRGAN_CX_model import SRGANModel as M
    elif model == 'srgandf':
        from .SRGAN_DF_model import SRGANModel as M
    elif model == 'srganres':
        from .SRGAN_residual_model import SRGANModel as M
    elif model == 'srragan':
        from .SRRaGAN_model import SRRaGANModel as M
    elif model == 'sftgan':
        from .SFTGAN_ACD_model import SFTGAN_ACD_Model as M
    elif model == 'srgan_rank':
        from .SRGAN_rank_model import SRGANModel as M
    elif model == 'srgan_rankpair':
        from .SRGAN_rankpair_model import SRGANModel as M
    elif model == 'rank':
        from .Estimation_model import Estimation_Model as M
    elif model == 'regression':
        from .regression_model import regression_Model as M
    else:
        raise NotImplementedError('Model [%s] not recognized.' % model)
    m = M(opt)
    print('Model [%s] is created.' % m.__class__.__name__)
    return m
