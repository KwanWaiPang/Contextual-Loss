import torch.utils.data


def create_dataloader(dataset, dataset_opt):
    phase = dataset_opt['phase']
    if phase == 'train':
        batch_size = dataset_opt['batch_size']
        shuffle = dataset_opt['use_shuffle']
        num_workers = dataset_opt['n_workers']
    else:
        batch_size = 1
        shuffle = False
        num_workers = 1
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    # shuffle:是否将数据打乱； pin_memory:如果True,将数据转换倒GPU会快点。

def create_dataset(dataset_opt,is_train = True):
    mode = dataset_opt['mode']
    if mode == 'LR':
        from data.LR_dataset import LRDataset as D
    elif mode == 'LRHR':
        from data.LRHR_dataset import LRHRDataset as D
    elif mode == 'LRHRseg':
        from data.LRHR_seg_dataset import LRHRSegDataset as D
    elif mode == 'RANK_IMIM':
        from data.Rank_IMIM_dataset import RANK_IMIM_Dataset as D
    elif mode == 'RANK_IMIM_Pair':
        from data.Rank_IMIM_Pair_dataset import RANK_IMIM_Pair_Dataset as D
    elif mode == 'RANK_IMIM_Random':
        from data.Rank_IMIM_Random_dataset import RANK_IMIM_Random_Dataset as D
    elif mode == 'RANK_IMIM_Random_twostyle':
        from data.Rank_IMIM_Random_dataset_twostyle import RANK_IMIM_Random_Dataset_twostyle as D
    elif mode == 'regression':
        from data.regression_dataset import regression_Dataset as D
    elif mode == 'RANK_IMIM_Pair_twostyle':
        from data.Rank_IMIM_dataset_twostyle import Rank_IMIM_dataset_twostyle as D
    else:
        raise NotImplementedError("Dataset [%s] is not recognized." % mode)
    if 'RANK_IMIM' in mode:
        dataset = D(dataset_opt, is_train = is_train)
    elif 'regression' in mode:
        dataset = D(dataset_opt, is_train = is_train)
    else:
        dataset = D(dataset_opt)
    # print('Dataset [%s - %s] is created.' % (dataset.name(), dataset_opt['name']))
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,dataset_opt['name']))
    return dataset
