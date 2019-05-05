import torch
def calc_NIQE(input_image_path, shave_width):
    # initialization
    NIQE = []
    blocksizerow = 96
    blocksizecol = 96
    blockrowoverlap = 0
    blockcoloverlap = 0
    # load mu_prisparam,cov_prisparam from modelparameters.mat

    # Step 1: remove a 'shave_width'-pixel wide strip from each board
    # input_image = convert_shave_image(imread(input_image_path),shave_width)

    # Step 2: compute quality
    #    NIQE = computequality(input_image,blocksizerow,blocksizecol,...
    #   blockrowoverlap,blockcoloverlap,mu_prisparam,cov_prisparam);
    return NIQE
def computequality(input_image,blocksizerow,blocksizecol,blockrowoverlap,
                   blockcoloverlap,mu_prisparam,cov_prisparam):
    # initialization
    feanum = 18

    # Step 1: Pretreatment
    # Grayscale the input_image
    # tranform the d-type of input_image into 'float'
    row, col = input_image.size()
    block_rownum = row / blocksizerow
    block_colnum = col / blocksizecol
    #im               = im(1:block_rownum*blocksizerow, ...
    #               1:block_colnum*blocksizecol);
