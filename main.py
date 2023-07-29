import torch

from my_parser import parsed_args
from train import Trainer


if __name__ == '__main__':
    torch.cuda.empty_cache()
    args = parsed_args

    args.load_pre_model = 0
    pre_model_path = 'point/LINUX_pre_model_07_28_19_53_34.pt'

    args.decoder = 'mlp5'

    trainer = Trainer(args)

    pre_train = 0
    if pre_train:
        args.switch = False
        args.align = False
        trainer.pre_train(pre_model_path)
    else:
        trainer.kfold_train(pre_model_path)
