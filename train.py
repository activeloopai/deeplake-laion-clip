import yaml
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from data.deeplake_dm import DeepLakeDataModule
from models import CLIPWrapper


def main(hparams):
    config_dir = 'models/configs/ViT.yaml' if 'ViT' in hparams.model_name else 'models/configs/RN.yaml'
    with open(config_dir) as fin:
        config = yaml.safe_load(fin)[hparams.model_name]

    if hparams.minibatch_size < 1:
        hparams.minibatch_size = hparams.batch_size

    model = CLIPWrapper(hparams.model_name, config, hparams.minibatch_size)
    del hparams.model_name
    
    dm = DeepLakeDataModule.from_argparse_args(hparams)
    trainer = Trainer.from_argparse_args(hparams, precision=hparams.fp, max_epochs=32, enable_model_summary=False)
    trainer.fit(model, dm)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--minibatch_size', type=int, default=0)
    parser.add_argument('--fp', type=int, default=0)
    parser = DeepLakeDataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args.replace_sampler_ddp = False
    main(args)
