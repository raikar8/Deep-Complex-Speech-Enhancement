import os, argparse, shutil
import torch, math
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from dataloader import loaddataset
from auraloss.time import SISDRLoss
from auraloss.freq import STFTLoss
import warnings
warnings.filterwarnings("ignore")
epsilon = torch.finfo(torch.float32).eps


class DeepLearningModel(pl.LightningModule):
    def __init__(self, net, batch_size=1):
        super(DeepLearningModel, self).__init__()
        self.model = net
        self.modelname = self.model.name
        self.batch_size = batch_size
        self.si_sdr = SISDRLoss()
        self.freqloss = STFTLoss(fft_size=320, hop_size=80, win_length=320, sample_rate=16000, scale_invariance=False,
                                 w_sc=0.0)
        print('\nUsing Si-SDR + STFT loss function to train the network! ....')

    def forward(self, x):
        return self.model(x)

    def loss_function(self, cln_audio, enh_audio):
        return self.si_sdr(cln_audio, enh_audio) + 25 * self.freqloss(cln_audio, enh_audio)

    def training_step(self, batch, batch_nb):
        enh_audio = self(batch['noisy'])
        loss = self.loss_function(batch['clean'], enh_audio)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        enh_audio = self(batch['noisy'])
        loss = self.loss_function(batch['clean'], enh_audio)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss}

    def on_validation_epoch_end(self, outputs):
        if outputs:
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            self.log('val_loss_epoch', avg_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, weight_decay=1e-5, betas=(0.5, 0.999))
        scheduler = {'scheduler': ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True),
                     'interval': 'epoch', 'frequency': 1, 'reduce_on_plateau': True, 'monitor': 'val_loss'}
        return [optimizer], [scheduler]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Speech Enhancement using complex DCCTN/MoE models')
    parser.add_argument('--model', type=str, help='ModelName', default='DCCTN')
    parser.add_argument('--mode', type=str, help='Choose between "summary", "fast_run" or "train"', default='summary')
    parser.add_argument('--b', type=int, help='Batch Size', default=8)
    parser.add_argument('--e', type=int, help='Epochs', default=2)
    parser.add_argument('--gpu', type=str, help='GPU-IDs used to Train', default='0')
    parser.add_argument('--loss', type=str, help='loss function', default='SISDR+FreqLoss')
    args = parser.parse_args()

    from Network import CFTNet, DCCTN, DATCFTNET, DATCFTNET_DSC
    from moe_networks import DCCTN_SoftMoE, DCCTN_HardMoE
    model_classes = {
        'CFTNet': CFTNet,
        'DCCTN': DCCTN,
        'DATCFTNET': DATCFTNET,
        'DATCFTNET_DSC': DATCFTNET_DSC,
        'DCCTN_SoftMoE': DCCTN_SoftMoE,
        'DCCTN_HardMoE': DCCTN_HardMoE,
    }
    if args.model not in model_classes:
        raise ValueError('Unknown model {}. Choose from: {}'.format(args.model, ', '.join(model_classes)))
    curr_model = model_classes[args.model]()

    print('This process has the PID', os.getpid())
    model = DeepLearningModel(curr_model, args.b)
    print('Training Model: ' + model.modelname)
    gpuIDs = [int(k) for k in args.gpu.split()]
    print('Training on GPU(s) : ', gpuIDs)
    save_dir = os.path.join(os.getcwd(), 'Saved_Models', model.modelname)
    os.makedirs(save_dir, exist_ok=True)
    callbacks = ModelCheckpoint(monitor='val_loss', dirpath=save_dir,
                                filename=model.modelname + '-DADX-IEEE-' + args.loss + '-{epoch:02d}-{val_loss:.2f}',
                                save_top_k=1, mode='min')
    TrainData = loaddataset(os.path.join(os.getcwd(), 'Database/Training_Samples/Train'))
    trainloader = DataLoader(TrainData, batch_size=args.b, shuffle=True, num_workers=12, pin_memory=True)
    DevData = loaddataset(os.path.join(os.getcwd(), 'Database/Training_Samples/Dev'))
    devloader = DataLoader(DevData, batch_size=args.b, shuffle=False, num_workers=12, pin_memory=True)
    print(torch.cuda.is_available())
    trainer = pl.Trainer(max_epochs=args.e, gpus=gpuIDs, strategy='ddp', callbacks=callbacks,
                         gradient_clip_val=10, accumulate_grad_batches=8)
    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=devloader)
    print('Done!')

# Examples:
# python3 train.py --model DCCTN_SoftMoE --b 8 --e 50 --loss SISDR+FreqLoss --gpu '0 1'
# python3 train.py --model DCCTN_HardMoE --b 8 --e 50 --loss SISDR+FreqLoss --gpu '0 1'
