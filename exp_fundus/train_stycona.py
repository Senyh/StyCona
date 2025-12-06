import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import time
import argparse
import copy
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from easydict import EasyDict
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset, random_split
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToPILImage
from exp_fundus.data.dataset import FundusDataset
from models import networks
from wheels.loss_functions import DSCLossH
from wheels.logger import logger as logging
from utils import ensure_dir
from wheels.mask_generator import BoxMaskGenerator, AddMaskParamsToBatch, SegCollate
from wheels.torch_utils import seed_torch
from wheels.model_init import init_weight
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args(known=False):
    parser = argparse.ArgumentParser(description='PyTorch Implementation')
    parser.add_argument('--seed', type=int, default=2025, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--project', type=str, default=os.path.dirname(os.path.realpath(__file__)) + '/runs/StyCona', help='project path for saving results')
    parser.add_argument('--backbone', type=str, default='UNet', choices=['UNet'], help='segmentation backbone')
    parser.add_argument('--data_path', type=str, default='YOUR_DATA_PATH', help='path to the data')
    parser.add_argument('--train_modality', type=str, default='ORIGA', 
                        choices=['BinRushed', 'Drishti_GS', 'Magrabia', 'ORIGA', 'REFUGE'], 
                        help='the image modality for training')
    parser.add_argument('--image_size', type=int, default=256, help='the size of images for training and testing')
    parser.add_argument('--labeled_percentage', type=float, default=1., help='the percentage of labeled data')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='number of inputs per batch')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers to use for dataloader')
    parser.add_argument('--in_channels', type=int, default=3, help='input channels')
    parser.add_argument('--num_classes', type=int, default=3, help='number of target categories')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--log_freq', type=int, default=10, help='logging frequency of metrics accord to the current iteration')
    parser.add_argument('--save_freq', type=int, default=10, help='saving frequency of model weights accord to the current epoch')
    args = parser.parse_known_args()[0] if known else parser.parse_args()
    return args


def get_data(args):
    val_set = FundusDataset(image_path=args.data_path, stage='val', image_size=args.image_size, is_augmentation=False, modality=args.train_modality)
    labeled_train_set = FundusDataset(image_path=args.data_path, stage='train', image_size=args.image_size, is_augmentation=True, labeled=True, percentage=args.labeled_percentage, modality=args.train_modality, stycona=True)
    unlabeled_train_set = FundusDataset(image_path=args.data_path, stage='train', image_size=args.image_size, is_augmentation=True, labeled=False, percentage=args.labeled_percentage, modality=args.train_modality, stycona=True)
    train_set = ConcatDataset([labeled_train_set, unlabeled_train_set])

    # repeat the labeled set to have a equal length with the unlabeled set (dataset)
    print('before: ', len(train_set), len(labeled_train_set), len(unlabeled_train_set), len(val_set))
    labeled_ratio = len(train_set) // len(labeled_train_set)
    labeled_train_set = ConcatDataset([labeled_train_set for i in range(labeled_ratio)])
    labeled_train_set = ConcatDataset([labeled_train_set,
                                       Subset(labeled_train_set, range(len(train_set) - len(labeled_train_set)))])
    print('after: ', len(train_set), len(labeled_train_set), len(unlabeled_train_set), len(val_set))
    assert len(labeled_train_set) == len(train_set)
    train_labeled_dataloder = DataLoader(dataset=labeled_train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    train_unlabeled_dataloder = DataLoader(dataset=train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(dataset=val_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    mask_generator = BoxMaskGenerator(prop_range=(0.25, 0.5),
                                        n_boxes=3,
                                        random_aspect_ratio=True,
                                        prop_by_area=True,
                                        within_bounds=True,
                                        invert=True)

    add_mask_params_to_batch = AddMaskParamsToBatch(mask_generator)
    mask_collate_fn = SegCollate(batch_aug_fn=add_mask_params_to_batch)
    aux_labeled_dataloader = DataLoader(dataset=train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=mask_collate_fn)
    return train_labeled_dataloder, train_unlabeled_dataloder, val_dataloader, aux_labeled_dataloader


def main(is_debug=False):
    args = get_args()
    seed_torch(args.seed)
    # Project Saving Path
    project_path = args.project + '_{}_label_{}_trainmodal_{}/'.format(args.backbone, args.labeled_percentage, args.train_modality)
    ensure_dir(project_path)
    save_path = project_path + 'weights/'
    ensure_dir(save_path)

    # Tensorboard & Statistics Results & Logger
    tb_dir = project_path + '/tensorboard{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
    writer = SummaryWriter(tb_dir)
    metrics = EasyDict()
    metrics.train_loss = []
    metrics.train_s_loss = []
    metrics.train_u_loss = []
    metrics.val_loss = []
    logger = logging(project_path + 'train_val.log')
    logger.info('PyTorch Version {}\n Experiment{}'.format(torch.__version__, project_path))

    # Load Data
    train_labeled_dataloader, _, val_dataloader, aux_labeled_dataloader = get_data(args=args)
    iters = len(train_labeled_dataloader)
    val_iters = len(val_dataloader)

    # Load Model & EMA
    student = networks.__dict__[args.backbone](in_channels=args.in_channels, out_channels=args.num_classes).to(device)
    init_weight(student.net.classifier, nn.init.kaiming_normal_,
                nn.BatchNorm2d, 1e-5, 0.1,
                mode='fan_in', nonlinearity='relu')
    best_model_wts = copy.deepcopy(student.state_dict())
    logger.info('#parameters: {}'.format(sum(param.numel() for param in student.parameters())))
    best_epoch = 0
    best_loss = 100

    # Criterion & Optimizer & LR Schedule
    criterion = DSCLossH(num_classes=args.num_classes, device=device)
    optimizer = optim.AdamW(student.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))

    # Train
    since = time.time()
    logger.info('start training')
    for epoch in range(1, args.num_epochs + 1):
        epoch_metrics = EasyDict()
        epoch_metrics.train_loss = []
        epoch_metrics.train_s_loss = []
        epoch_metrics.train_u_loss = []
        if is_debug:
            pbar = range(10)
        else:
            pbar = range(iters)
        iter_train_labeled_dataloader = iter(train_labeled_dataloader)

        ############################
        # Train
        ############################
        student.train()
        for idx in pbar:
            # labeled data
            limage, label, limageA1, limageA2 = next(iter_train_labeled_dataloader)
            limage, label = limage.to(device), label.to(device)
            limageA1, limageA2 = limageA1.to(device), limageA2.to(device)

            optimizer.zero_grad()
            
            # the forward pass of labeled images
            pred_lA1 = student(limageA1)
            pred_lA1_logits = pred_lA1['out']

            # the supervised loss
            loss = criterion(pred_lA1_logits, label.squeeze(1).long())
            loss.backward()
            optimizer.step()

            writer.add_scalar('train_loss', loss.item(), idx + len(pbar) * (epoch-1))
            if idx % args.log_freq == 0:
                logger.info("Train: Epoch/Epochs {}/{}, "
                            "iter/iters {}/{}, "
                            "loss {:.3f}".format(epoch, args.num_epochs, idx, len(pbar), loss.item()))
            epoch_metrics.train_loss.append(loss.item())
        metrics.train_loss.append(np.mean(epoch_metrics.train_loss))

        ############################
        # Validation
        ############################
        epoch_metrics.val_loss = []
        iter_val_dataloader = iter(val_dataloader)
        if is_debug:
            val_pbar = range(10)
        else:
            val_pbar = range(val_iters)
        student.eval()
        with torch.no_grad():
            for idx in val_pbar:
                image, label = next(iter_val_dataloader)
                image, label = image.to(device), label.to(device)
                pred = student(image)['out']
                loss = criterion(pred, label.squeeze(1).long())
                writer.add_scalar('val_loss', loss.item(), idx + len(val_pbar) * (epoch-1))
                if idx % args.log_freq == 0:
                    logger.info("Val: Epoch/Epochs {}/{}\t"
                                "iter/iters {}/{}\t"
                                "loss {:.3f}".format(epoch, args.num_epochs, idx, len(val_pbar),
                                                     loss.item()))
                epoch_metrics.val_loss.append(loss.item())
        metrics.val_loss.append(np.mean(epoch_metrics.val_loss))

        # Save Model
        if np.mean(epoch_metrics.val_loss) <= best_loss:
            best_model_wts = copy.deepcopy(student.state_dict())
            best_epoch = epoch
            best_loss = np.mean(epoch_metrics.val_loss)
            torch.save(best_model_wts, save_path + 'best.pth'.format(best_epoch))
        torch.save(student.state_dict(), save_path + 'last.pth'.format(best_epoch))
        logger.info("Average: Epoch/Epoches {}/{}, "
                    "train epoch loss {:.3f}, "
                    "val epoch loss {:.3f}, "
                    "best loss {:.3f} at {}\n".format(epoch, args.num_epochs, np.mean(epoch_metrics.train_loss),
                                                     np.mean(epoch_metrics.val_loss), best_loss, best_epoch))
    ############################
    # Save Metrics
    ############################
    data_frame = pd.DataFrame(
        data={'loss': metrics.train_loss,
              'val_loss': metrics.val_loss},
        index=range(1, args.num_epochs + 1))
    data_frame.to_csv(project_path + 'train_val_loss.csv', index_label='Epoch')
    plt.figure()
    plt.title("Loss During Training and Validating")
    plt.plot(metrics.train_loss, label="Train")
    plt.plot(metrics.val_loss, label="Val")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(project_path + 'train_val_loss.png')

    print(project_path)
    time_elapsed = time.time() - since
    logger.info('Training completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logger.info('TRAINING FINISHED!')


if __name__ == '__main__':
    main()

