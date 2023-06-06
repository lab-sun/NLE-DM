"""
This script is to train the network to jointly predict actions and descriptions.
In this script, the relative path for dataset is : '../Data/BDD_OIA10k/',
Please change both two paths according to your file structure.
"""

import os
import time
import datetime
import torch
from src import act_des_resnet50, act_des_resnet101, act_des_mobile_large, act_des_mobile_small
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from dataset.dataset_bddad import BDD_AD


def create_model(aux, num_classes: tuple, pretrain=True):
    # To change model with different backbone.
    # act_des_resnet50: ResNet50; act_des_resnet101: ResNet101;
    # act_des_mobile_large: MobilenetV3_Large; act_des_mobile_small: MobilenetV3_Small
    model = act_des_resnet50(aux=aux, num_classes=num_classes)

    """
    To load the pretrained weights on BDD10K:
    1) set pretrain=True;
    2) change the path for weights
    """
    if pretrain:
        weights_dict = torch.load("../weights/seg_weight/bdd10k_resnet50_1.pth", map_location='cpu')
        weights_dict = weights_dict["model"]

        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    return model

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    num_classes = args.num_classes

    # to save information during training and verification
    results_file = "results_{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    train_dataset = BDD_AD(args.data_path, train_set=True)

    val_dataset = BDD_AD(args.data_path, train_set=False)

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               drop_last=True,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             drop_last=True,
                                             pin_memory=True)

    model = create_model(aux=args.aux, num_classes=num_classes)
    model.to(device)

    params_to_optimize = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad]}, # comment out to fix
        {"params": [p for p in model.classifier.parameters() if p.requires_grad]},
        {"params": [p for p in model.classifier_neck.parameters() if p.requires_grad]},
        {"params": [p for p in model.classifier_action.parameters() if p.requires_grad]},
        {"params": [p for p in model.classifier_reason.parameters() if p.requires_grad]},
    ]

    if args.aux:
        pass

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # Create a learning rate update strategy, which is updated once for each step
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer']) # comment out when fix part of the model
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        total_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        val_loss, Action_overall, Reason_overall, f1_action, action_average, f1_reason, reason_average = evaluate(model, val_loader, device=device)
        val_info = """
        val loss for this epoch is: {0}.
        Overall F1 score for action is: {1}.
        Overall F1 score for description is: {2}.
        Category F1 score for action is: {3}.
        Average F1 score for action is: {4}.
        Category F1 score for description is: {5}.
        Average F1 score for description is: {6}.        
        """.format(str(val_loss), str(Action_overall), str(Reason_overall), str(f1_action), str(action_average), str(f1_reason), str(reason_average))
        print(val_info)

        # write into txt
        with open(results_file, "a") as f:
            # Record the train information for each epoch
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {total_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n"
            # val_info = "val loss for this epoch: " + val_info
            f.write(train_info + val_info + "\n\n")

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()
        torch.save(save_file, "./model_{}.pth".format(epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))



def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch deeplabv3 training")

    parser.add_argument("--data-path", default="/BDD_AD")
    parser.add_argument("--num-classes", default=(4, 6), help='the number of action and environment description/reason')
    parser.add_argument("--aux", default=False, type=bool, help="aux is turn off during whole training")
    parser.add_argument("--device", default="cuda:0", help="training device")
    parser.add_argument("-b", "--batch-size", default=2, type=int)
    parser.add_argument("--epochs", default=100, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument('--lr', default=0.005, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./debug_NLE-DM"):
        os.mkdir("./debug_NLE-DM")

    main(args)
