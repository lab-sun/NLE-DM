"""
To quantitatively evaluate the prediction performance of network that jointly predict action and description.
The F1 score is used for evaluate the prediction performance.
"""

import os
import datetime
import torch
import time
from src import act_des_resnet50, act_des_resnet101, act_des_mobile_large, act_des_mobile_small
from train_utils import evaluate
from dataset.dataset_bddad import BDD_AD


def create_model(arg, pretrain=True):
    # To change model with different backbone.
    # act_des_resnet50: ResNet50; act_des_resnet101: ResNet101;
    # act_des_mobile_large: MobilenetV3_Large; act_des_mobile_small: MobilenetV3_Small
    model = act_des_resnet50(arg)

    # to produce the prediction result, the trained weight need to be load
    # set pretrain=True, change the weight path
    if pretrain:
        weights_dict = torch.load("../weights/act_des_resnet50.pth", map_location='cpu')
        weights_dict = weights_dict["model"]

        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=True)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    return model

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size

    results_file = "result{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    test_dataset = BDD_AD(args.data_path, train_set=False)

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             drop_last=True,
                                             pin_memory=True)

    model = create_model(args)
    model.to(device)
    start_time = time.time()

    val_loss, Action_overall, Reason_overall, f1_action, action_average, f1_reason, reason_average = \
        evaluate(model, test_loader, device=device)

    val_info = """
            val loss: {0}.
            F1 for action: {1}.
            F1 for description: {2}.
            Overall F1 for action: {3}.
            Overall F1 for description: {4}.
            Mean F1 action: {5}.
            Mean F1 for description: {6}.        
            """.format(str(val_loss), str(Action_overall), str(Reason_overall), str(f1_action), str(action_average),
                       str(f1_reason), str(reason_average))
    print(val_info)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("testing time {}".format(total_time_str))

    # write into txt
    with open(results_file, "a") as f:
        f.write(val_info + "\n\n")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch deeplabv3 training")

    parser.add_argument("--data-path", default="path to the BDD-AD dataset")
    parser.add_argument("--num-classes", default=(4, 6), type=int)
    parser.add_argument("--aux", default=False, type=bool, help="auxilier loss")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=1, type=int)

    parser.add_argument("--epochs", default=20, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument('--lr', default=0.005, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
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
    main(args)
