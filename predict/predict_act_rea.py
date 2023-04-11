"""
To quantitatively evaluate the prediction performance of network that jointly predict action and explanation.
The F1 score is used for evaluate the prediction performance.
"""

import os
import datetime
import torch
import time
from src import act_exp
from train_utils import evaluate
from dataset.dataset_bddoia_predict import BddoiaDataset

def create_model(aux, num_classes: tuple, pretrain=True):
    model = act_exp(aux=aux, num_classes=num_classes)

    # to produce the prediction result, the trained weight need to be load
    # set pretrain=True, change the weight path
    if pretrain:
        weights_dict = torch.load("/home/fyc/project/NEL-DM/weights/act_rea/act_rea_resnet50.pth", map_location='cpu')
        weights_dict = weights_dict["model"]

        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    return model

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size

    results_file = "result{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    test_dataset = BddoiaDataset(args.data_path, train_set=False)

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             drop_last=True,
                                             pin_memory=True)

    model = create_model(aux=args.aux, num_classes=args.num_classes)
    model.to(device)

    start_time = time.time()

    val_loss, Action_overall, Reason_overall, f1_action, action_average, f1_reason, reason_average = evaluate(model, test_loader, device=device)
    val_info = """
    val loss for this epoch is: {0}.
    Overall F1 score for action is: {1}.
    Overall F1 score for explanation is: {2}.
    Category F1 score for action is: {3}.
    Average F1 score for action is: {4}.
    Category F1 score for explanation is: {5}.
    Average F1 score for explanation is: {6}.        
    """.format(str(val_loss), str(Action_overall), str(Reason_overall), str(f1_action), str(action_average),
               str(f1_reason), str(reason_average))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("testing time {}".format(total_time_str))

    print(val_info)

    # write into txt
    with open(results_file, "a") as f:
        f.write(val_info + "\n\n")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch deeplabv3 training")

    parser.add_argument("--data-path", default="/workspace/dataset/BDD-OIA/lastframe")
    parser.add_argument("--num-classes", default=(4, 21), type=int)
    parser.add_argument("--aux", default=False, type=bool, help="auxilier loss")
    parser.add_argument("--device", default="cuda:0", help="training device")
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
