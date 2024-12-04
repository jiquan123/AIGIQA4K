import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_info',
                        type=str,
                        help='info that will be displayed when logging',
                        default='AIGCIQA2023')

    parser.add_argument('--lr',
                        type=float,
                        help='learning rate',
                        default=1e-4)

    parser.add_argument('--weight_decay',
                        type=float,
                        help='L2 weight decay',
                        default=1e-5)

    parser.add_argument('--seed',
                        type=int,
                        help='manual seed',
                        default=1)

    parser.add_argument('--gpu',
                        type=str,
                        help='id of gpu device(s) to be used',
                        default='0')

    parser.add_argument('--train_batch_size',
                        type=int,
                        help='batch size for training phase',
                        default=8)

    parser.add_argument('--test_batch_size',
                        type=int,
                        help='batch size for test phase',
                        default=20)

    parser.add_argument('--num_epochs',
                        type=int,
                        help='number of training epochs',
                        default=100)

    parser.add_argument('--backbone',
                        type=str,
                        help='which backbone model to use',
                        default='vit')

    parser.add_argument('--true_score',
                        type=str,
                        help='which score to predict',
                        default='MOS_q')
    
    parser.add_argument('--using_prompt',
                        type=int,
                        help='whether to use image prompt',
                        default=1)
    
    parser.add_argument('--benchmark',
                        type=str,
                        help='which benchmark to use',
                        default='T2I')

    return parser