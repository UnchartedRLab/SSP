import argparse
import numpy as np
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.data import get_feature_data, get_feature_data_imagenet
from lib.train_tools import get_device, test, train_dfa

from lib.tinydfa import DFAManager, FeedbackLayer
from lib.tinydfa.alignment import GradientAlignmentMetrics
from lib.tinydfa.rp.differential_privacy import RandomProjectionDP, TernarizedRandomProjectionDP
from lib.tinydfa.utils.normalizations import FeedbackNormalization
from lib.tinydfa.asymmetric import AsymmetricFunction, ForwardFunction, BackwardFunction

from opacus.accountants.analysis.rdp import get_privacy_spent

DEFAULT_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

'''
tau_feedback_privacy is tau_B (tau_f)
tau_privacy[0] is tau_h_min
tau_privacy[1] is tau_h_max
np.sqrt(hidden_size) is np.sqrt(n_l)
'''


class FullyConnected(nn.Module):
    def __init__(
        self,
        hidden_size,
        input_num,
        output_num,
        sigma_privacy=0.01,
        tau_privacy=(1e-6, 0.1),
        tau_feedback_privacy=1,
        training_method="DFA",
    ):
        super(FullyConnected, self).__init__()
        self.fc1 = nn.Linear(input_num, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_num)

        self.training_method = training_method
        normalization = FeedbackNormalization.FAN_OUT
        activation = AsymmetricFunction(ForwardFunction.TANH, BackwardFunction.TANH)

        self.rp = RandomProjectionDP(
            sigma_privacy=sigma_privacy,
            tau_feedback_privacy=tau_feedback_privacy,
            verbose=False,
        )
        if self.training_method == "TDFA":
            self.rp = TernarizedRandomProjectionDP(
                ternarization_treshold=0.15,
                sigma_privacy=sigma_privacy,
                tau_feedback_privacy=tau_feedback_privacy,
                verbose=False,
            )

        self.dfa1 = FeedbackLayer()
        self.dfa = DFAManager(
            [self.dfa1],
            no_feedbacks=(self.training_method == "SHALLOW"),
            rp_operation=self.rp,
            normalization=normalization,
        )

        if self.training_method == "SHALLOW":
            self.dfa.record_feedback_point_all = False

        self.activation = lambda x: (activation(x) + (tau_privacy[0] / np.sqrt(hidden_size))).clamp(
            -tau_privacy[1] / np.sqrt(hidden_size), tau_privacy[1] / np.sqrt(hidden_size)
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.activation(self.fc1(x))
        x = self.dfa1(x)
        x = self.dfa(self.fc2(x))
        x = F.log_softmax(x, dim=0)
        return x



def main(args):
    device = get_device()

    dataset = args.dataset
    
    if dataset == 'cifar10':
        n_outputs=10
    elif dataset == 'plant':
        n_outputs=39
    elif dataset == 'eurosat':
        n_outputs=10
    elif dataset == 'isic2018':
        n_outputs=7
    elif dataset == 'cifar100':
        n_outputs=100
    elif dataset == 'mnist':
        n_outputs=10
    elif dataset == 'fmnist':
        n_outputs=10
    elif dataset == 'imagenet':
        n_outputs=1000
    else:
        n_outputs=100



    if dataset == 'imagenet':
        test_loader, train_loader, n_features, n_train = get_feature_data_imagenet(args.feature_path, args.batch_size, False, 1, False)
    else:
        test_loader, train_loader, n_features, n_train = get_feature_data(args.feature_path, args.batch_size, False, 1, False)
    
    model = FullyConnected(
        args.hidden_size,
        n_features,
        n_outputs,
        sigma_privacy=args.sigma_privacy,
        tau_privacy=(args.tau_min_privacy, args.tau_max_privacy),
        tau_feedback_privacy=args.tau_feedback_privacy,
        training_method=args.training_method,
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    alignment = GradientAlignmentMetrics(model)

    #q = args.batch_size/50000
    #s = math.ceil(1/q)
    sensitivity_per_layer = 2 * 2 / args.batch_size
    sensitivity = math.sqrt(2 * sensitivity_per_layer * sensitivity_per_layer)
    temp = sensitivity * sensitivity / (args.sigma_privacy * args.sigma_privacy * 2)
    rdp = np.array([ temp * alpha for alpha in DEFAULT_ALPHAS])
    epsilons = []
    test_accs = []
    alphas = []


    for epoch in range(args.epochs):
        train_dfa(model, train_loader, optimizer, alignment, device, epoch)
        lo, acc = test(model, test_loader, False, epoch, device)
        rdp_t = (epoch+1) * rdp
        eps, opt_alpha = get_privacy_spent(orders=DEFAULT_ALPHAS, rdp=rdp_t, delta=1e-5)
        print(f"{eps},{acc}")
        #print(f"(eps = {eps:.2f}, opt_alpha = {opt_alpha}) \t")
        epsilons.append(eps)
        test_accs.append(acc)
        alphas.append(opt_alpha)

    results_file_name = './results/{}_dfa_{}.csv'.format(args.dataset, args.extractor)

    a = np.array(epsilons)
    b = np.array(test_accs)
    c = np.array(alphas)
    df = pd.DataFrame({"epsilon" : a, "test_acc" : b, "alpha": c})
    df.to_csv(results_file_name, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DP DFA tuning")
    parser.add_argument(
        "--training-method",
        type=str,
        choices=["DFA", "TDFA", "SHALLOW"],
        default="DFA",
        help="training method to choose from DFA, TDFA or SHALLOW",
    )
    parser.add_argument(
        "--ternarization-treshold",
        type=float,
        default=0.15,
        help="treshold for ternarization with TDFA/ODFA (default: 0.15)",
    )
    parser.add_argument(
        "--sigma-privacy",
        type=float,
        default=0.2,
        help="DP sigma value for synthetic gradient noise (default: 0.01)",
    )
    parser.add_argument(
        "--tau-min-privacy",
        type=float,
        default=1e-6,
        help="tau_min value for activation offsetting (default:1e-6)",
    )
    parser.add_argument(
        "--tau-max-privacy",
        type=float,
        default=1.0,
        help="DP tau_max value for activation clipping (default:1.0)",
    )
    parser.add_argument(
        "--tau-feedback-privacy",
        type=float,
        default=1.0,
        help="DP tau_feedback value for feedback clipping (default:1.0)",
    )

    # Model definition:
    parser.add_argument("--hidden-size", type=int, default=512, help="hidden layer size (default: 256)")
    parser.add_argument("--batch-size", type=int, default=256, help="training batch size (default: 128)")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs to train (default: 15)")

    # Optimization:
    parser.add_argument("--lr", type=float, default=1e-2, help="SGD learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum (default: 0.9)")

    # Dataset:
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--feature-path", type=str, default="data", help="path to dataset")
    parser.add_argument("--extractor", default="vit")

    args = parser.parse_args()
    main(args)
