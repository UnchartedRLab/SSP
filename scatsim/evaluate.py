import yaml
import argparse

from src.trainer import ScatSimCLRTrainer, PretextTaskTrainer


def run_evaluation(config_path, mode):
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    if mode == 'unsupervised':
        trainer = ScatSimCLRTrainer(config)
    elif mode == 'pretext':
        trainer = PretextTaskTrainer(config)

    score = trainer.evaluate()
    print(f'Score: {score}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m',
                        help='Training mode. `unsupervised` - run training only with contrastive loss, '
                             '`pretext` - run training with contrastive loss and pretext task',
                        choices=['unsupervised', 'pretext'])
    parser.add_argument('--config', '-c',
                        type=str,
                        help='configuration file')
    args = parser.parse_args()
    mode = args.mode
    if mode not in ['unsupervised', 'pretext']:
        raise ValueError('Unsupported mode')

    run_evaluation(args.config, mode)













