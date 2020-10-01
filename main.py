import yaml
from train.train_main import train_main

if __name__ == '__main__':
    with open('configs/config.yaml') as f:
        args = yaml.load(f)
        args = {**args['general'], **args[args['general']['method']]}
        train_main(args)
