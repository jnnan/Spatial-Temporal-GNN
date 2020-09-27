import yaml
from train.train_main import train_main

if __name__ == '__main__':
    with open('configs/config.yaml') as f:
        config = yaml.load(f)
        config = config['dcrnn']
        train_main(config)
