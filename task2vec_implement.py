import argparse

from utils.config import create_config
from utils.common_config import get_train_dataset_name, get_transformations, get_model
from task2vec.task2vec import Task2Vec

from data.tabular import TabularRegression

import torch


def main():
    # read best model
    print('begin function')
    p = create_config(args.config_env, args.config_exp)
    model = get_model(p)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(p['checkpoint'], map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    # read data
    train_transforms, val_transforms = get_transformations(p)

    # task2vec
    embeddings = []
    for name in p.ALL_TASKS.NAMES[:10]:
        print(name)
        dataset = get_train_dataset_name(p, train_transforms, 1, name)
        embeddings.append(Task2Vec(model, max_samples=1000, target=name).embed(dataset))
    torch.save(embeddings, 'task2vec_result')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vanilla Training')
    parser.add_argument('--config-env',
                        help='Config file for the environment')
    parser.add_argument('--config-exp',
                        help='Config file for the experiment')
    args = parser.parse_args()

    main()
