import os

import torch
# from torchmetrics import R2Score
from torchmetrics import PearsonCorrCoef
from torchmetrics.functional import mean_squared_error


class RegressionMeter(object):
    def __init__(self, target_name: str):
        self.target_name = target_name
        self.eval_dict = {'r2': PearsonCorrCoef()**2, 'rmse': 0., 'n': 0}

    @torch.no_grad()
    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        pred = pred.detach().cpu().reshape(-1)
        gt = gt.detach().cpu().reshape(-1)

        self.eval_dict['n'] += 1
        self.eval_dict['r2'].update(pred, gt)
        self.eval_dict['rmse'] += mean_squared_error(pred, gt, squared=False)

    def reset(self):
        self.eval_dict = {'r2': PearsonCorrCoef()**2, 'rmse': 0., 'n': 0}

    def get_score(self, verbose=True):
        eval_result = {
            "r2": self.eval_dict["r2"].compute(),
            "rmse": self.eval_dict["rmse"] / self.eval_dict['n']
        }

        if verbose:
            print(f'Results {self.target_name}')
            for x in eval_result:
                spaces = ""
                for j in range(0, 15 - len(x)):
                    spaces += ' '
                print('{0:s}{1:s}{2:.4f}'.format(x, spaces, eval_result[x]))

        return eval_result


def eval_regression_predictions(database, save_dir):
    """ Evaluate the regression results that are stored in the save dir """

    # Dataloaders
    if database == 'PROSPECT':
        from data.tabular import TabularRegression
        gt_set = 'val'
        db = TabularRegression(
            path="./dataset/PROSPECT.csv", task_names=p.ALL_TASKS.NAMES, split_ratio=0.8, split=gt_set
        )

    else:
        raise NotImplementedError

    base_name = database + '_' + 'test' + '_normals'
    fname = os.path.join(save_dir, base_name + '.json')

    # Eval the model
    print('Evaluate the saved images (surface normals)')
    eval_results = eval_normals(db, os.path.join(save_dir, 'normals'))
    with open(fname, 'w') as f:
        json.dump(eval_results, f)

    # Print results
    print('Results for Surface Normal Estimation')
    for x in eval_results:
        spaces = ""
        for j in range(0, 15 - len(x)):
            spaces += ' '
        print('{0:s}{1:s}{2:.4f}'.format(x, spaces, eval_results[x]))

    return eval_results