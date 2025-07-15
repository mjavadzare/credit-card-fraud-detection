import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from tests import test_model
from src import visualization


def main():
    pred_lr, pred_lgbm = test_model.run()
    visualization.run(pred_lr=pred_lr, pred_lgbm=pred_lgbm)
    print('Operation is now completed.')


if __name__ == "__main__":
    main()