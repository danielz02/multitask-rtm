python main.py --config-env configs/env.yml --config-exp configs/prospect/multi_task_baseline.yml
python main_nortm.py --config-env configs/env_nortm.yml --config-exp configs/prospect/multi_task_baseline_nortm.yml
python main_single.py --config-env configs/env_single_N.yml --config-exp configs/prospect/single_task_test_N.yml
python test.py --config-env configs/env_nortm.yml --config-exp configs/prospect/multi_task_baseline_nortm.yml