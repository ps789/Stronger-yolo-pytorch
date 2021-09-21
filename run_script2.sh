#!/bin/bash
python main.py --config-file configs/strongerv3_quantile_dropout.yaml EXPER.experiment_name strongerv3_ensemble EXPER.resume dropout2 do_test True devices 0,1,
