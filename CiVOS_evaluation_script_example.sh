#!/usr/bin/env bash
## bash -i 

echo "Running $0"

# Activate conda env iVOT
conda activate CiVOS

#f1
echo "-----------------------------------------------"
echo "Running f1"
python CiVOS_for_DAVIS.py --output ./evaluation_space/eval_strategy_f1/ --params ./evaluation_space/eval_strategy_f1/strat_f1.yml

#f2
echo "-----------------------------------------------"
echo "Running f2"
python CiVOS_for_DAVIS.py --output ./evaluation_space/eval_strategy_f2/ --params ./evaluation_space/eval_strategy_f2/strat_f2.yml

#f3
echo "-----------------------------------------------"
echo "Running f3"
python CiVOS_for_DAVIS.py --output ./evaluation_space/eval_strategy_f3/ --params ./evaluation_space/eval_strategy_f3/strat_f3.yml
