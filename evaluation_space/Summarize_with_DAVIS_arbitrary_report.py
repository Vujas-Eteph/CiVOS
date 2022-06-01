import pandas as pd
from click_davisinteractive import evaluation
import glob

def read_my_csv(file_path):
    return pd.read_csv(file_path, index_col=0)


MAX_INTERACTIONS   = 8
MAX_TIME           = 30       # Official --> 8*30
SUBSET             = 'val'      #'train', 'val'
METRIC_TO_OPTIMIZE = 'J_AND_F'  #'J', 'F' or 'J_AND_F'
davis_path = '../DAVIS/2017'


path_2_csv_files = sorted(glob.glob('./csv_results/*.csv'))


for csv_file in path_2_csv_files:
    csv_table = read_my_csv(csv_file)

    Eval_report = evaluation.service.EvaluationService(davis_root=davis_path+'/trainval',
                                                       subset= SUBSET,
                                                       max_t = MAX_INTERACTIONS*MAX_TIME if MAX_TIME is not None else None,
                                                       max_i = MAX_INTERACTIONS,
                                                       metric_to_optimize=METRIC_TO_OPTIMIZE)


    output = Eval_report.summarize_report_vujas(csv_table, round_results = 2, use_time_axis=False)
    print(output)
