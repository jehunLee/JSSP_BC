from utils import get_load_result_data
import matplotlib.pyplot as plt
import numpy as np


merge_data = get_load_result_data()


##############################################################################################################
def get_chart(merge_data, method_list, problems, colors, save_path=''):
    merge_data = merge_data.groupby(['problem', 'method'], as_index=False).mean()
    merge_data = merge_data.pivot(index='problem', columns=['method'], values='opt_gap')

    merge_data = merge_data.loc[problems, method_list]

    plt.rcParams["figure.figsize"] = (11.5, 2.1)
    plt.rcParams["font.family"] = 'Times New Roman'

    plt.grid(True, axis='y', linestyle=':', alpha=0.3, color='0.1')
    merge_data.plot.bar(rot=0, color=colors, edgecolor='0.1', linewidth=0.05, width=0.8)

    plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.23), fontsize=11, ncol=10)
    # plt.legend(frameon=False, bbox_to_anchor=(1, 1), fontsize=8.5)
    plt.xlabel('Test benchmark', fontsize=14)
    plt.ylabel('Optimal gap', fontsize=14)
    plt.grid(True, axis='y', linestyle=':', alpha=0.3, color='0.1')
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.ylim([0, 0.45])
    plt.yticks(np.arange(0, 0.5, 0.1))
    if save_path:
        plt.savefig('./fig_result1.eps', format='eps', bbox_inches='tight', pad_inches=0.02)
    plt.show()

    print()

##############################################################################################################
# purple: '#63349D',  '#743CB4', '#A785D6',

# chart for TA ##########
method_list = ['CP (60 sec)',  'CP (10 sec)', 'CP (1 sec)',
               'Ours', 'ScheduleNet', 'Park', 'Zhang', 'LTT', 'MOR', 'SPT']
colors = ['#558B2F',  '#7CB342', '#AED581',
          '#F57C00', '#00BCD4', '#066182', '#002F5C', '#9E9E9E', '#757575', '#616161']
problems = ['TA 15x15', 'TA 20x15', 'TA 20x20', 'TA 30x15', 'TA 30x20',
            'TA 50x15', 'TA 50x20', 'TA 100x20']
get_chart(merge_data, method_list, problems, colors, save_path='./fig_result_TA.eps')

# chart for LA ##########
method_list = ['CP (60 sec)',  'CP (10 sec)', 'CP (1 sec)',
               'Ours', 'ScheduleNet', 'Park', 'LTT', 'MOR', 'SPT']
colors = ['#558B2F',  '#7CB342', '#AED581',
          '#F57C00', '#00BCD4', '#066182', '#9E9E9E', '#757575', '#616161']
problems = ['LA 10x5', 'LA 15x5', 'LA 20x5', 'LA 10x10', 'LA 15x10', 'LA 20x10',
            'LA 30x10', 'LA 15x15']
get_chart(merge_data, method_list, problems, colors, save_path='./fig_result_LA.eps')

# chart for else ##########
method_list = ['CP (60 sec)',  'CP (10 sec)', 'CP (1 sec)',
               'Ours', 'ScheduleNet', 'Park', 'LTT', 'MOR', 'SPT']
colors = ['#558B2F',  '#7CB342', '#AED581',
          '#F57C00', '#00BCD4', '#066182', '#9E9E9E', '#757575', '#616161']
problems = ['FT 6x6', 'FT 10x10', 'FT 20x5', 'ORB 10x10', 'ABZ 10x10', 'ABZ 20x15', 'YN 20x20',
            'SWV 20x10', 'SWV 20x15', 'SWV 50x10']
get_chart(merge_data, method_list, problems, colors, save_path='./fig_result_else.eps')

# chart for LA ##########
method_list = ['CP (60 sec)',  'CP (10 sec)', 'CP (1 sec)',
               'Ours', 'Zhang', 'LTT', 'MOR', 'SPT']
colors = ['#558B2F',  '#7CB342', '#AED581',
          '#F57C00', '#002F5C', '#9E9E9E', '#757575', '#616161']
problems = ['DMU 20x15', 'DMU 20x20', 'DMU 30x15', 'DMU 30x20', 'DMU 40x15',
            'DMU 40x20', 'DMU 50x15', 'DMU 50x20']
get_chart(merge_data, method_list, problems, colors, save_path='./fig_result_DMU.eps')