from utils import get_load_result_data


##############################################################################################################
def get_optimal_gap_rank(merge_data, method_list, del_problems=[], consider_problem='', compare_CPs=[]):
    merge_data2 = merge_data
    merge_data = merge_data[merge_data['method'].isin(method_list)]
    for benchmark in del_problems:
        merge_data = merge_data[-merge_data['problem'].str.contains(benchmark)]
    if consider_problem:
        merge_data = merge_data[merge_data['problem'].str.contains(consider_problem)]

    if 'Han' in method_list:
        problems = merge_data.loc[(merge_data['method'] == 'Han')]['name'].values.tolist()
        merge_data = merge_data[merge_data['name'].isin(problems)]
        print(f'# of Han instances: {len(problems)}')

    if 'Liu' in method_list:
        problems = merge_data.loc[(merge_data['method'] == 'Liu')]['name'].values.tolist()
        merge_data = merge_data[merge_data['name'].isin(problems)]
        print(f'# of Liu instances: {len(problems)}')

    merge_data['rank'] = merge_data.groupby('name')['opt_gap'].rank(method='min', ascending=True)

    merge_data['opt_TF'] = [1 if r == 0 else 0 for r in merge_data['opt_gap']]

    # best without CP ############################
    merge_data['best_without_CP'] = [1 if r == 1 else 0 for r in merge_data['rank']]

    # compare with CP ############################
    for cp_name in compare_CPs:
        merge_data['compare_gap'] = [merge_data2.loc[(merge_data2['problem'] == p) & (merge_data2['instance_i'] == i) &
                                                    (merge_data2['method'] == cp_name)
                                                    ].iloc[0]['opt_gap']
                                     for (p, i) in zip(merge_data['problem'], merge_data['instance_i'])]
        merge_data[f'{cp_name}_equal'] = [1 if o <= c_o else 0 for (o, c_o) in zip(merge_data['opt_gap'],
                                                                                   merge_data['compare_gap'])]
        merge_data[f'{cp_name}_better'] = [1 if o < c_o else 0 for (o, c_o) in zip(merge_data['opt_gap'],
                                                                                   merge_data['compare_gap'])]

    print(merge_data['method'].value_counts())

    rank_data = merge_data.groupby(['method'], as_index=False).mean()
    rank_data = rank_data.drop(['instance_i', 'job_n', 'mc_n'], axis=1)

    rank_data2 = merge_data.groupby(['method'], as_index=False).sum()

    rank_data['best_without_CP'] = rank_data2['best_without_CP']
    for cp_name in compare_CPs:
        rank_data[f'{cp_name}_equal'] = rank_data2[f'{cp_name}_equal']
        rank_data[f'{cp_name}_better'] = rank_data2[f'{cp_name}_better']

    rank_data['opt_TF'] = rank_data2['opt_TF']
    pivot = merge_data.pivot(index=['problem', 'instance_i'], columns=['method'], values='opt_gap')

    merge_data0 = merge_data.groupby(['problem', 'method'], as_index=False).mean()
    pivot0 = merge_data0.pivot(index=['problem'], columns=['method'], values='opt_gap')




    merge_data2 = merge_data2[merge_data2['method'].isin(method_list)]
    if consider_problem:
        merge_data2 = merge_data2[merge_data2['problem'].str.contains(consider_problem)]

    merge_data2['rank'] = merge_data2.groupby('name')['opt_gap'].rank(method='min', ascending=True)
    rank_data3 = merge_data2.groupby(['method', 'benchmark'], as_index=False).mean()
    rank_data3 = rank_data3.drop(['instance_i', 'job_n', 'mc_n'], axis=1)

    pivot1 = rank_data3.pivot(index=['benchmark'], columns=['method'], values='opt_gap')
    pivot2 = rank_data3.pivot(index=['benchmark'], columns=['method'], values='rank')

    return len(pivot), rank_data, pivot1, pivot2


##############################################################################################################
model_i = 3
file_name = f'./data_n/result_500_0.001_1e-05_{model_i}.csv'
merge_data = get_load_result_data(test_csv_file_name=file_name)
# compare_CPs = ['CP (1 sec)', 'CP (10 sec)', 'CP (60 sec)']

method_list = ['Zhang', 'ScheduleNet', 'Park', 'LTT', 'FDD/MWKR', 'MOR', 'SPT', 'Ours']
merge_data_ = merge_data[merge_data['method'].isin(method_list)]
group_data2 = merge_data_.groupby(['method', 'problem'], as_index=False).mean()
group_data3 = group_data2.pivot(index='problem', columns=['method'], values='opt_gap')
group_data2_ = merge_data_.groupby(['method', 'problem'], as_index=False).sum()
group_data3_ = group_data2_.pivot(index='problem', columns=['method'], values='instance_i')

# rank for -DMU ##########
method_list = ['ScheduleNet', 'Park', 'LTT', 'FDD/MWKR', 'MOR', f'model{model_i}']  # 'Ours', , 'SPT', 'LRPT'
instance_n, rank_data, pivot1, pivot2 = get_optimal_gap_rank(merge_data, method_list, del_problems=['DMU'])
# print('-DMU', instance_n)
print(rank_data)

method_list = ['Iklassov', 'ScheduleNet', 'Park', 'LTT', 'Zhang', 'Ours']  # 'Ours', , 'SPT', 'LRPT'
instance_n, rank_data, pivot1, pivot2 = get_optimal_gap_rank(merge_data, method_list, consider_problem='TA')
# print('-DMU', instance_n)
print(rank_data)


##############################################################################################################
def get_job_n_result(merge_data, method_list, del_problems, group_creteria, col='job_n', consider_problem=''):
    merge_data = merge_data[merge_data['method'].isin(method_list)]
    for benchmark in del_problems:
        merge_data = merge_data[-merge_data['problem'].str.contains(benchmark)]
    if consider_problem:
        merge_data = merge_data[merge_data['problem'].str.contains(consider_problem)]

    if 'Han' in method_list:
        problems = merge_data.loc[(merge_data['method'] == 'Han')]['name'].values.tolist()
        merge_data = merge_data[merge_data['name'].isin(problems)]
        print(f'# of Han instances: {len(problems)}')

    if 'Liu' in method_list:
        problems = merge_data.loc[(merge_data['method'] == 'Liu')]['name'].values.tolist()
        merge_data = merge_data[merge_data['name'].isin(problems)]
        print(f'# of Liu instances: {len(problems)}')

    def get_group(j, col, group_creteria):
        name = 'large'
        s_c = 0
        for e_c in group_creteria:
            if s_c < j and j <= e_c:
                name = f'{s_c} < {col} <= {e_c}'
                break
            s_c = e_c
        return name

    merge_data['group'] = [get_group(j, col, group_creteria) for j in merge_data[col]]

    # group_data = merge_data.groupby(['method', 'group'], as_index=False).mean()
    # group_data = group_data.pivot(index='group', columns=['method'], values='opt_gap')

    print(merge_data['group'].value_counts() / 7)
    # print(merge_data['method'].value_counts())
    #
    merge_data['rank'] = merge_data.groupby('name')['opt_gap'].rank(method='min', ascending=True)
    group_data = merge_data.groupby(['method', 'group'], as_index=False).mean()
    group_data1 = group_data.pivot(index='group', columns=['method'], values='rank')
    group_data2 = group_data.pivot(index='group', columns=['method'], values='opt_gap')

    return group_data1, group_data2


##############################################################################################################
# group_creteria = [10, 20, 30, 100]
group_creteria = [10, 15, 20, 30, 100]

group_creteria2 = [6, 10, 15, 20]

# group for -DMU ##########
# method_list = ['Ours', 'ScheduleNet', 'Park', 'LTT', 'FDD/MWKR', 'MOR', 'SPT']
# group_data = get_job_n_result(merge_data, method_list, ['DMU'], group_creteria, col='job_n')
# group_data2 = get_job_n_result(merge_data, method_list, ['DMU'], group_creteria2, col='mc_n')

# # group for DMU ##########
# method_list = ['Ours', 'Zhang', 'LTT', 'MOR', 'SPT']
# group_data = get_job_n_result(merge_data, method_list, [], group_creteria, consider_problem='DMU', col='job_n')
# group_data2 = get_job_n_result(merge_data, method_list, [], group_creteria2, consider_problem='DMU', col='mc_n')

method_list = [f'model{model_i}', 'Han', 'ScheduleNet', 'Park', 'LTT', 'FDD/MWKR', 'MOR']
group_data1, group_data2 = get_job_n_result(merge_data, method_list, ['DMU'], group_creteria, col='job_n')
# group_data2 = get_job_n_result(merge_data, method_list, ['DMU'], group_creteria2, col='mc_n')
print()

method_list = ['Ours', 'ScheduleNet', 'Park', 'LTT', 'FDD/MWKR', 'Han', 'Liu']  # , 'MOR', 'SPT',
group_data = get_job_n_result(merge_data, method_list, ['DMU'], group_creteria, col='job_n')
# group_data2 = get_job_n_result(merge_data, method_list, ['DMU'], group_creteria2, col='mc_n')
print()


