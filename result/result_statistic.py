from utils import get_load_result_data


merge_data = get_load_result_data()
compare_CPs = ['CP (1 sec)', 'CP (10 sec)', 'CP (60 sec)']


##############################################################################################################
def get_optimal_gap_rank(merge_data, method_list, del_problems, consider_problem='', compare_CPs=[]):
    merge_data2 = merge_data
    merge_data = merge_data[merge_data['method'].isin(method_list)]
    for benchmark in del_problems:
        merge_data = merge_data[-merge_data['problem'].str.contains(benchmark)]
    if consider_problem:
        merge_data = merge_data[merge_data['problem'].str.contains(consider_problem)]

    merge_data['rank'] = merge_data.groupby('name')['opt_gap'].rank(method='min', ascending=True)

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

    rank_data = merge_data.groupby(['method'], as_index=False).mean()
    rank_data = rank_data.drop(['instance_i', 'job_n', 'mc_n', 'compare_gap'], axis=1)

    rank_data2 = merge_data.groupby(['method'], as_index=False).sum()

    rank_data['best_without_CP'] = rank_data2['best_without_CP']
    for cp_name in compare_CPs:
        rank_data[f'{cp_name}_equal'] = rank_data2[f'{cp_name}_equal']
        rank_data[f'{cp_name}_better'] = rank_data2[f'{cp_name}_better']

    pivot = merge_data.pivot(index=['problem', 'instance_i'], columns=['method'], values='opt_gap')

    return len(pivot), rank_data


##############################################################################################################
# rank for -DMU ##########
method_list = ['Ours', 'ScheduleNet', 'Park', 'LTT', 'MOR', 'SPT']
instance_n, rank_data = get_optimal_gap_rank(merge_data, method_list, ['DMU'], compare_CPs=compare_CPs)
print('-DMU', instance_n)
print(rank_data)


# # rank for -DMU, -TA ##########
# method_list = ['Ours', 'ScheduleNet', 'Park', 'LTT', 'MOR', 'SPT']
# instance_n, rank_data = get_optimal_gap_rank(merge_data, method_list, ['DMU', 'TA'], compare_CPs=compare_CPs)
# print('-DMU, -TA', instance_n)
# print(rank_data)
#
#
# # rank for TA ##################
# method_list = ['Ours', 'ScheduleNet', 'Park', 'Zhang', 'LTT', 'MOR', 'SPT']
# instance_n, rank_data = get_optimal_gap_rank(merge_data, method_list, [], compare_CPs=compare_CPs, consider_problem='TA')
# print('TA', instance_n)
# print(rank_data)
#
#
# # rank for DMU ################
# method_list = ['Ours', 'Zhang', 'LTT', 'MOR', 'SPT']
# instance_n, rank_data = get_optimal_gap_rank(merge_data, method_list, [], compare_CPs=compare_CPs, consider_problem='DMU')
# print('DMU', instance_n)
# print(rank_data)


##############################################################################################################
def get_job_n_result(merge_data, method_list, del_problems, group_creteria, col='job_n', consider_problem=''):
    merge_data = merge_data[merge_data['method'].isin(method_list)]
    for benchmark in del_problems:
        merge_data = merge_data[-merge_data['problem'].str.contains(benchmark)]
    if consider_problem:
        merge_data = merge_data[merge_data['problem'].str.contains(consider_problem)]

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

    group_data = merge_data.groupby(['method', 'group'], as_index=False).mean()
    group_data = group_data.pivot(index='group', columns=['method'], values='opt_gap')

    return group_data


##############################################################################################################
group_creteria = [10, 20, 30, 100]
group_creteria2 = [6, 10, 15, 20]

# group for -DMU ##########
method_list = ['Ours', 'ScheduleNet', 'Park', 'LTT', 'MOR', 'SPT']
group_data = get_job_n_result(merge_data, method_list, ['DMU'], group_creteria, col='job_n')
group_data2 = get_job_n_result(merge_data, method_list, ['DMU'], group_creteria2, col='mc_n')

# # group for DMU ##########
# method_list = ['Ours', 'Zhang', 'LTT', 'MOR', 'SPT']
# group_data = get_job_n_result(merge_data, method_list, [], group_creteria, consider_problem='DMU', col='job_n')
# group_data2 = get_job_n_result(merge_data, method_list, [], group_creteria2, consider_problem='DMU', col='mc_n')


print()


