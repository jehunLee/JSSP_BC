# https://developers.google.com/optimization/scheduling/job_shop

import collections
from ortools.sat.python import cp_model
import csv
from utils import load_data
import os
from tqdm import tqdm
from collections import defaultdict


def JSSP_solver(benchmark: str, job_n: int, mc_n: int, instance_i: int, time_limit: int=3600,
                save_path: str=None, obj_type: str='makespan') -> (int, int):
    # Creates the model.
    model = cp_model.CpModel()

    job_mcs, job_prts = load_data(benchmark, job_n, mc_n, instance_i)
    all_mcs = range(0, mc_n)
    all_jobs = range(0, job_n)

    # Computes horizon statically.
    horizon = sum([sum(prts) for prts in job_prts])
    # Named tuple to store information about created variables.
    task_type = collections.namedtuple('task_type', 'start end interval')

    # Creates jobs.
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)
    for i in all_jobs:
        for j, mc_i in enumerate(job_mcs[i]):
            start_var = model.NewIntVar(0, horizon, f'start_{i}_{j}')
            end_var = model.NewIntVar(0, horizon, f'end_{i}_{j}')
            interval_var = model.NewIntervalVar(start_var, int(job_prts[i][j]), end_var, f'interval_{i}_{j}')
            all_tasks[(i, j)] = task_type(start=start_var, end=end_var, interval=interval_var)
            machine_to_intervals[mc_i].append(interval_var)

    # Create disjunctive constraints.
    for mc_i in all_mcs:
        model.AddNoOverlap(machine_to_intervals[mc_i])

    # Precedences inside a job.
    for i in all_jobs:
        for j in range(len(job_mcs[i]) - 1):
            model.Add(all_tasks[(i, j+1)].start >= all_tasks[(i, j)].end)

    # objective.
    if obj_type == 'makespan':
        obj_var = model.NewIntVar(0, horizon, 'makespan')
        model.AddMaxEquality(obj_var, [all_tasks[(i, len(job_mcs[i])-1)].end for i in all_jobs])
        model.Minimize(obj_var)

    elif obj_type == 'total_completion':
        horizon_ = horizon * job_n

        obj_var = model.NewIntVar(0, horizon_, 'total_completion')
        model.Add(obj_var == sum(all_tasks[(i, len(job_mcs[i])-1)].end for i in all_jobs))
        model.Minimize(obj_var)

    else:
        obj_var = None
        NotImplementedError()

    # Solve model.
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(model)

    # UNKNOWN = 0    # MODEL_INVALID = 1    # FEASIBLE = 2    # INFEASIBLE = 3    # OPTIMAL = 4
    obj_v = solver.ObjectiveValue()

    if save_path:
        result_save(save_path, benchmark, job_n, mc_n, instance_i, time_limit,
                    obj_v, solver.WallTime(), status)

    # Output solution.
    obj_v = solver.Value(obj_var)
    if status == cp_model.OPTIMAL:
        # Finally print the solution found.
        # print('Optimal makespan: %i' % obj_v)
        # print(output)
        his_mc_s_t = defaultdict(list)
        his_mc_e_t = defaultdict(list)
        his_mc_job_i = defaultdict(list)
        his_mc_job_op_i = defaultdict(list)
        mc_tuple = defaultdict(list)
        for i in all_jobs:
            for j in range(len(job_mcs[i])):
                mc_i = job_mcs[i][j]
                his_mc_s_t[mc_i].append(solver.Value(all_tasks[i, j].start))
                his_mc_e_t[mc_i].append(solver.Value(all_tasks[i, j].end))
                his_mc_job_i[mc_i].append(i)
                his_mc_job_op_i[mc_i].append(j)
                mc_tuple[mc_i].append((i, solver.Value(all_tasks[i, j].start)))

        solution_save(benchmark, job_n, mc_n, instance_i, mc_tuple, obj_v, obj_type)

    return status, obj_v


def solution_save(benchmark: str, job_n: int, mc_n: int, instance_i: int, mc_tuple: dict, obj: int,
                  obj_type: str='makespan') -> None:
    problem = f'{benchmark}{job_n}x{mc_n}'
    folder_path = f'./../benchmark/{benchmark}/{problem}'
    if not os.path.isdir(folder_path):
        folder_path = f'./benchmark/{benchmark}/{problem}'

    # folder_path = folder_path + f'/{obj_type}'
    # if not os.path.isdir(folder_path):
    #     os.mkdir(folder_path)

    with open(folder_path + f'/opt_{instance_i}.csv', 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow([obj, benchmark, job_n, mc_n, instance_i])

        for j in range(mc_n):
            mc_tuple[j] = sorted(mc_tuple[j], key=lambda x: x[1])
            job_is = [x[0] for x in mc_tuple[j]]  # job_idx
            wr.writerow(job_is)

        for j in range(mc_n):
            job_s_ts = [x[1] for x in mc_tuple[j]]  # start time
            wr.writerow(job_s_ts)


def result_save(save_path: str, benchmark: str, job_n: int, mc_n: int, instance_i: int, time_limit: int,
                obj_value: float, search_t: float, convergence: int) -> None:
    with open(save_path, 'a', newline='') as f:
        wr = csv.writer(f)
        wr.writerow([time_limit, benchmark, job_n, mc_n, instance_i, obj_value, round(search_t, 4), convergence])


if __name__ == "__main__":
    from utils import all_benchmarks
    save_path = None
    obj_type = 'makespan'
    # save_path = './../result/bench_cp.csv'
    for time_limit in [3600]:  # 1, 10, 60, 300, 3600
        for (benchmark, job_n, mc_n, save_is) in [('HUN', 5, 2, [0])]:
        # for (benchmark, job_n, mc_n, save_i_n) in all_benchmarks:
            print(benchmark, job_n, mc_n)

            for instance_i in tqdm(save_is):
                _, _ = JSSP_solver(benchmark, job_n, mc_n, instance_i, time_limit, save_path=save_path,
                                   obj_type=obj_type)
