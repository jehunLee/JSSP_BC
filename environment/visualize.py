import plotly.express as px
import pandas as pd


def show_gantt(mc_n: int, mc_s_t: dict, mc_e_t: dict, mc_job: dict, mc_job_op_i: dict):
    for mc_i in range(mc_n):
        for i, s_t in enumerate(mc_s_t[mc_i]):
            e_t = mc_e_t[mc_i][i]
            job_i = mc_job[mc_i][i]
            op_i = mc_job_op_i[mc_i][i]

            if mc_i == 0 and i == 0:
                first_list = [(mc_i, s_t, e_t, job_i, op_i)]
                col_name = ['resource', 'start', 'end', 'job', 'op_i']
                df = pd.DataFrame(first_list, columns=col_name)
            else:
                df = df.append(dict(resource=mc_i, start=s_t, end=e_t, job=job_i, op_i=op_i), ignore_index=True)

    df['delta'] = df['end'] - df['start']
    fig = px.timeline(df, x_start="start", x_end="end", y="resource", color="task", text="task", opacity=0.6,
                      color_continuous_scale='rainbow')
    fig.update_yaxes(autorange="reversed")

    fig.layout.xaxis.type = 'linear'
    fig.data[0].x = df.delta.tolist()
    fig.show()


