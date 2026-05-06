#
#
#
# %%
import pandas as pd
import numpy as np
from util import Feature_functions as ut


def Event_rate_event_based(df, time_window, space_window, ev_lim, Mc=None):
    if Mc:
        df_mc = df[df['Magnitude'] >= Mc]
    else:
        df_mc = df

    if not space_window:
        space_window = ['inf']

    results_dict = {'Index': [], 'Time[d.s]': []}
    for ti, t_win in enumerate(time_window):
        for si, s_win in enumerate(space_window):
            column_name = f'n_T({t_win})_S({s_win})_EvLim({ev_lim})'
            results_dict[column_name] = []
            for i in range(df.shape[0]):
                print(f'n_T({t_win})_S({s_win}): {i} out of {df.shape[0]}')
                if i >= ev_lim:
                    candidate_events = ut.CandidateEventsTS(df.loc[:i,:].to_numpy(), t_win, s_win*1000, Mc)
                    data = candidate_events.filter_data()
                    if len(data) > 0:
                        if data.shape[0] < ev_lim:
                            last_df = df_mc[df_mc['Time[d.s]'] < data[0,4]]
                            data = np.concatenate((last_df.iloc[-(ev_lim-data.shape[0]):,:].to_numpy(), data))
                        if ti + si == 0:
                            results_dict['Time[d.s]'].append(df['Time[d.s]'].values[i])
                            results_dict['Index'].append(df['Event_ID'].values[i])
                        results_dict[column_name].append(ut.Event_rate(data, t_win))
                    else:
                        if ti + si == 0:
                            results_dict['Time[d.s]'].append(df['Time[d.s]'].values[i])
                            results_dict['Index'].append(df['Event_ID'].values[i])
                        results_dict[column_name].append(np.nan)

    return pd.DataFrame(results_dict)

def Moment_rate_event_based(df, time_window, space_window, ev_lim, Mc=None):
    if Mc:
        df_mc = df[df['Magnitude'] >= Mc]
    else:
        df_mc = df

    if not space_window:
        space_window = ['inf']

    results_dict = {'Index': [], 'Time[d.s]': []}
    for ti, t_win in enumerate(time_window):
        for si, s_win in enumerate(space_window):
            column_name = f'logM_T({t_win})_S({s_win})_EvLim({ev_lim})'
            results_dict[column_name] = []
            for i in range(df.shape[0]):
                print(f'logM_T({t_win})_S({s_win}): {i} out of {df.shape[0]}')
                if i >= ev_lim:
                    candidate_events = ut.CandidateEventsTS(df.loc[:i,:].to_numpy(), t_win, s_win*1000, Mc)
                    data = candidate_events.filter_data()
                    if data.size > 0:
                        if data.shape[0] < ev_lim:
                            last_df = df_mc[df_mc['Time[d.s]'] < data[0,4]]
                            data = np.concatenate((last_df.iloc[-(ev_lim-data.shape[0]):,:].to_numpy(), data))
                        if ti + si == 0:
                            results_dict['Time[d.s]'].append(df['Time[d.s]'].values[i])
                            results_dict['Index'].append(df['Event_ID'].values[i])
                        results_dict[column_name].append(ut.Moment_rate(data, t_win))
                    else:
                        if ti + si == 0:
                            results_dict['Time[d.s]'].append(df['Time[d.s]'].values[i])
                            results_dict['Index'].append(df['Event_ID'].values[i])
                        results_dict[column_name].append(0)

    return pd.DataFrame(results_dict)

def b_value_event_based(df, time_window, space_window, b_flag, ev_lim, Mc=None):
    if Mc:
        df_mc = df[df['Magnitude'] >= Mc]
    else:
        df_mc = df

    if not space_window:
        space_window = ['inf']

    results_dict = {'Index': [], 'Time[d.s]': []}
    for ti, t_win in enumerate(time_window):
        for si, s_win in enumerate(space_window):
            column_name = f'{b_flag}_T({t_win})_S({s_win})_EvLim({ev_lim})'
            results_dict[column_name] = []
            for i in range(df.shape[0]):
                print(f'b_T({t_win})_S({s_win}): {i} out of {df.shape[0]}')
                if i >= ev_lim:
                    if b_flag == 'bp':
                        candidate_events = ut.CandidateEventsTS(df.loc[:i,:].to_numpy(), t_win, s_win*1000)
                    else:
                        candidate_events = ut.CandidateEventsTS(df.loc[:i,:].to_numpy(), t_win, s_win*1000, Mc)
                    data = candidate_events.filter_data()
                    if data.size > 0:
                        if data.shape[0] < ev_lim:
                            if b_flag == 'bp':
                                last_df = df_mc[df_mc['Time[d.s]'] < data[0,4]]
                            else:
                                last_df = df[df['Magnitude'] > Mc]
                                last_df = last_df[last_df['Time[d.s]'] < data[0,4]]
                            data = np.concatenate((last_df.iloc[-(ev_lim-data.shape[0]):,:].to_numpy(), data))
                        if ti + si == 0:
                            results_dict['Time[d.s]'].append(df['Time[d.s]'].values[i])
                            results_dict['Index'].append(df['Event_ID'].values[i])
                        results_dict[column_name].append(ut.b_value(data, b_flag, Mc)[0])
                    else:
                        if ti + si == 0:
                            results_dict['Time[d.s]'].append(df['Time[d.s]'].values[i])
                            results_dict['Index'].append(df['Event_ID'].values[i])
                        results_dict[column_name].append(0)

    return pd.DataFrame(results_dict)

def Correlation_integral_event_based(df, time_window, space_window, r_max, num_steps, ev_lim, Mc=None):
    if Mc:
        df_mc = df[df['Magnitude'] >= Mc]
    else:
        df_mc = df

    if not space_window:
        space_window = ['inf']

    results_dict = {'Index': [], 'Time[d.s]': []}
    num_steps += 1
    
    for ti, t_win in enumerate(time_window):
        for si, s_win in enumerate(space_window):
            if s_win != 'inf':
                r_max = s_win*1000
            r_values = np.linspace(0, r_max, num_steps)
            for r in range(1,num_steps):
                column_name = f'C-r({r_values[r]/1000:.0f})_T({t_win})_S({s_win})_EvLim({ev_lim})'
                results_dict[column_name] = []

    for ti, t_win in enumerate(time_window):
        for si, s_win in enumerate(space_window):
            if s_win != 'inf':
                r_max = s_win*1000
            r_values = np.linspace(0, r_max, num_steps)
            for i in range(df.shape[0]):
                print(f'C_T({t_win})_S({s_win}): {i} out of {df.shape[0]}')
                if i >= ev_lim:
                    candidate_events = ut.CandidateEventsTS(df.loc[:i,:].to_numpy(), t_win, s_win*1000, Mc)
                    data = candidate_events.filter_data()
                    if data.size > 0:
                        if data.shape[0] < ev_lim:
                            last_df = df_mc[df_mc['Time[d.s]'] < data[0,4]]
                            data = np.concatenate((last_df.iloc[-(ev_lim-data.shape[0]):,:].to_numpy(), data))
                        if ti + si == 0:
                            results_dict['Time[d.s]'].append(df['Time[d.s]'].values[i])
                            results_dict['Index'].append(df['Event_ID'].values[i])
                        res = ut.correlation_integral(data[:,1:4], r_max, num_steps)
                        for r in range(1,num_steps):
                            column_name = f'C-r({r_values[r]/1000:.0f})_T({t_win})_S({s_win})_EvLim({ev_lim})'
                            results_dict[column_name].append(res[r-1])
                    else:
                        if ti + si == 0:
                            results_dict['Time[d.s]'].append(df['Time[d.s]'].values[i])
                            results_dict['Index'].append(df['Event_ID'].values[i])
                        for r in range(1,num_steps):
                            column_name = f'C-r({r_values[r]/1000:.0f})_T({t_win})_S({s_win})_EvLim({ev_lim})'
                            results_dict[column_name].append(0)

    return pd.DataFrame(results_dict)


def InterEvent_t_event_based(df, time_window, space_window, ev_lim, Mc=None):
    if Mc:
        df_mc = df[df['Magnitude'] >= Mc]
    else:
        df_mc = df

    if not space_window:
        space_window = ['inf']

    results_dict = {'Index': [], 'Time[d.s]': []}
    for ti, t_win in enumerate(time_window):
        for si, s_win in enumerate(space_window):
            column_name = f'IEt_T({t_win})_S({s_win})_EvLim({ev_lim})'
            results_dict[column_name] = []
            for i in range(df.shape[0]):
                print(f'IEt_T({t_win})_S({s_win}): {i} out of {df.shape[0]}')
                if i >= ev_lim:
                    candidate_events = ut.CandidateEventsTS(df.loc[:i,:].to_numpy(), t_win, s_win*1000, Mc)
                    data = candidate_events.filter_data()
                    if data.size > 0:
                        if data.shape[0] < ev_lim:
                            last_df = df_mc[df_mc['Time[d.s]'] < data[0,4]]
                            data = np.concatenate((last_df.iloc[-(ev_lim-data.shape[0]):,:].to_numpy(), data))
                        if ti + si == 0:
                            results_dict['Time[d.s]'].append(df['Time[d.s]'].values[i])
                            results_dict['Index'].append(df['Event_ID'].values[i])
                        results_dict[column_name].append(ut.Interevent_ts(data)[0])
                    else:
                        if ti + si == 0:
                            results_dict['Time[d.s]'].append(df['Time[d.s]'].values[i])
                            results_dict['Index'].append(df['Event_ID'].values[i])
                        results_dict[column_name].append(0)

    return pd.DataFrame(results_dict)

def InterEvent_s_event_based(df, time_window, space_window, ev_lim, Mc=None):
    if Mc:
        df_mc = df[df['Magnitude'] >= Mc]
    else:
        df_mc = df

    if not space_window:
        space_window = ['inf']

    results_dict = {'Index': [], 'Time[d.s]': []}
    for ti, t_win in enumerate(time_window):
        for si, s_win in enumerate(space_window):
            column_name = f'IEs_T({t_win})_S({s_win})_EvLim({ev_lim})'
            results_dict[column_name] = []
            for i in range(df.shape[0]):
                print(f'IEs_T({t_win})_S({s_win}): {i} out of {df.shape[0]}')
                if i >= ev_lim:
                    candidate_events = ut.CandidateEventsTS(df.loc[:i,:].to_numpy(), t_win, s_win*1000, Mc)
                    data = candidate_events.filter_data()
                    if data.size > 0:
                        if data.shape[0] < ev_lim:
                            last_df = df_mc[df_mc['Time[d.s]'] < data[0,4]]
                            data = np.concatenate((last_df.iloc[-(ev_lim-data.shape[0]):,:].to_numpy(), data))
                        if ti + si == 0:
                            results_dict['Time[d.s]'].append(df['Time[d.s]'].values[i])
                            results_dict['Index'].append(df['Event_ID'].values[i])
                        results_dict[column_name].append(ut.Interevent_ts(data)[1])
                    else:
                        if ti + si == 0:
                            results_dict['Time[d.s]'].append(df['Time[d.s]'].values[i])
                            results_dict['Index'].append(df['Event_ID'].values[i])
                        results_dict[column_name].append(0)

    return pd.DataFrame(results_dict)

def Clustering_event_based(df, time_window, space_window, ev_lim, Mc=None, b=1, fd=2.5, p=0.5, event_lag=250, time_lag=None, c=6.5):
    if Mc:
        df_mc = df[df['Magnitude'] >= Mc]
    else:
        df_mc = df

    if not space_window:
        space_window = ['inf']

    results_dict = {'Index': [], 'Time[d.s]': []}
    out_names = ['trp', 'trr', 'pfo', 'pma', 'paf', 'num', 'mem', 'len', 'siz', 'rad', 'dim', 'den', 'cer', 'enum']
    for t_win in time_window:
        for s_win in space_window:
            for name_ in out_names:
                column_name = f'{name_}_T({t_win})_S({s_win})_EvLim({ev_lim})'
                results_dict[column_name] = []

    for ti, t_win in enumerate(time_window):
        for si, s_win in enumerate(space_window):
            for i in range(df.shape[0]):
                print(f'Clust_T({t_win})_S({s_win}): {i} out of {df.shape[0]}')
                if i >= ev_lim:
                    candidate_events = ut.CandidateEventsTS(df.loc[:i,:].to_numpy(), t_win, s_win*1000, Mc)
                    data = candidate_events.filter_data()
                    if data.size > 0:
                        if data.shape[0] < ev_lim:
                            last_df = df_mc[df_mc['Time[d.s]'] < data[0,4]]
                            data = np.concatenate((last_df.iloc[-(ev_lim-data.shape[0]):,:].to_numpy(), data))
                        if ti + si == 0:
                            results_dict['Time[d.s]'].append(df['Time[d.s]'].values[i])
                            results_dict['Index'].append(df['Event_ID'].values[i])
                        res = ut.Clustering_analysis(data, b=b, fd=fd, p=p, event_lag=event_lag, time_lag=time_lag, c=c)
                        for n, name_ in enumerate(out_names):
                            column_name = f'{name_}_T({t_win})_S({s_win})_EvLim({ev_lim})'
                            results_dict[column_name].append(res[n])
                    else:
                        if ti + si == 0:
                            results_dict['Time[d.s]'].append(df['Time[d.s]'].values[i])
                            results_dict['Index'].append(df['Event_ID'].values[i])
                        for n, name_ in enumerate(out_names):
                            column_name = f'{name_}_T({t_win})_S({s_win})_EvLim({ev_lim})'
                            results_dict[column_name].append(0)

    return pd.DataFrame(results_dict)


def Volume_CH_event_based(df, time_window, space_window, ev_lim, Mc=None):
    if Mc:
        df_mc = df[df['Magnitude'] >= Mc]
    else:
        df_mc = df

    if not space_window:
        space_window = ['inf']

    results_dict = {'Index': [], 'Time[d.s]': []}
    for ti, t_win in enumerate(time_window):
        for si, s_win in enumerate(space_window):
            column_name = f'logV_T({t_win})_S({s_win})_EvLim({ev_lim})'
            results_dict[column_name] = []
            for i in range(df.shape[0]):
                print(f'logV_T({t_win})_S({s_win}): {i} out of {df.shape[0]}')
                if i >= ev_lim:
                    candidate_events = ut.CandidateEventsTS(df.loc[:i,:].to_numpy(), t_win, s_win*1000, Mc)
                    data = candidate_events.filter_data()
                    if data.size > 0:
                        if data.shape[0] < ev_lim:
                            last_df = df_mc[df_mc['Time[d.s]'] < data[0,4]]
                            data = np.concatenate((last_df.iloc[-(ev_lim-data.shape[0]):,:].to_numpy(), data))
                        if ti + si == 0:
                            results_dict['Time[d.s]'].append(df['Time[d.s]'].values[i])
                            results_dict['Index'].append(df['Event_ID'].values[i])
                        results_dict[column_name].append(ut.Vol_ConvexHull(data))
                    else:
                        if ti + si == 0:
                            results_dict['Time[d.s]'].append(df['Time[d.s]'].values[i])
                            results_dict['Index'].append(df['Event_ID'].values[i])
                        results_dict[column_name].append(0)

    return pd.DataFrame(results_dict)

def Strain_event_based(df, time_window, space_window, ev_lim, Mc=None, strain_model = 'Kostrov', Mu = 35*10**(9)):
    if Mc:
        df_mc = df[df['Magnitude'] >= Mc]
    else:
        df_mc = df

    if not space_window:
        space_window = ['inf']

    results_dict = {'Index': [], 'Time[d.s]': []}
    for ti, t_win in enumerate(time_window):
        for si, s_win in enumerate(space_window):
            column_name = f'logKs_T({t_win})_S({s_win})_EvLim({ev_lim})'
            results_dict[column_name] = []
            for i in range(df.shape[0]):
                print(f'logKs_T({t_win})_S({s_win}): {i} out of {df.shape[0]}')
                if i >= ev_lim:
                    candidate_events = ut.CandidateEventsTS(df.loc[:i,:].to_numpy(), t_win, s_win*1000, Mc)
                    data = candidate_events.filter_data()
                    if data.size > 0:
                        if data.shape[0] < ev_lim:
                            last_df = df_mc[df_mc['Time[d.s]'] < data[0,4]]
                            data = np.concatenate((last_df.iloc[-(ev_lim-data.shape[0]):,:].to_numpy(), data))
                        if ti + si == 0:
                            results_dict['Time[d.s]'].append(df['Time[d.s]'].values[i])
                            results_dict['Index'].append(df['Event_ID'].values[i])
                        results_dict[column_name].append(ut.Strain(data, t_win = t_win, strain_model=strain_model, Mu = Mu))
                    else:
                        if ti + si == 0:
                            results_dict['Time[d.s]'].append(df['Time[d.s]'].values[i])
                            results_dict['Index'].append(df['Event_ID'].values[i])
                        results_dict[column_name].append(0)

    return pd.DataFrame(results_dict)

def Scaled_E_event_based(df, time_window, space_window, ev_lim, Mc=None):
    if Mc:
        df_mc = df[df['Magnitude'] >= Mc]
    else:
        df_mc = df

    if not space_window:
        space_window = ['inf']

    results_dict = {'Index': [], 'Time[d.s]': []}
    for ti, t_win in enumerate(time_window):
        for si, s_win in enumerate(space_window):
            column_name = f'EngIdx_T({t_win})_S({s_win})_EvLim({ev_lim})'
            results_dict[column_name] = []
            for i in range(df.shape[0]):
                print(f'EngIdx_T({t_win})_S({s_win}): {i} out of {df.shape[0]}')
                if i >= ev_lim:
                    candidate_events = ut.CandidateEventsTS(df.loc[:i,:].to_numpy(), t_win, s_win*1000, Mc)
                    data = candidate_events.filter_data()
                    if data.size > 0:
                        if data.shape[0] < ev_lim:
                            last_df = df_mc[df_mc['Time[d.s]'] < data[0,4]]
                            data = np.concatenate((last_df.iloc[-(ev_lim-data.shape[0]):,:].to_numpy(), data))
                        if ti + si == 0:
                            results_dict['Time[d.s]'].append(df['Time[d.s]'].values[i])
                            results_dict['Index'].append(df['Event_ID'].values[i])
                        results_dict[column_name].append(ut.Scaled_energy(data, t_win))
                    else:
                        if ti + si == 0:
                            results_dict['Time[d.s]'].append(df['Time[d.s]'].values[i])
                            results_dict['Index'].append(df['Event_ID'].values[i])
                        results_dict[column_name].append(0)

    return pd.DataFrame(results_dict)
# %%
