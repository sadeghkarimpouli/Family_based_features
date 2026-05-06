#
#
#
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_feature(df, df_feature, save_path=None, log_scale = False, Scatter = False, lab = False):
    if lab:
        Time_date_df = df['Time[d.s]']
        Time_date = df_feature['Time[d.s]']
    else:
        Time_date_df = df['Time[d.s]'].apply(lambda x: pd.Timestamp.fromordinal(int(x)) + pd.Timedelta(days=x % 1))
        Time_date = df_feature['Time[d.s]'].apply(lambda x: pd.Timestamp.fromordinal(int(x)) + pd.Timedelta(days=x % 1))

    # Extract column prefixes
    prefix_to_columns = {}
    # df_feature_clm = []
    df_feature_clm = [clm for clm in df_feature.columns if clm not in ['Index', 'Time[d.s]']]
    for col in df_feature_clm:
        prefix = col.split('_')[0]
        if prefix not in prefix_to_columns:
            prefix_to_columns[prefix] = []
        prefix_to_columns[prefix].append(col)

    for prefix, columns in prefix_to_columns.items():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(Time_date_df, df['Magnitude'], s=np.exp(df['Magnitude'])/1, edgecolors='k', alpha=0.6)
        ax1 = ax.twinx()
        colors = cm.rainbow(np.linspace(0, 1, len(columns)))
        for i, name in enumerate(columns):
            if Scatter:
                ax1.scatter(Time_date, df_feature[name], s=0.5, color = colors[i], label = name)
            else:
                ax1.plot(Time_date, df_feature[name], c = colors[i], label = name)
    
        ax.set_xlabel('Date')
        ax.set_ylabel('Magnitude (Mw)')
        ax1.set_ylabel('Features')
        if log_scale:
            ax1.set_yscale('log')
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(f'{save_path}/{prefix}.png', dpi=300)
            plt.savefig(f'{save_path}/{prefix}.pdf', dpi=300)
        # plt.show()

# %%
