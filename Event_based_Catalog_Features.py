#
#
#
# %%
import pandas as pd
from util import Event_based_Features_functions as eff
from util.Plot_features import plot_feature

# -------------------------
# General settings 
# -------------------------
catalog_file_path = "Data/Kahramanmaras_Cat.csv" 
save_path = 'Extracted_features' 
df = pd.read_csv(catalog_file_path)
time_window = [15, 30]      # in days
space_window = [17, 33]     # in km; use [] when no window is used, meaning all events are used
Mc = 1.1                    # Completeness magnitude; set to None if not used
ev_lim = 20                 # Minimum number of events in each step to compute reliable features; set to None if not used

# -------------------------
# Feature switches
# Set a feature to True to compute it, or False to skip it.
# Example: to compute only b-value, set all other features to False.
# !!! Note: some features contain more parameters to be set. Before computing refer to the target feature down below for more details.
# -------------------------
feature_flags = {
    "event_rate": True,
    "moment_rate": False,
    "b_value": False,
    "correlation_integral": False,
    "inter_event_time": False,
    "inter_event_space": False,
    "clustering": False,
    "convex_hull_volume": False,
    "kostrov_strain": False,
}

# -------------------------
# Feature calculations
# -------------------------
# # # Event rate ----------
if feature_flags["event_rate"]:
    df_n = eff.Event_rate_event_based(
        df,
        time_window=time_window,
        space_window=space_window,
        ev_lim=ev_lim,
        Mc=Mc,
    )
    plot_feature(df, df_n, save_path, Scatter=True)
    df_n.to_csv(save_path + "/n.csv", index=False)

# Moment rate
if feature_flags["moment_rate"]:
    df_logM = eff.Moment_rate_event_based(
        df,
        time_window=time_window,
        space_window=space_window,
        ev_lim=ev_lim,
        Mc=Mc,
    )
    plot_feature(df, df_logM, save_path, Scatter=False)
    df_logM.to_csv(save_path + "/logM.csv", index=False)

# # # b-value -------------
# Set b_flag to specify the method for b-value calculation. Options are:
# "1": b-value of 1 for all events
# "b": Maximum-likelihood estimate (MLE) of b (Kagan 2002)
# "bp": B-positive (van der Elst 2021)
# "b_TGR": Tapered Gutenberg_Richter (TGR) distribution (Kagan 2002) 
b_flag="bp"

if feature_flags["b_value"]:
    df_b = eff.b_value_event_based(
        df,
        time_window=time_window,
        space_window=space_window,
        b_flag=b_flag,  
        ev_lim=ev_lim,  
        Mc=Mc,
    )
    plot_feature(df, df_b, save_path, Scatter=False)
    df_b.to_csv(save_path + f"/{b_flag}.csv", index=False)

# # # Correlation integral ------
# Set distance parameters for correlation integral calculation
r_max = 50 # in km: Maximum distance for correlation integral calculation
num_steps = 3 # Number of steps out of r_max to define distance parameter

if feature_flags["correlation_integral"]:
    df_c = eff.Correlation_integral_event_based(
        df,
        time_window=time_window,
        space_window=space_window,
        r_max=r_max*1000,
        num_steps=num_steps,
        ev_lim=ev_lim,
        Mc=Mc,
    )
    plot_feature(df, df_c, save_path, Scatter=False)
    df_c.to_csv(save_path + "/c.csv", index=False)

# # # Inter-event time ------
if feature_flags["inter_event_time"]:
    df_IE_t = eff.InterEvent_t_event_based(
        df,
        time_window=time_window,
        space_window=space_window,
        ev_lim=ev_lim,
        Mc=Mc,
    )
    plot_feature(df, df_IE_t, save_path, Scatter=False)
    df_IE_t.to_csv(save_path + "/IEt.csv", index=False)

# # # Inter-event space ------
if feature_flags["inter_event_space"]:
    df_IE_s = eff.InterEvent_s_event_based(
        df,
        time_window=time_window,
        space_window=space_window,
        ev_lim=ev_lim,
        Mc=Mc,
    )
    plot_feature(df, df_IE_s, save_path, Scatter=False)
    df_IE_s.to_csv(save_path + "/IEs.csv", index=False)

# # # Clustering features ------
# This part contains modified portions of code from eqclustering
# by Mark Williams (University of Nevada, Reno),
# licensed under Apache License 2.0.

# Set parameters for clustering analysis. see Zaliapin and Ben-Zion (2013)
# Note: Use a high value for Minimum number of events, e.g., ev_lim=100, to compute reliable clustering features.
b=1             # b_value
fd=2.5          # Fractal dimention
p=0.5           # Weight parameter
event_lag=250   # Maximum number of events to look back for each event; set to None to use all previous events
time_lag=None   # Maximum time to look back for each event in seconds; set to None to use all previous events
c=6.5           # Clustering boundary log10(T*P). Use 'None' to let the code compute it based on the data 
                # in the current windows, which is not recommended.

if feature_flags["clustering"]:
    df_clust = eff.Clustering_event_based(
        df,
        time_window=time_window,
        space_window=space_window,
        ev_lim=ev_lim,  # Minimum number of events in each step
        Mc=Mc,
        b=b,
        fd=fd,
        p=p,
        event_lag=event_lag,
        time_lag=time_lag,
        c=c
    )
    plot_feature(df, df_clust, save_path, Scatter=False)
    df_clust.to_csv(save_path + "/clust.csv", index=False)

# # # Convex hull volume ------
if feature_flags["convex_hull_volume"]:
    df_v = eff.Volume_CH_event_based(
        df,
        time_window=time_window,
        space_window=space_window,
        ev_lim=ev_lim,
        Mc=Mc,
    )
    plot_feature(df, df_v, save_path, Scatter=False)
    df_v.to_csv(save_path + "/logV.csv", index=False)

# # # Kostrov strain ------
# Set shear strength (Mu) for Kostrov strain calculation.
Mu = 35*10**(9)

if feature_flags["kostrov_strain"]:
    df_ks = eff.Strain_event_based(
        df,
        time_window=time_window,
        space_window=space_window,
        ev_lim=ev_lim,
        Mc=Mc,
        strain_model="Kostrov",
        Mu = Mu
    )
    plot_feature(df, df_ks, save_path, Scatter=False)
    df_ks.to_csv(save_path + "/logKs.csv", index=False)

# %%
