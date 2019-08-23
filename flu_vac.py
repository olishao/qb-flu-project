# FLU VACCINATION PROJECT CODE
# SUMMER 2019
# OLIVIA SHAO

#
# 0 Setup
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams["font.family"] = "Tex Gyre Pagella"
WANING_RATE = 0.025 # weekly waning rate
N_SEASONS = 22
SEASON_START_WEEK = 28 # mid-july, with week indices beginning at 0 \
# (actually the 29th week of the year)
T_TO_MAX = 2 # weeks it takes to reach max VE, after which waning begins

# 1997-2015 flu surveillance data
df9715 = pd.read_csv('data/WHO_NREVSS_Combined_prior_to_2015_16.csv', header=1)
# 2015-2019 flu surveillance data (clinical labs)
df1519c = pd.read_csv('data/WHO_NREVSS_Clinical_Labs.csv', header=1)
# 2015-2019 flu surveillance data (public health labs)
df1519ph = pd.read_csv('data/WHO_NREVSS_Public_Health_Labs.csv', header=1)
# 1997-2015 influenza like illness data
df_ili = pd.read_csv('data/ILINet.csv', header=1)

#
# 1 Cleaning data
#
df9715['TOTAL POSITIVE'] = df9715.iloc[:, -7:].sum(axis=1)
df9715 = df9715.drop(['REGION TYPE', 'REGION', 'A (2009 H1N1)', 'A (H1)', \
    'A (H3)', 'A (Subtyping not Performed)', 'A (Unable to Subtype)', \
    'B', 'H3N2v'], axis=1)
df1519c = df1519c.drop(['REGION TYPE', 'REGION', 'PERCENT A', 'PERCENT B'], \
    axis=1)
df1519ph = df1519ph.drop(['REGION TYPE', 'REGION'], axis=1)
df_ili = df_ili.drop(['REGION TYPE', 'REGION', 'AGE 0-4', 'AGE 25-49', \
    'AGE 25-64', 'AGE 5-24', 'AGE 50-64', 'AGE 65', 'NUM. OF PROVIDERS'], \
    axis=1)
df_ili['% WEIGHTED ILI'] = df_ili['% WEIGHTED ILI'] / 100
df_ili['% UNWEIGHTED ILI'] = df_ili['% UNWEIGHTED ILI'] / 100
## combine 2015-2019 clinical and public health lab data
df1519c.rename(columns={'TOTAL SPECIMENS':'TOTAL C'}, inplace=True)
df1519ph.rename(columns={'TOTAL SPECIMENS':'TOTAL PH'}, inplace=True)
df1519c['TOTAL POS C'] = df1519c.loc[:, 'TOTAL A':'TOTAL B'].sum(axis=1)
df1519ph['TOTAL POS PH'] = df1519ph.iloc[:, -7:].sum(axis=1)
df1519c = df1519c.drop(['TOTAL A', 'TOTAL B', 'PERCENT POSITIVE'], axis=1)
df1519ph = df1519ph.drop(['A (2009 H1N1)', 'A (H3)', \
    'A (Subtyping not Performed)', 'B', 'BVic', 'BYam', 'H3N2v'], axis=1)
df1519 = pd.merge(df1519c, df1519ph, how='left', on=['YEAR','WEEK'])
df1519['TOTAL POSITIVE'] = df1519['TOTAL POS C'] + df1519['TOTAL POS PH']
df1519['TOTAL'] = df1519['TOTAL C'] + df1519['TOTAL PH']
df1519['PCT POSITIVE'] = df1519['TOTAL POSITIVE'] / df1519['TOTAL']
df1519 = df1519.drop(['TOTAL C', 'TOTAL POS C', 'TOTAL PH', 'TOTAL POS PH'], \
    axis=1)
## combine 1997-2015 and 2015-2019 data
df9715.rename(columns={'TOTAL SPECIMENS':'TOTAL'}, inplace=True)
df9715['PERCENT POSITIVE'] = df9715['PERCENT POSITIVE'] / 100
df9715.rename(columns={'PERCENT POSITIVE':'PCT POSITIVE'}, inplace=True)
df = pd.concat([df9715, df1519], sort=False).reset_index()

def seasons(df, start_week):
    '''
    adds columns in df for the flu season (0-21) and week in season (0-53)
    input:
        df: dataframe
        start_week (int): week of the year defining start of season 
            (using zero-based indexing)
    '''
    season_lb, season_ub = 0, 52 - (40 - start_week) # bounding week indices for first season
    for n in range(N_SEASONS):
        df.loc[season_lb:season_ub, 'SEASON'] = n # zero-based indexing for seasons
        if n == 0: # seasons/years with 53 weeks
            season_lb += 53 - (40 - start_week) # compensate for shorter first season
            season_ub += 52 
        elif n == 5 or n == 10 or n == 16:
            season_lb += 52
            season_ub += 53
        elif n == 6 or n == 11 or n == 17:
            season_lb += 53
            season_ub += 52
        else: # normal seasons/years
            season_lb += 52
            season_ub += 52
    df.loc[df['WEEK'] >= start_week, 'WEEK IN SEASON'] = \
        df.loc[df['WEEK'] >= start_week, 'WEEK'] - start_week
    df.loc[(df['WEEK'] < start_week) & (df['SEASON'].isin([0, 6, 11, 17])), \
        'WEEK IN SEASON'] = df.loc[df['WEEK'] < start_week, 'WEEK'] + 53 \
        - start_week
    df.loc[(df['WEEK'] < start_week) & (~df['SEASON'].isin([0, 6, 11, 17])), \
        'WEEK IN SEASON'] = df.loc[df['WEEK'] < start_week, 'WEEK'] + 52 \
        - start_week
## define flu seasons in flu and ili dataframes for analysis and graphing purposes
seasons(df, SEASON_START_WEEK)
seasons(df_ili, SEASON_START_WEEK)

def scale_inc(df, var):
    '''
    scales the incidence at each week by the total seasonal incidence
    inputs:
        df: dataframe
        var (str): incidence variable to be scaled
    '''
    for n in range(N_SEASONS):
        season_inc = df.loc[df['SEASON'] == n, var]
        if len(season_inc) < 52: # if season is incomplete (e.g. 1997-1998 season)
            pass
        else:
            df.loc[df['SEASON'] == n, ('SCALED '+ var)] = \
                season_inc / season_inc.sum(axis=0) # create new scaled variable in df
## create scaled variables for flu incidence
scale_inc(df, 'TOTAL POSITIVE')
scale_inc(df_ili, 'ILITOTAL')

## stats for peak of flu season and peak ili
def get_stats(df, var, rv=False):
    '''
    gives mean peak value of var and mean week for peak var across all seasons
    inputs:
        df: dataframe
        var (str): variable in df
        rv (bool): if True, function returns tuple of lists for peak var 
            and peak week
    '''
    peak_var = []
    peak_week = []
    for n in range(N_SEASONS):
        var_max = max(df.loc[df['SEASON'] == n, var])
        if ~np.isnan(var_max):
            peak_var.append(var_max)
        week = df.loc[df[var] == var_max, 'WEEK IN SEASON']
        if len(week) > 1:
            peak_week.append(int(np.mean(week)))
        elif len(week) == 0:
            pass
        else:
            peak_week.append(int(week))
    mean_peak_week = np.mean(peak_week)
    print("mean peak {} is {}, with standard deviation {}".format(var, \
        np.mean(peak_var), np.std(peak_var)))
    if mean_peak_week > 52.143 - (SEASON_START_WEEK + 1):
        print("mean peak week is {}, with standard deviation {}".format( \
            SEASON_START_WEEK + 1 + mean_peak_week - 52, np.std(peak_week)))
    else:
        print("mean peak week is {}, with standard deviation {}".format( \
            SEASON_START_WEEK + 1 + mean_peak_week, np.std(peak_week)))
    if rv:
        return peak_var, peak_week
    else:
        return None

#
# 2 Functions for model
#
def vac_eff(t, t_vac, waning_rate, t_max, max_ve=1):
    '''
    gives simulated distribution for flu vaccine effectiveness (VE) 
    after vaccination
    inputs:
        t (array of integers): represents weeks
        t_vac (int): week of flu vaccination
        waning_rate (float): waning rate
        t_max (int): time until maximum VE is reached
        max_ve (float): maximum VE on a scale from 0 to 1
    '''
    t = t.astype(float)
    dist = np.piecewise(t, [t < t_max, t >= t_max], [lambda t: \
        (np.exp((np.log(2) / t_max) * t) - 1) * max_ve, lambda t: \
        np.exp(-waning_rate * (t - t_max)) * max_ve])
    unvac_weeks = np.array([0] * t_vac)
    dist = np.concatenate([unvac_weeks, dist])
    dist = dist[0:len(t)]
    return dist

# def flu_foi(t):
#     '''
#     gives simulated distribution for a flu force of infection curve
#     input:
#         t (array of integers): represents days/weeks
#     '''
#     a = 1/2
#     k = 1/30
#     p = np.pi
#     b = 0.5
#     return a*np.cos(k*t + p) + b

def season_inc(df, season, var):
    '''
    returns incidence distribution for a season    
    inputs:
        df: dataframe
        season (int): flu season of interest
        var (str): variable in df representing flu incidence
    '''
    inc_dist = []
    weeks = (df['SEASON'] == season).sum() # weeks in season
    for week in range(weeks):
        if season == 0:
            week += 40 - SEASON_START_WEEK
        inc_dist.append(df.loc[(df['WEEK IN SEASON'] == week) & \
            (df['SEASON'] == season), var].item())
    return inc_dist

def red_dist_inc(inc, ve_distributions=None, waning_rate=None, max_ve=1):
    '''
    returns x and y to graph distribution for flu reduction
    inputs: 
        inc: flu incidence distribution
        ve_distributions (list of arrays of floats): VE distributions
        waning_rate (float): waning rate
        max_ve (float): maximum VE on a scale from 0 to 1
    '''
    assert((ve_distributions and not isinstance(waning_rate, float)) or \
        (isinstance(waning_rate, float) and not ve_distributions))
    weeks_tot = np.arange(0, len(inc))
    pct_flu_red = []
    for week in weeks_tot:
        if not ve_distributions:
            ve = vac_eff(weeks_tot, week, waning_rate, T_TO_MAX, max_ve)
        elif not waning_rate:
            ve = ve_distributions[week]
        pct_flu_red.append(sum(ve * inc))
    red_max = max(pct_flu_red)
    return np.arange(0, len(inc)), pct_flu_red

def red_dist_df(df, var, waning_rate, max_ve=1):
    '''
    returns GroupBy object to graph time series for flu reduction/protection
    (excludes incomplete seasons)
    inputs:
        df: dataframe
        var (str): variable in df representing flu incidence
        waning_rate (float): waning rate
        max_ve (float): maximum VE on a scale from 0 to 1
    '''
    for season in range(N_SEASONS):
        inc = season_inc(df, season, var)
        weeks_tot = np.arange(0, len(inc))
        for week in weeks_tot:
            ve = vac_eff(weeks_tot, week, waning_rate, T_TO_MAX, max_ve)
            df.loc[(df['WEEK IN SEASON'] == week) & (df['SEASON'] \
                == season), 'FLU REDUCTION'] = sum(ve * inc)
    new_df = df.set_index('WEEK IN SEASON')
    return new_df.groupby('SEASON')['FLU REDUCTION']

def get_ve_dists(waning_rate, inc_dist, max_ve=1):
    '''
    returns list of 52 distributions representing the vaccine effectiveness 
    curves for vaccination at each week of the year
    inputs:
        waning_rate (float): waning rate
        inc_dist (list of floats): incidence distribution
        max_ve (float): maximum VE on a scale from 0 to 1
    '''
    weeks_tot = np.arange(0, len(inc_dist))
    ve_distributions = []
    for week in weeks_tot:
        ve_distributions.append(vac_eff(weeks_tot, week, waning_rate, T_TO_MAX, \
            max_ve))
    return ve_distributions

def get_inc_dists(df, var):
    '''
    returns list of 22 distributions representing incidence in each season
    inputs:
        df: dataframe
        var (str): variable in df representing flu incidence
    '''
    inc_distributions = []
    for season in range(N_SEASONS):
        inc = season_inc(df, season, var)
        inc_distributions.append(inc)
    return inc_distributions

def avg_dist(df, var):
    '''
    gives average/aggregate distribution for variable var across all seasons
    inputs:
        df: dataframe
        var (str): variable in df
    '''
    mean_dist = []
    for week in range(52):
        mean_dist.append(np.mean(df.loc[df['WEEK IN SEASON'] == week, var]))
    return mean_dist

def get_rec_week(df, var):
    '''
    returns optimal week for flu vaccination
    inputs:
        df: dataframe
        var (str): variable in df representing flu incidence
    '''
    avg_inc = avg_dist(df, var)
    flu_red_dist = red_dist_inc(avg_inc, waning_rate=WANING_RATE)[1]
    red_max = max(flu_red_dist)
    peak_week = flu_red_dist.index(red_max)
    return peak_week
# week of the season for optimal protection from vaccination assuming 2.5% weekly waning
OPT_T_VAC = get_rec_week(df, 'SCALED TOTAL POSITIVE') # returns week in terms of 'WEEK IN SEASON'
# week of the year for optimal protection from vaccination
REC_WEEK = OPT_T_VAC + SEASON_START_WEEK + 1 # add 1 to account for difference in indexing
# week of the season with highest flu activity on average
avg_inc = avg_dist(df, 'SCALED TOTAL POSITIVE')
PEAK_FLU_WEEK = avg_inc.index(max(avg_inc)) + SEASON_START_WEEK + 1 - 52

#
# 3 Functions for figures
#
def format_axes(ax, label_fontsize, tick_fontsize, ve=False):
    '''
    formats axes for plots with time on the x axis; used for plotting VE,
    incidence, and flu reduction/protection
    inputs:
        ax (matplotlib Axes object): axes to be formatted
        label_fontsize (int): font size for labels
        tick_fontsize (int): font size for tick labels
        ve (bool): True if plotting VE, removes month tick labels
    '''
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.set_xticks(np.arange(4.345 / 2, 47.8 + 4.345 / 2, step=4.345)) # 4.345 weeks on average in a month
    ax.set_xticks(np.arange(0, 53, step=4.345), minor=True)
    ax.set_xticklabels([])
    for label in ([ax.yaxis.label, ax.xaxis.label]):
        label.set_fontsize(label_fontsize)
    if not ve:
        labels = calendar.month_abbr[7:13] + calendar.month_abbr[1:8]
        ax.set_xticklabels(labels, fontsize=tick_fontsize, minor=True)

def plot_ve(ax, ve_distribution, max_ve=1):
    '''
    plots vaccine effectiveness curve
    inputs:
        ax (matplotlib Axes object): axes to be formatted
        ve_distribution (array of floats): VE distribution to be plotted
        max_ve (float): maximum VE on a scale from 0 to 1
    '''
    format_axes(ax, 20, 16, ve=True)
    if max_ve == 1:
        ax.set_ylim(-0.05, 1.1)
        ax.set_ylabel('Relative \n VE')
    else:
        ax.set_ylim(-0.05, max_ve + 0.05)
        ax.set_ylabel('VE')
    ax.set_xlabel('Time')
    ax.plot(np.arange(0, 52), ve_distribution, color='black')

def plot_inc(ax, inc_distributions, avg_inc, var, color=False):
    '''
    plots disease distribution curves for each season and the mean curve
    inputs:
        ax (matplotlib Axes object): axes to be formatted
        inc_distributions (list of lists of floats): seasonal incidence 
            distributions
        avg_inc (list of floats): average incidence distribution 
        var (str): variable in df representing flu incidence
        color (bool): plots seasonal incidence curves in color if True
    '''
    if 'SCALED' in var:
        ax.set_ylabel('Scaled \n Incidence')
    else:
        ax.set_ylabel('Incidence')
    ax.set_xlabel('Time')
    format_axes(ax, 20, 16)
    if color:
        for dist in inc_distributions:
            ax.plot(np.arange(0, len(dist)), dist, linewidth=0.3)
    else:   
        for dist in inc_distributions:
            ax.plot(np.arange(0, len(dist)), dist, color='gray', linewidth=0.3)
    ax.plot(np.arange(0, 52), avg_inc, color='black')

def plot_red(ax, avg_inc, ve_distributions, grouped, color=False):
    '''
    plots flu reduction/protection curves for each season and the mean curve
    inputs:
        ax (matplotlib Axes object): axes to be formatted
        avg_inc (list of floats): average incidence distribution 
        ve_distributions (list of arrays of floats): VE distributions
        grouped (Pandas GroupBy object): flu reduction by season
        color (bool): plots seasonal flu reduction curves in color if True
    '''
    ax.set_ylabel('Relative \n Protection')
    ax.set_xlabel('Time of Vaccination')
    format_axes(ax, 20, 16)
    ax.set_yticks(np.arange(0, 0.8, step=0.25))
    if color:
        for g in grouped:
            ax.plot(g[1], linewidth=0.3)
    else:   
        for g in grouped:
            ax.plot(g[1], color='gray', linewidth=0.3)
    x, y = red_dist_inc(avg_inc, ve_distributions=ve_distributions)
    ax.plot(x, y, color='black')

# three-panel plots
def plot_all(df, var, waning_rate, max_ve=1, color=False):
    '''
    plots vaccine effectiveness, scaled incidence, and flu reduction 
    inputs:
        df: dataframe
        var (str): variable in df representing flu incidence
        waning_rate (float): waning rate
        max_ve (float): maximum VE on a scale from 0 to 1
        color (bool): plots seasonal curves in color if True
    '''
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    avg_inc = avg_dist(df, var)
    ve_distributions = get_ve_dists(waning_rate, avg_inc, max_ve)
    inc_distributions = get_inc_dists(df, var)
    grouped = red_dist_df(df, var, waning_rate)
    # for dist in ve_distributions:
        # ax1.plot(np.arange(0, 52), dist, color='gray', linewidth=0.3)
    plot_ve(ax1, ve_distributions[0], max_ve)
    plot_inc(ax2, inc_distributions, avg_inc, var, color)
    plot_red(ax3, avg_inc, ve_distributions, grouped, color)
    plt.tight_layout()
    plt.show()
# plot using scaled measure of incidence
plot_all(df, 'SCALED TOTAL POSITIVE', WANING_RATE)
# plot_all(df_ili, 'SCALED ILITOTAL', WANING_RATE)

# individual plots
def plot_just(df, var, ve=False, inc=False, red=False):
    '''
    plots vaccine effectiveness, scaled incidence, or flu reduction
    inputs:
        df: dataframe
        var (str): variable in df representing flu incidence
        ve: plots VE distributions if True
        inc: plots incidence distributions if True
        red: plots flu reduction distributions if True
    '''
    fig, ax = plt.subplots() 
    avg_inc = avg_dist(df, var)
    if ve:
        ve_distributions = get_ve_dists(WANING_RATE, avg_inc)
        for dist in ve_distributions:
            plot_ve(ax, dist)
    if inc:
        inc_distributions = get_inc_dists(df, var)
        plot_inc(ax, inc_distributions, avg_inc, var, color=True)
    if red:
        ve_distributions = get_ve_dists(WANING_RATE, avg_inc)
        grouped = red_dist_df(df, var, WANING_RATE)
        plot_red(ax, avg_inc, ve_distributions, grouped)
    plt.tight_layout()
    plt.show()
plot_just(df, 'TOTAL POSITIVE', inc=True) # plotting positive flu cases in each season
# plot_just(df, 'SCALED TOTAL POSITIVE', ve=True)
# plot_just(df, 'SCALED TOTAL POSITIVE', inc=True)
# plot_just(df, 'SCALED TOTAL POSITIVE', red=True)

def plot_flu_red(df, var):
    '''
    plots mean flu reduction/protection curve for varying waning rates
    inputs:
        df: dataframe
        var (str): variable in df representing flu incidence
    '''
    distributions = []
    waning_rates = np.linspace(0, 0.1, 6)
    avg_inc = avg_dist(df, var)
    x = np.arange(0, 52)
    a = 0.2
    for wr in waning_rates:
        dist = red_dist_inc(avg_inc, waning_rate=wr)[1]
        distributions.append(dist)
    lines = LineCollection([np.column_stack([x, dist]) for dist in \
        distributions], cmap='viridis_r')
    lines.set_array(waning_rates)
    fig, ax = plt.subplots()
    if 'SCALED' in var:
        ax.set_ylim(0, 1.025)
    else:
        ax.set_ylim(0, max(distributions[0]))
    ax.set_xlim(0, 52)
    ax.set_ylabel('Relative Protection')
    ax.set_xlabel('Time of Vaccination')
    format_axes(ax, 24, 18)
    ax.tick_params(axis="x", length=7,  which='major')
    ax.add_collection(lines)
    ax.yaxis.label.set_fontsize(24)
    ax.xaxis.label.set_fontsize(24)
    axins = inset_axes(ax, width="5%", height="50%", loc='upper left', \
        bbox_to_anchor=(0.75, -0.05, 1, 1), bbox_transform=ax.transAxes,)
    cb = fig.colorbar(lines, cax=axins)
    cb.set_label('Waning Rate', fontsize=22, labelpad=10)
    cb.set_ticks(np.linspace(0, 0.1, 6))
    cb.ax.tick_params(labelsize=16)
    plt.tight_layout()
    plt.show()
plot_flu_red(df, 'SCALED TOTAL POSITIVE')
# plot_flu_red(df_ili, 'SCALED ILITOTAL')

def opttime_plot(df, var, label=None, fmt=None):
    '''
    plots vaccination week of max protection for range of waning rates
    inputs:
        df: dataframe
        var (str): variable in df representing flu incidence
        label (str): label for legend
        fmt (str): formatting string for plotting the curve
    '''
    week_max_red = []
    waning_rates = np.linspace(0, 0.1)
    avg_inc = avg_dist(df, var)
    for wr in waning_rates:
        flu_red_dist = red_dist_inc(avg_inc, waning_rate=wr)[1]
        week_max_red.append(flu_red_dist.index(max(flu_red_dist)))
    if not label:
        plt.plot(waning_rates, week_max_red, color='black')
    else:
        if fmt:
            plt.plot(waning_rates, week_max_red, fmt, label=label)
        else:
            plt.plot(waning_rates, week_max_red, color='black', label=label)
    plt.ylim([-1, 21.75])
    plt.xlim([-0.0025, 0.1025])
    ax = plt.axes()
    ax.yaxis.grid(color='#d3d3d3')
    ax.xaxis.grid(color='#d3d3d3')
    ax.set_yticks(np.arange(4.345 / 2, 21.75 + 4.345 / 2, step=4.345))
    ax.set_yticks(np.arange(0, 21.75, step=4.345), minor=True)
    ax.set_yticklabels([])
    labels = calendar.month_abbr[7:13]
    ax.set_yticklabels(labels, rotation=90, fontsize=20, minor=True, va='center')
    ax.set_xticks(np.arange(0, 0.11, step=0.01))
    ax.tick_params(axis='y', length=7,  which='major')
    ax.tick_params(axis='x', labelsize=20)
    plt.ylabel('Optimal Week to Vaccinate', fontsize=24)
    plt.xlabel('Weekly Waning Rate', fontsize=24)
    if label:
        plt.legend(loc='best', fontsize=14, borderpad=0.5, borderaxespad=1)
    plt.tight_layout()
opttime_plot(df, 'SCALED TOTAL POSITIVE')
plt.show()
# opttime_plot(df, 'SCALED TOTAL POSITIVE', label='scaled positive flu tests used \n to measure flu incidence')
# opttime_plot(df_ili, 'SCALED ILITOTAL', label='scaled ILI visits used to \n measure flu incidence', fmt='--k')
# plt.show()

def prot_diff_seasonal(df, var, t_vac):
    '''
    plots added protection of vaccination at optimal week vs. week t_vac for 
    each season
    inputs:
        df: dataframe
        var (str): variable in df representing flu incidence
        t_vac: week of vaccination
    '''
    added_prot = [] 
    opttime = get_rec_week(df, var)
    for n in range(N_SEASONS):
        red_opttime = df.loc[(df['SEASON'] == n) & (df['WEEK IN SEASON'] \
            == opttime), 'FLU REDUCTION'].item()
        if ~np.isnan(red_opttime):
            red_cur = df.loc[(df['SEASON'] == n) & (df['WEEK IN SEASON'] \
                == t_vac), 'FLU REDUCTION'].item()
            added_prot.append(red_opttime - red_cur)
    plt.plot(np.arange(0, len(added_prot)), added_prot)
    plt.plot(np.arange(0, len(added_prot)), np.repeat(np.mean(added_prot), \
        len(added_prot)))
    plt.ylabel('Added Protection From Vaccinating \n End of November Instead of End of October')
    plt.xlabel('Season')
    plt.tight_layout()
    plt.show()
# prot_diff_seasonal(df, 'SCALED TOTAL POSITIVE', 14) 
# week 14 of the season = week 43 of the year = end of oct

def prot_diff_wr_dep(df, var, t_vac):
    '''
    plots waning rate vs. added benefit of vaccinating at optimal time as opposed
    to end of october
    inputs:
        df: dataframe
        var (str): variable in df representing flu incidence
        t_vac
    '''
    prot_ratio = []
    waning_rates = np.linspace(0, 0.1)
    avg_inc = avg_dist(df, var)
    for wr in waning_rates:
        flu_red_dist = red_dist_inc(avg_inc, waning_rate=wr)[1]
        red_opttime = max(flu_red_dist)
        red_cur = flu_red_dist[t_vac]
        prot_ratio.append(red_opttime / red_cur)
    plt.plot(waning_rates, prot_ratio, color='black')
    # ax.set_title('Vaccinating at Waning-Dependent \n Optimal Time vs. Week 43')
    plt.ylabel('Relative Protection', fontsize=24)
    plt.xlabel('Weekly Waning Rate', fontsize=24)
    ax = plt.axes()
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=20)
    ax.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()
prot_diff_wr_dep(df, 'SCALED TOTAL POSITIVE', 14) # week 14 of season = end of oct

def added_prot(df, var, opt_t_vac, t_vac):
    '''
    plots waning rate vs. added benefit of vaccinating at optimal time as opposed
    to end of october
    inputs:
        df: dataframe
        var (str): variable in df representing flu incidence
        t_vac
    '''
    added_prot = []
    waning_rates = np.linspace(0, 0.1)
    avg_inc = avg_dist(df, var)
    for wr in waning_rates:
        flu_red_dist = red_dist_inc(avg_inc, waning_rate=wr)[1]
        red_opttime = flu_red_dist[OPT_T_VAC]
        red_cur = flu_red_dist[t_vac]
        added_prot.append(red_opttime / red_cur)
    plt.plot(waning_rates, added_prot, color='black')
    # ax.set_title('Vaccinating at Week 48 vs. Week 43')
    plt.ylabel('Relative Protection', fontsize=24)
    plt.xlabel('Weekly Waning Rate', fontsize=24)
    ax = plt.axes()
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=20)
    plt.vlines(0.0111, min(added_prot), 1, linestyle='dashed', linewidth=0.5)
    ax.axhline(y=1, color='black', linewidth=0.5)
    ax.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()
added_prot(df, 'SCALED TOTAL POSITIVE', OPT_T_VAC, 14) # week 14 of the season = week 43 of the year = end of oct

def compare_ili_flu(df_ili, df):
    '''
    compares peak week of flu activity and optimal vaccination week using
    ili and using flu to model flu incidence
    inputs:
        df_ili: ili dataframe
        df: flu dataframe
    '''
    iliweeks = get_stats(df_ili, 'SCALED ILITOTAL', rv=True)[1]
    fluweeks = get_stats(df, 'SCALED TOTAL POSITIVE', rv=True)[1]
    plt.scatter(iliweeks, fluweeks, label='week of peak flu activity')
    optili = get_stats(df_ili, 'FLU REDUCTION', rv=True)[1] # 15-27
    optflu = get_stats(df, 'FLU REDUCTION', rv=True)[1] # 10-16
    plt.scatter(optili, optflu, label='optimal vaccination week')
    plt.ylim([0, 52])
    plt.xlim([0, 52])
    plt.ylabel('Confirmed Flu')
    plt.xlabel('ILI')
    plt.legend(loc='best')
    plt.show()
# compare_ili_flu(df_ili, df)
