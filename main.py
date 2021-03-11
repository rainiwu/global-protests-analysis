import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import plotly as pl
import plotly.express as px
import plotly.figure_factory as ff
import plotly.offline as pyo
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import pygal
from pygal.maps.world import COUNTRIES
# Set notebook mode to work in offline
pyo.init_notebook_mode()

def preprocess(data):
    '''
    Clean the data
    '''
    MM = data.copy()
    MM = MM.sort_values(['country','year'])

def viol_percentage_line_plot(data):
    '''
    Processes the input data and generates a line plot for the violence percentage of each reason.
    '''
    # Initializing dataframe
    df = data
    df = df[['protest', 'protesterviolence', 'reasons_labor', 'reasons_social', 'reasons_land', 'reasons_removal',
            'reasons_political', 'reasons_price', 'reasons_policebrutality']]    # Calculates percentage of violent protests for each reason
    reasons = ['reasons_labor', 'reasons_social', 'reasons_land', 'reasons_removal', 'reasons_political',
            'reasons_price', 'reasons_policebrutality']
    df_temp = pd.DataFrame()
    for reason in reasons:
        df_temp[reason] = [round((sum(df['protesterviolence'] & df[reason] > 0))/(sum(df[reason]))*100)]    # Reformats and renames the df
    df_percentage = df_temp.T
    df_percentage.rename(index = {'reasons_social':'social reforms',
                                'reasons_policebrutality':'police brutality',
                                'reasons_land':'property',
                                'reasons_political':'law & politics',
                                'reasons_labor':'labor rights',
                                'reasons_price':'inflation',
                                'reasons_removal':'anti-authoritarian'},
                        columns = {0:'percentage'},
                        inplace = True)
    df_per = df_percentage    # Plotting dimensions
    fig_dims = (12, 4)
    fig, ax = plt.subplots(figsize = fig_dims, dpi=190)
    mpl.style.use("bmh")
    # Plotting
    sns.color_palette('coolwarm')
    clrs = ['grey' if (x < max(df_per['percentage'])) else '#E36C55' for x in df_per['percentage'] ]    
    mybar = plt.bar(df_per.index, df_per['percentage'])
    for bar in mybar:
        bar.set_color('grey')
    mybar[6].set_color('#E36C55')
    mybar[1].set_color('#6A8BEF')    
    plt.xlabel('Reason', size = 14)
    plt.ylabel('Percentage', size = 14)
    plt.title('Percentage of Violent Protests For Each Reason', size = 16)
    plt.grid(False)

def kenya_line_plot(data):
    '''
    Processes the data and generates a line plot of the number of protests in Kenya from 1990 to 2020.
    '''
    df = data
    df_kenya = df[df['country'] == 'Kenya']
    df_kenya_yrs = df_kenya.groupby('year').sum()
    # Plot of number of protests over the years for Kenya
    kenya_line_plt = sns.lineplot(x = 'year', y = 'protest', data = df_kenya_yrs)
    plt.title('Number of Protests in Kenya from 1990-2020', size = 16)
    plt.ylabel('protest', size = 14)
    plt.xlabel('year', size = 14)

def kenya_bar_plot(data):
    '''
    Processes the data and generates a bar plot for the number of protests for each reason in Kenya 2015.
    '''
    df = data
    df_kenya = df[df['country'] == 'Kenya']
    df_kenya_rzns = df_kenya[['year', 'reasons_social', 'reasons_policebrutality', 'reasons_land',
                            'reasons_political', 'reasons_labor', 'reasons_price', 'reasons_removal']]
    df_kenya_2015 = df_kenya_rzns[df_kenya_rzns['year'] == 2015]
    # Creating list for iterable and names for new df
    reasons = ['reasons_labor', 'reasons_social', 'reasons_land', 'reasons_removal', 'reasons_political',
            'reasons_price', 'reasons_policebrutality']
    # Initializing empty df
    df_temp = pd.DataFrame()
    for reason in reasons:
        df_temp[reason] = [sum(df_kenya_2015[reason])]
    df_reasons = df_temp.T
    df_reasons.rename(index = {'reasons_social':'social reforms',
                                'reasons_policebrutality':'police brutality',
                                'reasons_land':'property',
                                'reasons_political':'law & politics',
                                'reasons_labor':'labor rights',
                                'reasons_price':'inflation',
                                'reasons_removal':'anti-authoritarian'},
                    columns = {0:'Total Protests'},
                    inplace = True)
    # Plotting dimensions
    fig_dims = (12, 4)
    fig, ax = plt.subplots(figsize = fig_dims)    # Plotting
    values = df_reasons['Total Protests']
    clrs = ['gray' if (x < max(values)) else '#E36C55' for x in values]    
    violence_plot = sns.barplot(x = df_reasons.index, y = 'Total Protests', data = df_reasons, ax = ax, palette = clrs)    
    for p in violence_plot.patches:
        height = int(p.get_height())
        ax.annotate('{}'.format(height), xy = (p.get_x() + p.get_width() / 2, height), xytext = (0, 2), textcoords = 'offset points', ha = 'center', va = 'bottom')    
    plt.xlabel('Reason', size = 14)
    plt.ylabel('Occurences', size = 14)
    plt.title('Percentage of Violent Protests For Each Reason', size = 16)
