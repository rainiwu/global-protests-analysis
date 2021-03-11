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
