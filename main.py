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
    start_df = MM[['startday', 'startmonth', 'startyear']].copy()
    start_df.columns = ["day", "month", "year"]
    start_df = pd.to_datetime(start_df)
    end_df = MM[['endday', 'endmonth', 'endyear']].copy()
    end_df.columns = ["day", "month", "year"]
    end_df = pd.to_datetime(end_df)
    endIdx = ~end_df.isnull()
    startIdx = ~start_df.isnull()
    validIdx = endIdx & startIdx
    MM['start'] = start_df
    MM['end'] = end_df

    MM['protest_time'] = MM.end - MM.start + pd.Timedelta(days=1)
    reasons_dict = {'labor wage dispute':'labor',
     'land farm issue':'land',
     'police brutality': 'policebrutality',
     'political behavior, process': 'political',
     'price increases, tax policy': 'price',
     'removal of politician':'removal',
     'social restrictions':'social'}

    reasons = {'labor','land','policebrutality','political','price','removal','social','other'}

    MM = MM.replace(reasons_dict)
    for resn in reasons:
        MM['reasons_'+resn] = 0
    responses = {'accomodation',
     'arrests',
     'beatings',
     'crowd_dispersal',
     'ignore',
     'killings',
     'shootings',
     'other'}

    MM = MM.replace({'crowd dispersal':'crowd_dispersal'})
    for resp in responses:
        MM['responses_'+resp] = 0

    for i in MM.index:
        otherFlag = 1
        for fld in ['protesterdemand1','protesterdemand2', 'protesterdemand3', 'protesterdemand4']:
            if pd.isnull(MM[fld][i]) or MM[fld][i]=='.':
                pass
            else:
                otherFlag = 0
                MM.loc[i,'reasons_'+MM[fld][i]] = 1
        if otherFlag:
            MM.loc[i,'reasons_other'] = MM['protest'][i]

    for i in MM.index:
        otherFlag = 1
        for fld in ['stateresponse1', 'stateresponse2', 'stateresponse3', 'stateresponse4', 'stateresponse5', 'stateresponse6', 'stateresponse7']:
            if pd.isnull(MM[fld][i]) or MM[fld][i]=='.':
                pass
            else:
                otherFlag = 0
                MM.loc[i,'responses_'+MM[fld][i]] = 1
        if otherFlag:
            MM.loc[i,'responses_other'] = MM['protest'][i]
    MM['protesterviolence'] = MM['protesterviolence'].fillna(int(0)).astype(np.int64)
    # # Or
    # MM = MM[~MM['protesterviolence'].isnull()]

    MM['protest_time'] = MM['protest_time'].dt.days.fillna(int(0)).astype(np.int64)
    # # Or
    # MM = MM[~MM['protest_time'].isnull()]
    # MM['protest_time'] = MM['protest_time'].dt.days.astype(np.int64)

    MM['violent_response'] = MM['responses_beatings'] | MM['responses_killings'] | MM['responses_shootings']
    MM['success'] = MM['responses_accomodation'].copy()

    MM['violence_both'] = MM['violent_response'] | MM['protesterviolence']
    MM['violent_protest_time'] = MM['protesterviolence']*MM['protest_time']
    MM = MM.drop(['id', 'ccode', 'region', 'protestnumber', 'start', 'end',
           'startday', 'startmonth', 'startyear', 'endday', 'endmonth', 'endyear',
           'location', 'participants_category',
           'participants', 'protesteridentity', 'protesterdemand1',
           'protesterdemand2', 'protesterdemand3', 'protesterdemand4',
           'stateresponse1', 'stateresponse2', 'stateresponse3', 'stateresponse4',
           'stateresponse5', 'stateresponse6', 'stateresponse7', 'sources', 'notes'], axis=1)
    MM.to_csv('main_data.csv')

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

def duration_and_violence(main_data):    
    # process the input file
    MM = main_data
    violence = MM.sort_values(['country', 'year'])
    violence = violence.sort_values(['protest_time'])
    violence = violence.drop(['reasons_political', 'reasons_labor', 'reasons_price',
                              'reasons_land', 'reasons_policebrutality', 'reasons_removal',
                              'reasons_social', 'reasons_other', 'responses_crowd_dispersal',
                              'responses_killings', 'responses_beatings', 'responses_shootings',
                              'responses_arrests', 'responses_ignore', 'responses_accomodation',
                              'responses_other', 'violent_response', 'success'], axis=1)    
    violence = violence.sort_values(['protest_time']).drop(['Unnamed: 0', 'ccode',
                                                            'country', 'year',
                                                            'protest', 'protesterviolence', 'violent_protest_time'], axis=1)    
    violence0 = violence.copy()    

    # More than 10 days
    ten_days_more = violence0[violence0.protest_time_interval == 'More than 10 days']
    ten_days_more = ten_days_more.groupby(['violence']).size().reset_index(name='total_count')    

    # piechart for more than ten days
    fig, ax = plt.subplots(figsize=(15, 10))
    labels = ten_days_more['violence']
    percentages = ten_days_more['total_count']
    color = plt.cm.coolwarm([0.2, 0.68])
    ax.pie(percentages, explode=[0.5, 0], labels=labels,
           autopct='%.1f%%',
           shadow=False, startangle=90, colors=color,
           pctdistance=0.8, labeldistance=0.5, radius=10.5)
    ax.axis('equal')
    ax.set_title("More than 10 days")    
    ax.legend(frameon=False, bbox_to_anchor=(1.5, 0.8))    
    fig.show()

    # Within one day
    one_day = violence0[violence0.protest_time_interval == 'End within 1 day']
    one_day = one_day.groupby(['violence']).size().reset_index(name='total_count')
    row = {"violence": "Violent", "total_count": 0}
    one_day = one_day.append(row, ignore_index=True)    

    # pie chart_within one day
    def my_autopct(pct):
        return ('%.1f%%' % pct) if pct > 0 else ""    

    fig, ax = plt.subplots(figsize=(15, 10))
    labels = one_day['violence']
    percentages = one_day['total_count']
    color = plt.cm.coolwarm([0.2, 0.68])
    ax.pie(percentages, explode=[0.5, 0], labels=labels,
           autopct=my_autopct,
           shadow=False, startangle=0, colors=color,
           pctdistance=0, labeldistance=1.1, radius=10.5)
    ax.axis('equal')
    ax.set_title("Within 1 day")    
    ax.legend(frameon=False, bbox_to_anchor=(1.5, 0.8))    
    fig.show()

    # 1 day to 10 days
    one_to_ten = violence0.copy()
    one_to_ten = one_to_ten.groupby(['protest_time_interval', 'violence']).size().reset_index(name='total_count')
    one_to_ten = one_to_ten[one_to_ten.protest_time_interval == '1 days to 10 days']    

    # pie chart for 1 day to 10 days
    fig, ax = plt.subplots(figsize=(15, 10))
    labels = one_to_ten['violence']
    percentages = one_to_ten['total_count']
    color = plt.cm.coolwarm([0.2, 0.68])
    ax.pie(percentages, explode=[0.5, 0], labels=labels,
           autopct='%.1f%%',
           shadow=False, startangle=90, colors=color,
           pctdistance=0.8, labeldistance=0.5, radius=10.5)
    ax.axis('equal')
    ax.set_title("1 day to 10 days")    
    ax.legend(frameon=False, bbox_to_anchor=(1.5, 0.8))    
    fig.show()
