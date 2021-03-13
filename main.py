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

'''Cleaning the data'''
def preprocess(data):
    '''
    Clean the data
    '''
    assert isinstance(data, pd.DataFrame), 'expected loaded dataframe'
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

'''Plot the percentage of Violent Protests For Each Protest Reason'''
def viol_percentage_line_plot(data):
    '''
    Processes the input data and generates a line plot for the violence percentage of each reason.
    '''
    assert isinstance(data, pd.DataFrame), 'expected loaded dataframe'
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

'''Plot the Number of Protests in Kenya from 1990-2020'''
def kenya_line_plot(data):
    '''
    Processes the data and generates a line plot of the number of protests in Kenya from 1990 to 2020.
    '''
    assert isinstance(data, pd.DataFrame), 'expected loaded dataframe'
    df = data
    df_kenya = df[df['country'] == 'Kenya']
    df_kenya_yrs = df_kenya.groupby('year').sum()
    # Plot of number of protests over the years for Kenya
    plt.figure(figsize=(6,4), dpi=190)
    kenya_line_plt = sns.lineplot(x = 'year', y = 'protest', data = df_kenya_yrs)
    plt.title('Number of Protests in Kenya from 1990-2020', size = 16)
    plt.ylabel('protest', size = 14)
    plt.xlabel('year', size = 14)

'''Plot the Number of Protests in Kenya for each Reason in Kenya 2015'''
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

    mybar = plt.bar(df_reasons.index, df_reasons['Total Protests'])
    for bar in mybar:
        bar.set_color("grey")
    mybar[0].set_color('#e36c55')
    mybar[4].set_color('#e36c55')

    plt.xlabel('Reason', size = 14)
    plt.ylabel('Occurences', size = 14)
    plt.title('Percentage of Violent Protests For Each Reason', size = 16)
    plt.grid(False)

'''Plot Violence vs Duration'''
def duration_and_violence(main_data):    
    '''
    Three pie charts related to violence and duration
    '''
    assert isinstance(main_data, pd.DataFrame), 'expected loaded dataframe'
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
    violence = violence.sort_values(['protest_time']).drop(['country', 'year',
                                                            'protest', 'protesterviolence', 'violent_protest_time'], axis=1)    
    violence0 = violence.copy()
    bins1 = [-1, 0, 10, 950]
    index = ['End within 1 day', '1 days to 10 days', 'More than 10 days']
    violence0.insert(0, "protest_time_interval", pd.cut(violence0['protest_time'], bins=bins1, labels=index))
    violence0 = violence0.dropna(subset=['protest_time_interval'])
    violence0['violence'] = np.where(violence0['violence_both'] == 1, 'Violent', 'Nonviolent')
    violence0['count'] = 1

    # More than 10 days
    ten_days_more = violence0[violence0.protest_time_interval == 'More than 10 days']
    ten_days_more = ten_days_more.groupby(['violence']).size().reset_index(name='total_count')    

    # piechart for more than ten days
    labels = ten_days_more['violence']
    percentages = ten_days_more['total_count']
    sns.set_palette("coolwarm")
    plt.figure(figsize=(6,4), dpi=190)
    col = plt.get_cmap('coolwarm')(np.linspace(.2, .9, len(labels)))
    plt.pie(percentages, labels=labels, autopct='%2.1f%%', shadow=True, colors=col)
    plt.axis('equal')

    plt.title('More than 10 days')
    
    # Within one day
    one_day = violence0[violence0.protest_time_interval == 'End within 1 day']
    one_day = one_day.groupby(['violence']).size().reset_index(name='total_count')
    row = {"violence": "Violent", "total_count": 0}
    one_day = one_day.append(row, ignore_index=True)    

    # pie chart_within one day
    labels = one_day['violence']
    percentages = one_day['total_count']
    sns.set_palette("coolwarm")
    plt.figure(figsize=(6,4), dpi=190)
    col = plt.get_cmap('coolwarm')(np.linspace(.2, .9, len(labels)))
    plt.pie(percentages, labels=labels, autopct='%2.1f%%', shadow=True, colors=col)
    plt.axis('equal')

    plt.title('Within 1 day')

    # 1 day to 10 days
    one_to_ten = violence0.copy()
    one_to_ten = one_to_ten.groupby(['protest_time_interval', 'violence']).size().reset_index(name='total_count')
    one_to_ten = one_to_ten[one_to_ten.protest_time_interval == '1 days to 10 days']    

    # pie chart for 1 day to 10 days
    labels = one_to_ten['violence']
    percentages = one_to_ten['total_count']
    sns.set_palette("coolwarm")
    plt.figure(figsize=(6,4), dpi=190)
    col = plt.get_cmap('coolwarm')(np.linspace(.2, .9, len(labels)))
    plt.pie(percentages, labels=labels, autopct='%2.1f%%', shadow=True, colors=col)
    plt.axis('equal')

    plt.title('1 day to 10 days')

'''Plot Response to Violent and Non-Violent protests'''
def violance_response(data):    
    '''
    Two pie charts describing the response to violent and nonviolent events
    '''
    assert isinstance(data, pd.DataFrame), 'expected loaded dataframe'
    df_NoNviolance = data[data.protesterviolence == 0]
    df_violance = data[data.protesterviolence == 1]
    non_viloance_sum = df_NoNviolance.sum()[['responses_accomodation',
             'responses_arrests', 'responses_beatings', 'responses_crowd_dispersal', 'responses_ignore', 
             'responses_killings', 'responses_shootings']]
    viloance_sum = df_violance.sum()[['responses_accomodation',
             'responses_arrests', 'responses_beatings', 'responses_crowd_dispersal', 'responses_ignore', 
             'responses_killings', 'responses_shootings']]    
    list1 = ['responses_accomodation',
             'responses_arrests', 'responses_beatings', 'responses_crowd_dispersal', 'responses_ignore', 
             'responses_killings', 'responses_shootings']  
    list_non_violance = []
    for key in list1:
        list_non_violance.append(non_viloance_sum[key])
    # print(list_non_violance)    
    list_violance = []
    for key in list1:
        list_violance.append(viloance_sum[key])
    # print(list_violance)    
    labels = 'Accomodation', 'Arrests', 'Beatings', 'Crowd Dispersal', 'Ignore', 'Killings', 'Shooting'
    col = ['lightblue','brown','lavenderblush', 'teal', 'darksalmon', 'blueviolet']    
    plt.figure(figsize=(6,4), dpi=190)
    col = plt.get_cmap('coolwarm')(np.linspace(.2, .9, len(labels)))
    plt.pie(list_violance, labels=labels, autopct='%1.0f%%', shadow=True, colors=col)
    plt.axis('equal')
    plt.title('State Responses for Violent Protests')    
    plt.figure(figsize=(6,4), dpi=190)
    col = plt.get_cmap('coolwarm')(np.linspace(.2, .9, len(labels)))
    plt.pie(list_non_violance, labels=labels, autopct='%1.0f%%', shadow=True, colors=col)
    plt.axis('equal')
    plt.title('State Responses for Non-Violent Protests')

'''Plot violence percentage per day'''
def line_percent_violence_day(data):
    '''
    Line chart of percent violence versus duration 
    '''
    assert isinstance(data, pd.DataFrame), 'expected loaded dataframe'
    tmp = dict()
    tmp1 = []
    tmp2 = []
    for x in range(1,8):
        Idx = data['protest_time']==x
        v_Idx = data['violence_both']>0
        
        if(sum(Idx)==0):
            tmp[x]=0
        else:
            tmp[(x,x+10)]=sum(v_Idx & Idx)/sum(v_Idx)*100
            tmp1.append(sum(v_Idx & Idx)/sum(v_Idx)*100)
            tmp2.append(x)

    plt.figure(figsize=(6,4),dpi=190)

    plt.plot(tmp2, tmp1)
    plt.xlabel('Number of Days')
    plt.ylabel('Percentage of Violence')
    plt.title('')
    plt.grid(False)

'''Plot success per reason'''
def percent_per_reason_bar(data):
    '''
    Bar graph of percent success per reason 
    '''
    assert isinstance(data, pd.DataFrame), 'expected loaded dataframe'
    list_reasons = ['reasons_social', 'reasons_policebrutality','reasons_land', 'reasons_political', 'reasons_labor','reasons_price', 'reasons_removal']
    s_Idx = data['success']>0
    labels = ['social reforms', 'police brutality', 'property', 'law & politics', 'labor rights','inflation', 'anti-authoritarian']
    s_dist = []
    for reason in list_reasons:
        Idx = data[reason]>0
        s_dist.append(sum(Idx&s_Idx)/sum(Idx)*100)   
    fig, ax = plt.subplots(figsize=(6,4), dpi=190)

    y_pos = np.arange(len(labels))
    y_pos = np.linspace(0, 2.5, len(labels))

    mybars = ax.barh(y_pos, s_dist, align='center', height=0.3, color=(0.2, 0.4, 0.6, 0.6))
    for bar in mybars:
        bar.set_color('grey')
    mybars[2].set_color('#E36C55')
    mybars[3].set_color('#6a8bef')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Percent Successful')
    ax.set_ylabel('Reasons')
    ax.set_title("Percent of Successful Protests For Each Reason")
    plt.grid(False)

def percent_per_duration_bar(data):
    '''
    Bar graph of percent success per duration
    '''
    list_d = [(0,1),(1,7),(7,30),(30,365),(365,7000)]
    s_Idx = data['success']>0
    labels = ['0-1 day','1-7 days','7-30 days','30-365 days','>1 year']
    s_dist = []
    for a,b in list_d:
        Idx1 = data['protest_time']>a
        Idx2 = data['protest_time']<=b
        s_dist.append(sum(Idx1&Idx2&s_Idx)/sum(Idx1&Idx2)*100)
    fig, ax = plt.subplots(figsize=(6,4),dpi=190)

    y_pos = np.arange(len(labels))
    y_pos = np.linspace(0, 1.5, len(labels))

    col = plt.get_cmap('coolwarm')(np.linspace(0.3, 0.3, len(labels)+1))
    mybars = ax.barh(y_pos, s_dist, align='center', height=0.3, color=col)

        
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Percent Successful')
    ax.set_ylabel('Protest Duration')
    ax.set_title("Percent of Successful Protests For Various Durations")

    plt.grid(False)

    plt.show()


def plot_heatmap(data):
    '''
    Plot a heatmap of the correlation between reasons and responses
    '''
    assert isinstance(data, pd.DataFrame), 'expected loaded dataframe'
    # feature extraction
    useful_data = data[['country','reasons_social', 'reasons_policebrutality',
                                            'reasons_other', 'reasons_land', 'reasons_political', 'reasons_labor',
                                            'reasons_price', 'reasons_removal', 'responses_accomodation',
                                            'responses_arrests', 'responses_shootings', 'responses_other',
                                            'responses_crowd_dispersal', 'responses_beatings', 'responses_killings',
                                            'responses_ignore']].groupby(['country']).sum().reset_index()

    # find correlation
    correlation = useful_data.corr()

    trimmed_correlation = correlation[['reasons_social', 'reasons_policebrutality','reasons_land', 'reasons_political', 'reasons_labor',
                                                                            'reasons_price', 'reasons_removal']].loc[['responses_accomodation',
                                                                            'responses_arrests', 'responses_shootings',
                                                                            'responses_crowd_dispersal', 'responses_beatings', 'responses_killings',
                                                                            'responses_ignore']]
    # ploting the heatmap
    plt.figure(figsize = (11,7))
    cmp = sns.color_palette("coolwarm", as_cmap=True)
    ax = sns.heatmap(trimmed_correlation.T, annot = True, cmap = cmp, cbar=True, xticklabels=['accomodation', 'detentions', 'shootings', 'crowd dispersal', 'assault', 'deaths','indifference'], yticklabels=['social reforms', 'police brutality', 'property', 'law & politics', 'labor rights','inflation', 'anti-authoritarian'])

    plt.ylabel('Reasons',fontsize='x-large')
    plt.xlabel('Responses',fontsize='x-large')

    plt.xticks(fontweight='roman')
    plt.yticks(fontweight='roman')

def plot_success_stats(data):
    '''
    Plot of success rates violence/nonviolence circle
    '''
    assert isinstance(data, pd.DataFrame), 'expected loaded dataframe'
    success_Idx = data['success']>0
    violent_Idx = data['violence_both']>0

    violent_success = sum(violent_Idx & success_Idx)/sum(success_Idx)*100

    sizes = [violent_success, 100-violent_success]

    labels = ['violent', 'non-violent']
    color_palette = plt.get_cmap('coolwarm')(
            np.linspace(0.1, 0.9, len(labels)))

    plt.figure(figsize=(12,7))

    plt.pie(sizes, labels=labels, shadow=True, colors=color_palette, textprops={'fontsize': 'x-large'})
    plt.axis('equal')

    central_circle=plt.Circle( (0,0), 0.6, color='white')
    p=plt.gcf()
    p.gca().add_artist(central_circle)
    plt.text(0, 0, 'Success', ha='center', fontsize='xx-large')
    plt.show()

def plot_ignored_stats(data):
    '''
    Circle plot of ignore rate violent vs nonviolent
    '''
    assert isinstance(data, pd.DataFrame), 'expected loaded dataframe'
    ignored_Idx = data['responses_ignore']>0
    violent_Idx = data['violence_both']>0

    violent_ignored = sum(violent_Idx & ignored_Idx)/sum(ignored_Idx)*100

    sizes = [violent_ignored, 100-violent_ignored]

    labels = ['violent', 'non-violent']
    color_palette = plt.get_cmap('coolwarm')(
            np.linspace(0.1, 0.9, len(labels)))

    plt.figure(figsize=(12,7))

    plt.pie(sizes, labels=labels, shadow=True, colors=color_palette, textprops={'fontsize': 'x-large'})
    plt.axis('equal')

    central_circle=plt.Circle( (0,0), 0.6, color='white')
    p=plt.gcf()
    p.gca().add_artist(central_circle)
    plt.text(0, 0, 'Ignorance', ha='center', fontsize='xx-large')
    plt.show()

def plot_reasons(data):
    '''
    Plot of the reasons in the dataset
    '''
    assert isinstance(data, pd.DataFrame), 'expected loaded dataframe'

    summed_data = data.sum()
    reasons_list = ['reasons_social','reasons_policebrutality','reasons_land','reasons_political','reasons_labor','reasons_price','reasons_removal']
    score_reasons = [summed_data[reason] for reason in reasons_list]
    score_reasons = [784, 1101, 602, 3750, 1511, 1214, 1689]

    category_names = ['social', 'police brutality', 'property', 'law & politics', 'labor rights', 'inflation', 'anti-authoritarian']
    results = {"":score_reasons}

    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('mako_r')(np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(35, 3))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, category_names[i], ha='center', va='center',
                color=text_color, fontsize=20, fontweight='bold', fontname='Gill Sans MT')

    ax.axis('off')
    plt.show()



def plot_responses(data):
    '''
    Plot of responses in the dataset
    '''
    assert isinstance(data, pd.DataFrame), 'expected loaded dataframe'
    summed_data = data.sum()
    responses_list = ['responses_arrests', 'responses_shootings', 'responses_crowd_dispersal', 'responses_beatings', 'responses_killings','responses_ignore', 'success']
    score_responses = [summed_data[response] for response in responses_list]
    score_responses = [2149, 1200, 3072, 900, 1000, 5285, 1027]

    category_names = ['detentions', 'shootings', 'crowd dispersal', 'assault', 'deaths', 'indifference', 'success']
    results = {"":score_responses}

    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('flare')(np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(35, 3))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, category_names[i], ha='center', va='center',
                color=text_color, fontsize=20, fontweight='bold', fontname='Gill Sans MT')

    ax.axis('off')
    plt.show()

def plot_worldmap(data_here):
    '''
    Plot of the 25 highest and 25 lowest countries in
    terms of protest occurrences
    '''
    assert isinstance(data_here, pd.DataFrame), 'expected loaded dataframe'
    data = data_here.copy()
    # copy necessitated because non-existent countries are changed
    data['country'] = data['country'].replace(['United Arab Emirate'], 'United Arab Emirates')
    data['country'] = data['country'].replace(['USSR'], 'Russia')
    data['country'] = data['country'].replace(['Germany West','Germany East'], 'Germany')
    data['country'] = data['country'].replace(['Yugoslavia'], 'Croatia')
    data['country'] = data['country'].replace(['Czechoslovakia'], 'Slovakia')
    data['country'] = data['country'].replace(['Bosnia'], 'Bosnia and Herzegovina')
    data['country'] = data['country'].replace(['Serbia and Montenegro', 'Kosovo'], 'Serbia')
    data['country'] = data['country'].replace(['Congo Kinshasa', 'Congo Brazzaville'], 'Congo')
    data['country'] = data['country'].replace(['South Sudan', 'North Sudan'], 'Sudan')

    processed_data = data[['country','protest']].groupby('country').sum().sort_values('protest')

    reverse_country_map  = {v:k for k,v in COUNTRIES.items()}

    most_protests = []
    for x in processed_data[-25:].iterrows():
        try:
            most_protests.append(reverse_country_map[x[0]])
        except:
            pass

    least_protests = []
    for x in processed_data[:25].iterrows():
        try:
            least_protests.append(reverse_country_map[x[0]])
        except:
            pass

    worldmap_chart = pygal.maps.world.World()
    worldmap_chart.title = 'Outliers of the World'
    worldmap_chart.add('Most Protests', most_protests)
    worldmap_chart.add('Least Protests', least_protests)
    worldmap_chart.render_in_browser()
