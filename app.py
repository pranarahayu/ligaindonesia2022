import pandas as pd
import numpy as np
import streamlit as st
from mplsoccer import Pitch, VerticalPitch, PyPizza, FontManager
import matplotlib.pyplot as plt

st.set_page_config(page_title='Liga Indonesia')
st.header('Liga Indonesia 2022/23 Data and Statistics')
st.markdown('Created by: Prana (@prxrhx on Twitter) | Data: Lapangbola.com')

xgdata = pd.read_excel('/app/ligaindonesia2022/data/xGData.xlsx').sort_values(by=['Team', 'Player']).reset_index(drop=True)
oppdata = pd.read_excel('/app/ligaindonesia2022/data/xGData.xlsx').sort_values(by=['Opponent', 'Player']).reset_index(drop=True)
pct1 = pd.read_excel('/app/ligaindonesia2022/data/pct_rank_liga1.xlsx').sort_values(by=['Team_pct', 'Name']).reset_index(drop=True)
pct1_x = pct1[pct1['Team_pct']!='League Average']
pct2 = pd.read_excel('/app/ligaindonesia2022/data/pct_rank_liga2.xlsx').sort_values(by=['Team_pct', 'Name']).reset_index(drop=True)
pct2_x = pct2[pct2['Team_pct']!='League Average']

temp = xgdata[['Team', 'xG', 'GW']]
forxg = temp.groupby(['Team', 'GW']).sum()
temp = xgdata[['Opponent', 'xG', 'GW']].rename(columns={'Opponent':'Team', 'xG':'xGA'})
forxga = temp.groupby(['Team', 'GW']).sum()
data3 = pd.merge(forxg, forxga, on=['Team', 'GW'], how='left').reset_index()

pitch = VerticalPitch(half=True, pitch_type='wyscout',
                      pitch_color='#0E1117', line_color='#FAFAFA',
                      line_alpha=1, goal_type='box', goal_alpha=1,
                      pad_bottom=0.5, pad_right=0.5, pad_left=0.5)

def shotmaplot(data):
    goals = data[data['Event']=='Goal']
    ngoals = data[data['Event']!='Goal']
    fig, ax = plt.subplots(dpi=500)
    fig.patch.set_facecolor('#0E1117')
    pitch.draw(ax=ax)
    ax.scatter(ngoals['Y'], ngoals['X'], s=ngoals.xG*600,
               c='#22AF15', marker='o', edgecolors='#FAFAFA', alpha=0.7)
    ax.scatter(goals['Y'], goals['X'], s=goals.xG*600,
               c='#22AF15', marker='*', edgecolors='#FAFAFA', alpha=0.8)
    return fig

def get_metric(data):
    goal = data[data['Event']=='Goal']['Event'].count()
    xg = round((data['xG'].sum()),2)
    shot = data['Event'].count()
    xgshot = round((round((data['xG'].sum()),2))/(data['Event'].count()),2)
    match = pd.unique(data['GW'])
    xgmatch = round(((round((data['xG'].sum()),2))/len(match)), 2)
    return [goal, xg, shot, xgshot, xgmatch]

gw = max(data3['GW'])     
def xgaplot(team):
    fig, ax = plt.subplots(dpi=500)
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    auxdata = data3[data3['Team'] == team]
    
    ax.plot([1,gw], [1,1], color='#FAFAFA', ls='--', alpha=0.2, zorder=1)
    ax.plot([1,gw], [2,2], color='#FAFAFA', ls='--', alpha=0.2, zorder=1)
    ax.plot([1,gw], [3,3], color='#FAFAFA', ls='--', alpha=0.2, zorder=1)
    ax.plot([1,gw], [1.5,1.5], color='#FAFAFA', ls='--', alpha=0.2, zorder=1)
    ax.plot([1,gw], [.5,.5], color='#FAFAFA', ls='--', alpha=0.2, zorder=1)
    ax.plot([1,gw], [2.5,2.5], color='#FAFAFA', ls='--', alpha=0.2, zorder=1)

    ax.plot(auxdata['GW'], auxdata['xG'], color='#22af15', zorder=6)
    ax.scatter(auxdata['GW'], auxdata['xG'], c='#22af15', marker='o', zorder=7)
    ax.scatter(2.5, 3.4, c='#22af15', marker='o', zorder=7)
    ax.text(2.8, 3.4, 'xG', c='#fafafa', zorder=7, va='center', size=12)

    ax.plot(auxdata['GW'], auxdata['xGA'], color='#a215af', zorder=6)
    ax.scatter(auxdata['GW'], auxdata['xGA'], c='#a215af', marker='o', zorder=7)
    ax.scatter(2.5, 3.2, c='#a215af', marker='o', zorder=7)
    ax.text(2.8, 3.2, 'xGA', c='#fafafa', zorder=7, va='center', size=12)

    ax.fill_between(x=auxdata['GW'], y1=auxdata['xG'], y2=auxdata['xGA'], where=(auxdata['xG'] > auxdata['xGA']),
                    alpha=0.5, interpolate=True, color='#22af15', zorder=6)
  
    ax.fill_between(x=auxdata['GW'], y1=auxdata['xG'], y2=auxdata['xGA'], where=(auxdata['xG'] < auxdata['xGA']),
                    alpha=0.5, interpolate=True, color='#a215af', zorder=6)
    
    ax.set_ylim(0, 3.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#FAFAFA')
    ax.spines['left'].set_color('#FAFAFA')
    for t in ax.xaxis.get_ticklines(): t.set_color('#FAFAFA')
    ax.tick_params(axis='x', colors='#FAFAFA')
  
    for t in ax.yaxis.get_ticklines(): t.set_color('#FAFAFA')
    ax.tick_params(axis='y', colors='#FAFAFA')

    ax.set_xticks([x + 1 for x in range(gw)])
    ax.set_yticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
    ax.tick_params(axis='both', labelsize=10)
    
    return fig

def beli_pizza(komp, pos, klub, name):
    if (komp == 'Liga 1'):
        datapizza = pct1
    else:
        datapizza = pct2
    
    temp = pd.DataFrame()
    if (pos=='Forward'):
        temp['Name'] = datapizza['Name']
        temp['Team'] = datapizza['Team_pct']

        temp['Shots'] = round(datapizza['Shots_pct'])
        temp['Goals'] = round(datapizza['Goals_pct'])
        temp['Chances Created'] = round(datapizza['Create Chance_pct'])
        temp['Shots on Target %'] = round(datapizza['Shot on Target Ratio_pct'])
        temp['Conversion Ratio'] = round(datapizza['Conversion Ratio_pct'])

        temp['Assist'] = round(datapizza['Assist_pct'])

        temp['Pass Accuracy'] = round(datapizza['Pass Accuracy_pct'])
        temp['Successful Dribbles'] = round(datapizza['Dribble_pct'])

        temp['Successful Tackles'] = round(datapizza['Tackle_pct'])
        temp['Intercepts'] = round(datapizza['Intercepts_pct'])
        temp['Recoveries'] = round(datapizza['Recovery_pct'])
        temp['Aerial Won %'] = round(datapizza['Aerial Won %_pct'])

        temp = temp[(temp['Name']==name) | (temp['Name']=='Average FW')].reset_index()
        
        slice_colors = ["#22af15"] * 5 + ["#adaf15"] * 1 + ["#a215af"] * 2 + ["#2115af"] * 4
        text_colors = ["#FAFAFA"] * 5 + ["#0E1117"] * 1 + ["#FAFAFA"] * 6
    elif (pos=='Attacking 10') or (pos=='Winger'):
        temp['Name'] = datapizza['Name']
        temp['Team'] = datapizza['Team_pct']

        temp['Shots'] = round(datapizza['Shots_pct'])
        temp['Goals'] = round(datapizza['Goals_pct'])
        temp['Chances Created'] = round(datapizza['Create Chance_pct'])
        temp['Shots on Target %'] = round(datapizza['Shot on Target Ratio_pct'])
        temp['Conversion Ratio'] = round(datapizza['Conversion Ratio_pct'])

        temp['Assist'] = round(datapizza['Assist_pct'])

        temp['Pass Accuracy'] = round(datapizza['Pass Accuracy_pct'])
        temp['Successful Dribbles'] = round(datapizza['Dribble_pct'])
        temp['Successful Crosses'] = round(datapizza['Cross_pct'])

        temp['Successful Tackles'] = round(datapizza['Tackle_pct'])
        temp['Intercepts'] = round(datapizza['Intercepts_pct'])
        temp['Recoveries'] = round(datapizza['Recovery_pct'])
        
        if (pos=='Winger'):
            temp = temp[(temp['Name']==name) | (temp['Name']=='Average W')].reset_index()
        else:
            temp = temp[(temp['Name']==name) | (temp['Name']=='Average CAM')].reset_index()
        
        slice_colors = ["#22af15"] * 5 + ["#adaf15"] * 1 + ["#a215af"] * 3 + ["#2115af"] * 3
        text_colors = ["#FAFAFA"] * 5 + ["#0E1117"] * 1 + ["#FAFAFA"] * 6
    elif (pos=='Midfielder'):
        temp['Name'] = datapizza['Name']
        temp['Team'] = datapizza['Team_pct']

        temp['Shots'] = round(datapizza['Shots_pct'])
        temp['Goals'] = round(datapizza['Goals_pct'])
        temp['Chances Created'] = round(datapizza['Create Chance_pct'])
        temp['Shots on Target %'] = round(datapizza['Shot on Target Ratio_pct'])

        temp['Assist'] = round(datapizza['Assist_pct'])

        temp['Pass Accuracy'] = round(datapizza['Pass Accuracy_pct'])
        temp['Successful Dribbles'] = round(datapizza['Dribble_pct'])

        temp['Successful Tackles'] = round(datapizza['Tackle_pct'])
        temp['Intercepts'] = round(datapizza['Intercepts_pct'])
        temp['Recoveries'] = round(datapizza['Recovery_pct'])
        temp['Blocks'] = round(datapizza['Block_pct'])

        temp = temp[(temp['Name']==name) | (temp['Name']=='Average CM')].reset_index()
        
        slice_colors = ["#22af15"] * 4 + ["#adaf15"] * 1 + ["#a215af"] * 2 + ["#2115af"] * 4
        text_colors = ["#FAFAFA"] * 4 + ["#0E1117"] * 1 + ["#FAFAFA"] * 6
    elif (pos=='Fullback'):
        temp['Name'] = datapizza['Name']
        temp['Team'] = datapizza['Team_pct']

        temp['Shots'] = round(datapizza['Shots_pct'])
        temp['Goals'] = round(datapizza['Goals_pct'])
        temp['Chances Created'] = round(datapizza['Create Chance_pct'])

        temp['Assist'] = round(datapizza['Assist_pct'])

        temp['Pass Accuracy'] = round(datapizza['Pass Accuracy_pct'])
        temp['Successful Dribbles'] = round(datapizza['Dribble_pct'])
        temp['Successful Crosses'] = round(datapizza['Cross_pct'])

        temp['Successful Tackles'] = round(datapizza['Tackle_pct'])
        temp['Intercepts'] = round(datapizza['Intercepts_pct'])
        temp['Recoveries'] = round(datapizza['Recovery_pct'])
        temp['Blocks'] = round(datapizza['Block_pct'])
        temp['Aerial Won %'] = round(datapizza['Aerial Won %_pct'])

        temp = temp[(temp['Name']==name) | (temp['Name']=='Average FB')].reset_index()
        
        slice_colors = ["#22af15"] * 3 + ["#adaf15"] * 1 + ["#a215af"] * 3 + ["#2115af"] * 5
        text_colors = ["#FAFAFA"] * 3 + ["#0E1117"] * 1 + ["#FAFAFA"] * 8
    elif (pos=='Center Back'):
        temp['Name'] = datapizza['Name']
        temp['Team'] = datapizza['Team_pct']

        temp['Shots'] = round(datapizza['Shots_pct'])
        temp['Goals'] = round(datapizza['Goals_pct'])

        temp['Assist'] = round(datapizza['Assist_pct'])

        temp['Pass Accuracy'] = round(datapizza['Pass Accuracy_pct'])

        temp['Successful Tackles'] = round(datapizza['Tackle_pct'])
        temp['Intercepts'] = round(datapizza['Intercepts_pct'])
        temp['Recoveries'] = round(datapizza['Recovery_pct'])
        temp['Blocks'] = round(datapizza['Block_pct'])
        temp['Aerial Won %'] = round(datapizza['Aerial Won %_pct'])

        temp = temp[(temp['Name']==name) | (temp['Name']=='Average CB')].reset_index()
        
        slice_colors = ["#22af15"] * 2 + ["#adaf15"] * 1 + ["#a215af"] * 1 + ["#2115af"] * 5
        text_colors = ["#FAFAFA"] * 2 + ["#0E1117"] * 1 + ["#FAFAFA"] * 6
    else:
        temp['Name'] = datapizza['Name']
        temp['Team'] = datapizza['Team_pct']

        temp['Long Goal Kick %'] = round(datapizza['Long Goal Kick %_pct'])
        temp['Pass Accuracy'] = round(datapizza['Pass Accuracy_pct'])

        temp['Cross Claimed'] = round(datapizza['Cross Claim_pct'])
        temp['Sweeping Actions'] = round(datapizza['Keeper - Sweeper_pct'])
        temp['Save Percentage'] = round(datapizza['Save Percentage_pct'])
        temp['Saves'] = round(datapizza['Saves_pct'])
        temp['Penalties Saved'] = round(datapizza['Penalty Save_pct'])

        temp = temp[(temp['Name']==name) | (temp['Name']=='Average GK')].reset_index()
        
        slice_colors = ["#22af15"] * 2 + ["#a215af"] * 5
        text_colors = ["#FAFAFA"] * 7
        
    temp = temp.drop(['Team'],axis=1)

    avg_player = temp[temp['Name'].str.contains('Average')]
    av_name = list(avg_player['Name'])[0]

    params = list(temp.columns)
    params = params[2:]

    a_values = []
    b_values = []
    
    for x in range(len(temp['Name'])):
        if temp['Name'][x] == name:
            a_values = temp.iloc[x].values.tolist()
        if temp['Name'][x] == av_name:
            b_values = temp.iloc[x].values.tolist()
        
    a_values = a_values[2:]
    b_values = b_values[2:]

    values = [a_values,b_values]
    
    baker = PyPizza(params=params, background_color="#0E1117", straight_line_color="#0E1117",
                    straight_line_lw=2, last_circle_lw=0, other_circle_lw=0, inner_circle_size=5)

    fig, ax = baker.make_pizza(a_values, compare_values=b_values, figsize=(10, 10),
                               color_blank_space="same", slice_colors=slice_colors,
                               value_colors=text_colors, value_bck_colors=slice_colors,
                               blank_alpha=0.35,

                               kwargs_slices=dict(edgecolor="none", zorder=0, linewidth=2),
                               kwargs_compare=dict(facecolor="none", edgecolor="#0E1117",
                                                   zorder=8, linewidth=2, ls='--'),
                               kwargs_params=dict(color="#FAFAFA", fontsize=10, va="center"),
                               kwargs_values=dict(color="#0E1117", fontsize=11, zorder=3,
                                                  bbox=dict(edgecolor="#FAFAFA", boxstyle="round,pad=0.2", lw=1)),
                               kwargs_compare_values=dict(color="#252627", fontsize=11, zorder=3, alpha=0,
                                                          bbox=dict(edgecolor="#252627", facecolor="#E1E2EF",
                                                                    boxstyle="round,pad=0.2", lw=1, alpha=0)))

    fig.text(0.515, 0.975, name + ' - ' + klub, size=16,
             ha="center", color="#FAFAFA", weight='bold')

    fig.text(0.515, 0.953, "Percentile Rank vs League Average "+pos,
             size=11, ha="center", color="#FAFAFA")

    CREDIT_1 = "Data: Lapangbola.com | Created by: @prxrhx"
    if (komp=='Liga 2'):
        CREDIT_2 = "Liga 2 | Season 2022/23 | Min. 180 mins played"
    else:
        CREDIT_2 = "Liga 1 | Season 2022/23 | Min. 500 mins played"

    fig.text(0.515, 0.02, f"{CREDIT_1}\n{CREDIT_2}", size=9,
             color="#FAFAFA", ha="center")
             
    if (pos != 'Goalkeeper'):
        fig.text(0.268, 0.935, "Goal Threat              Creativity             In Possession              Out of Possession",
                 size=10, color="#FAFAFA", va='center')

        fig.patches.extend([
        plt.Rectangle((0.247, 0.9275), 0.015, 0.015, fill=True, color="#22af15",
                      transform=fig.transFigure, figure=fig),
        plt.Rectangle((0.390, 0.9275), 0.015, 0.015, fill=True, color="#adaf15",
                      transform=fig.transFigure, figure=fig),
        plt.Rectangle((0.515, 0.9275), 0.015, 0.015, fill=True, color="#a215af",
                      transform=fig.transFigure, figure=fig),
        plt.Rectangle((0.668, 0.9275), 0.015, 0.015, fill=True, color="#2115af",
                      transform=fig.transFigure, figure=fig)
        ])
    else:
        fig.text(0.398, 0.935, "Distribution                     Goalkeeping",
                 size=10, color="#FAFAFA", va='center')

        fig.patches.extend([
        plt.Rectangle((0.375, 0.9275), 0.015, 0.015, fill=True, color="#22af15",
                      transform=fig.transFigure, figure=fig),
        plt.Rectangle((0.550, 0.9275), 0.015, 0.015, fill=True, color="#a215af",
                      transform=fig.transFigure, figure=fig)
        ])
    
    plt.savefig('/app/ligaindonesia2022/data/pizza.jpg', dpi=500, bbox_inches='tight')
    
    return fig

gw = max(xgdata['GW'])    
tab1, tab2 = st.tabs(['xG and xGA', 'Player Radar'])

with tab1:
    tab1.subheader('Expected Goal & Expected Goal Allowed')
    fil1, fil2 = st.columns(2)
    with fil1:
        team_filter = st.selectbox('Select Team', pd.unique(xgdata['Team']))
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    df_players = xgdata[xgdata['Team']==team_filter].reset_index(drop=True)
    col1, col2 = st.columns(2)
    
    xg = xgdata[xgdata['Team']==team_filter].reset_index(drop=True)
    xga = xgdata[xgdata['Opponent']==team_filter].reset_index(drop=True)
    
    mtr = get_metric(xg)
    m1.metric(label='Goals', value=mtr[0])
    m2.metric(label='xG', value=mtr[1])
    m3.metric(label='xG/Match', value=mtr[4])
    mtr = get_metric(xga)
    m4.metric(label='Conceded', value=mtr[0])
    m5.metric(label='xGA', value=mtr[1])
    m6.metric(label='xGA/Match', value=mtr[4])
    
    st.markdown('''<style>
                [data-testid="stMetricLabel"] > div:nth-child(1) {justify-content: center;}
                [data-testid="stMetricValue"] > div:nth-child(1) {justify-content: center;}
                </style>''', unsafe_allow_html=True)
    
    with col1:
        st.subheader(team_filter+'\'s Shots Attempted')
        data = xgdata[xgdata['Team']==team_filter].reset_index(drop=True)
        map = shotmaplot(data)
        st.pyplot(map)
        
        st.subheader(team_filter+'\'s xG vs xGA over GWs')
        st.markdown(team_filter +'\'s Attacking & Defending Performances in Liga 1')
        gb = xgaplot(team_filter)
        st.pyplot(gb)
        
    with col2:
        st.subheader(team_filter+'\'s Shots Allowed')
        data = xgdata[xgdata['Opponent']==team_filter].reset_index(drop=True)
        map = shotmaplot(data)
        st.pyplot(map)
        
        st.subheader('Player\'s Shots Map')
        player_filter = st.selectbox('Select Player', pd.unique(df_players['Player']))
        data = df_players[df_players['Player']==player_filter].reset_index(drop=True)
        map = shotmaplot(data)
        st.pyplot(map)
    
    
with tab2:
    tab2.subheader('Player\'s Performance Radar')
    with st.expander('Read Me!'):
        st.write('This tab shows player performance radar based on the filters below. Minimum minutes played for Liga 1 is 720 mins and 180 mins for Liga 2. Players shown in the filter are the one that met that condition.')
    f1, f2 = st.columns(2)
    with f1:
        komp_filter = st.selectbox('Select League', ['Liga 1', 'Liga 2'])
        if (komp_filter=='Liga 1'):
            f_team = st.selectbox('Select Team', pd.unique(xgdata['Team']))
            dfp_y = pct1[pct1['Team_pct']==f_team].reset_index(drop=True)
            dfp_x = pct1[pct1['Team_pct']!='League Average'].reset_index(drop=True)
        else:
            f_team = st.selectbox('Select Team', pd.unique(pct2_x['Team_pct']))
            dfp_y = pct2[pct2['Team_pct']==f_team].reset_index(drop=True)
            dfp_x = pct2[pct2['Team_pct']!='League Average'].reset_index(drop=True)
        all_teams = st.checkbox('Select All Teams')
    with f2:
        pos_filter = st.selectbox('Select Position', pd.unique(pct1['Position_pct']))
        if all_teams:
          dfp = dfp_x[dfp_x['Position_pct']==pos_filter].reset_index(drop=True)
          f_player = st.selectbox('Select Player', pd.unique(dfp['Name']))
          if len(dfp[dfp['Name']==f_player])==1:
            f_team = list(dfp[dfp['Name']==f_player]['Team_pct'])[0]
          else:
            f_team = list(dfp[dfp['Name']==f_player]['Team_pct'])[1]
        else:
          dfp = dfp_y[dfp_y['Position_pct']==pos_filter].reset_index(drop=True)
          f_player = st.selectbox('Select Player', pd.unique(dfp['Name']))
    
    piz = beli_pizza(komp_filter, pos_filter, f_team, f_player)
    with f2:
        with open('/app/ligaindonesia2022/data/pizza.jpg', 'rb') as img:
            fn = 'Pizza_'+f_player+'-'+f_team+'.jpg'
            btn = st.download_button(label="Download Radar", data=img,
                                     file_name=fn, mime="image/jpg")
    st.pyplot(piz)
