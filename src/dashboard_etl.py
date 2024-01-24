import sqlite3
import zipfile
import pandas as pd
import numpy as np
from datetime import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os


########################################################################################################################
# FUNCTIONS
def temp_regression(df, y_var):
    # split b/t area
    vt = df[(df.Location == 'VT') & (df[f'{y_var}'].notnull())]
    nema = df[(df.Location == 'NEMA') & (df[f'{y_var}'].notnull())]

    # HDD:
    vt_heating = vt[vt.t_f < 40]
    nema_heating = nema[nema.t_f < 40]

    # CDD:
    vt_cooling = vt[vt.t_f > 75]
    nema_cooling = nema[nema.t_f > 75]

    # Regr
    model = LinearRegression()

    # Fit the model to the data
    model.fit(vt_heating[['t_f']], vt_heating[f'{y_var}'])
    vt_h_coef = {'name': 'VT_h', 'Intercept': model.intercept_, 'Coefficient': model.coef_[0]}

    # Fit the model to the data
    model.fit(nema_heating[['t_f']], nema_heating[f'{y_var}'])
    nema_h_coef = {'name': 'NEMA_h', 'Intercept': model.intercept_, 'Coefficient': model.coef_[0]}

    # Fit the model to the data
    model.fit(vt_cooling[['t_f']], vt_cooling[f'{y_var}'])
    vt_c_coef = {'name': 'VT_c', 'Intercept': model.intercept_, 'Coefficient': model.coef_[0]}

    # Fit the model to the data
    model.fit(nema_cooling[['t_f']], nema_cooling[f'{y_var}'])
    nema_c_coef = {'name': 'NEMA_c', 'Intercept': model.intercept_, 'Coefficient': model.coef_[0]}

    return pd.DataFrame((vt_h_coef, nema_h_coef, vt_c_coef, nema_c_coef))

def clean_file(df, filename):
    if 'weather' in filename:
        df.columns = ['datetime', '4008', '4003']
        df = pd.melt(raw, id_vars=['datetime'], var_name='Location', value_name='t_f')

    else:
        df = df.rename(columns={'BeginDate': 'datetime'})
        df['Location'] = df.Location.apply(lambda x: x[12:16])

    df = df.drop_duplicates()

    df['datetime'] = pd.to_datetime(df.datetime, utc=True)
    df['datetime_hourly'] = df.datetime.dt.floor("h")

    df['datetime'] = df.datetime.dt.tz_convert('America/New_York')  # re-assign EST
    df['datetime_hourly'] = df.datetime_hourly.dt.tz_convert('America/New_York')

    df['datetime'] = df.datetime.astype('str')  # convert to str for sql
    df['datetime_hourly'] = df.datetime_hourly.astype('str')

    df['datetime_hourly'] = df.datetime_hourly.str.slice(stop=-6)
    return df

########################################################################################################################
# DATA PROCESSING

zip_file_path = './data/dataset.zip'

# Connect to the SQLite database
conn = sqlite3.connect('Dashboard_DB.db')

# Create a cursor object to interact with the database
cursor = conn.cursor()

# Open the ZIP file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    for f in zip_ref.infolist():
        if (~f.filename.startswith('__MAC') & f.filename.endswith('csv')): # only files we want
            csv_file_name = f.filename[30:-4].replace('-', '_')
            print(csv_file_name,'data/dataset/' + f.filename)

            raw = pd.read_csv('data/dataset/' + f.filename)  # read file
            df = clean_file(raw, csv_file_name) # defined function

            # Write the DataFrame to the SQLite database
            df.to_sql(csv_file_name.split('.')[0], conn, if_exists='replace', index=False)

# Commit the changes to the database
conn.commit()

# Close the cursor and connection
cursor.close()
conn.close()

# CREATE COMPLETE DATATIME RANGE (x2) WITH LOCATIONS
start_date = dt(2021, 1, 1)
end_date = dt(2021, 12, 31, 23)  # Include the last hour of the year
hourly_index = pd.date_range(start=start_date, end=end_date, freq='H')
hourly_data_vt = pd.DataFrame({'datetime_hourly': hourly_index, 'Location': ['4003' for _ in range(len(hourly_index))]})
hourly_data_ma = pd.DataFrame({'datetime_hourly': hourly_index, 'Location': ['4008' for _ in range(len(hourly_index))]})
hourly_data_df = pd.concat([hourly_data_vt,hourly_data_ma])


# Step 2: Create or connect to a SQLite3 database
conn = sqlite3.connect('Dashboard_DB.db')

# Step 3: Create a table in the database
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS HourlyData (
        id INTEGER PRIMARY KEY,
        datetime_hourly DATETIME,
        Location FLOAT
    )
''')

# Step 4: Insert the generated hourly data into the table
hourly_data_df.to_sql('HourlyData', conn, index=False, if_exists='replace')

# Commit the changes and close the connection
conn.commit()
conn.close()

# CREATE ONE HOURLY TABLE
conn = sqlite3.connect('Dashboard_DB.db')

# Create a cursor object to interact with the database
cursor = conn.cursor()

# Create a composite index on column1 and column2
cursor.execute('CREATE INDEX IF NOT EXISTS idx_column1_column2 ON iso_ne_hourly_demand_2021(datetime_hourly, Location)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_column1_column2 ON weather_data(datetime_hourly, Location)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_column1_column2 ON iso_ne_fivemin_lmp_2021(datetime_hourly, Location)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_column1_column2 ON HourlyData(datetime_hourly, Location)')

# CREATE ONE HOURLY TABLE
conn = sqlite3.connect('Dashboard_DB.db')

# Create a cursor object to interact with the database
cursor = conn.cursor()

########################################################################################################################
# NOTE: THIS WAS TOO SLOW, USED SQLITE3 AND COULDN'T ASSIGN KEYS TO THE TABLES FOR QUICKER JOINING...

# df_raw = pd.read_sql_query('''SELECT
#                     HourlyData.datetime_hourly,
#                     HourlyData.Location as Location,
#
#                     AVG(t_f) as 't_f',
#                     AVG(LmpTotal) as 'lmp',
#                     AVG(Load) as 'load'
#
#                     FROM HourlyData
#                     LEFT JOIN iso_ne_hourly_demand_2021
#                     ON HourlyData.datetime_hourly = iso_ne_hourly_demand_2021.datetime_hourly
#                     AND HourlyData.Location = iso_ne_hourly_demand_2021.Location
#
#                     LEFT JOIN iso_ne_fivemin_lmp_2021
#                     ON HourlyData.datetime_hourly = iso_ne_fivemin_lmp_2021.datetime_hourly
#                     AND HourlyData.Location = iso_ne_fivemin_lmp_2021.Location
#
#                     LEFT JOIN weather_data
#                     ON HourlyData.datetime_hourly = weather_data.datetime_hourly
#                     AND HourlyData.Location = weather_data.Location
#
#                     group by HourlyData.datetime_hourly, HourlyData.Location
#                     ''', conn)
#
# cursor.close()
# conn.close()

# CREATE ONE HOURLY TABLE
conn = sqlite3.connect('Dashboard_DB.db')

# Create a cursor object to interact with the database
cursor = conn.cursor()

x = pd.read_sql_query('''SELECT
                    datetime_hourly,Location
                    FROM HourlyData
                    ''', conn)

e = pd.read_sql_query('''SELECT
                    Location,
                    datetime_hourly,
                    AVG(Load) as 'energy'
                    FROM iso_ne_hourly_demand_2021
                    GROUP BY Location, datetime_hourly
                    ''', conn)

l = pd.read_sql_query('''SELECT
                    Location,
                    datetime_hourly,
                    AVG(LmpTotal) as 'lmp',
                    AVG(LossComponent) as 'loss',
                    AVG(CongestionComponent) as 'congestion',
                    AVG(EnergyComponent) as 'energyCost'
                    FROM iso_ne_fivemin_lmp_2021
                    GROUP BY LOCATION, datetime_hourly
                    ''', conn)

w = pd.read_sql_query('''SELECT
                    Location,
                    datetime_hourly,
                    t_f
                    FROM weather_data
                    GROUP BY LOCATION, datetime_hourly
                    ''', conn)

cursor.close()
conn.close()

df_final = x.merge(e,how='left').merge(l,how='left').merge(w,how='left')

df_final.datetime_hourly = pd.to_datetime(df_final.datetime_hourly,utc=True)
df_final['day'] = df_final.datetime_hourly.dt.date
df_final['month'] = df_final.datetime_hourly.dt.month
df_final['hour'] = df_final.datetime_hourly.dt.hour + 1
df_final['season'] = df_final.month.apply(lambda x: 'summer' if x in (6,7,8,9) else 'winter')
df_final['Location'] = df_final.Location.apply(lambda x: 'VT' if x == '4003' else 'NEMA')

vt_max = max(df_final[(df_final.energy.notnull()) & (df_final.Location == 'VT')].energy)
nema_max = max(df_final[(df_final.energy.notnull()) & (df_final.Location == 'NEMA')].energy)

vt_total_energy = sum(df_final[(df_final.energy.notnull()) & (df_final.Location == 'VT')].energy)
nema_total_energy = sum(df_final[(df_final.energy.notnull()) & (df_final.Location == 'NEMA')].energy)


df_final['energy_normalized'] = df_final.apply(lambda row: row['energy']/nema_max if row['Location'] == 'NEMA' else row['energy']/vt_max, axis=1)
df_final['total_lmp'] = (df_final.energy * df_final.lmp)
df_final['total_energyCost'] = (df_final.energy * df_final.energyCost)
df_final['total_loss'] = (df_final.energy * df_final.loss)
df_final['total_congestion'] = (df_final.energy * df_final.congestion)

df_final['weighted_lmp'] = df_final.apply(lambda row: row['total_lmp']/nema_total_energy if row['Location'] == 'NEMA' else row['total_lmp']/vt_total_energy, axis=1)
df_final['weighted_energyCost'] = df_final.apply(lambda row: row['total_energyCost']/nema_total_energy if row['Location'] == 'NEMA' else row['total_energyCost']/vt_total_energy, axis=1)
df_final['weighted_loss'] = df_final.apply(lambda row: row['total_loss']/nema_total_energy if row['Location'] == 'NEMA' else row['total_loss']/vt_total_energy, axis=1)
df_final['weighted_congestion'] = df_final.apply(lambda row: row['total_congestion']/nema_total_energy if row['Location'] == 'NEMA' else row['total_congestion']/vt_total_energy, axis=1)


df_final.to_csv('./data/final_dashboard_df.csv')

########################################################################################################################
# EXPLORATORY DATA ANALYSIS

# 1. HEAT MAP
vt_load = df_final[df_final.Location == 'VT'][['day','hour','weighted_lmp']]
vt_heatmap = vt_load.pivot_table(index='hour',columns='day',values='weighted_lmp',aggfunc='mean')

nema_load = df_final[df_final.Location == 'NEMA'][['day','hour','weighted_lmp']]
nema_heatmap = nema_load.pivot_table(index='hour',columns='day',values='weighted_lmp',aggfunc='mean')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),sharex=True)
sns.heatmap(nema_heatmap, cmap='viridis', cbar=True,ax=ax1)
ax1.set_title('NEMA Weighted LMP')
sns.heatmap(vt_heatmap, cmap='viridis', cbar=True,ax=ax2)
ax2.set_title('VT Weighted LMP')

# 2. HEAT MAP
vt_load = df_final[df_final.Location == 'VT'][['day','hour','energy_normalized']]
vt_heatmap = vt_load.pivot_table(index='hour',columns='day',values='energy_normalized',aggfunc='mean')

nema_load = df_final[df_final.Location == 'NEMA'][['day','hour','energy_normalized']]
nema_heatmap = nema_load.pivot_table(index='hour',columns='day',values='energy_normalized',aggfunc='mean')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),sharex=True)
sns.heatmap(nema_heatmap, cmap='viridis', cbar=True,ax=ax1)
ax1.set_title('NEMA Normalized Demand')
sns.heatmap(vt_heatmap, cmap='viridis', cbar=True,ax=ax2)
ax2.set_title('VT Normalized Demand')

# 3. VIOLIN PLOT
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),sharex=True)

custom_palette = {'summer': '#FFB100', 'winter': '#202850'}
# sns.set_palette(custom_palette)

sns.violinplot(x="season", y="energy_normalized", data=df_final[df_final.Location =='NEMA'],ax=ax1,palette=custom_palette)
ax1.set_title('NEMA Normalized Energy Distribution by Season')

sns.violinplot(x="season", y="energy_normalized", data=df_final[df_final.Location =='VT'],ax=ax2,palette=custom_palette)
ax2.set_title('VT Normalized Energy Distribution by Season')

# 4. CUMULATIVE LMP COST BROKEN OUT

df_final['weighted_lmp_c'] = df_final.groupby(['Location'])['weighted_lmp'].cumsum()
df_final['weighted_loss_c'] = df_final.groupby(['Location'])['weighted_loss'].cumsum()
df_final['weighted_cong_c'] = df_final.groupby(['Location'])['weighted_congestion'].cumsum()
df_final['weighted_energyCost_c'] = df_final.groupby(['Location'])['weighted_energyCost'].cumsum()

df_final.groupby('Location').agg(weighted_lmp=('weighted_lmp','sum')) # Weighted LMP

plt.figure(figsize=(15,5))
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8),sharex=True)

custom_palette = {'VT': '#FEB000', 'NEMA': '#212850'}

sns.lineplot(data=df_final,x='datetime_hourly',y='weighted_lmp_c',hue='Location',ax=ax4,palette=custom_palette)
ax4.set_title('Weighted LMP')
ax4.set_ylabel('$ / MWh')

sns.lineplot(data=df_final,x='datetime_hourly',y='weighted_loss_c',hue='Location',ax=ax3,palette=custom_palette)
ax3.set_title('Weighted Loss Component')
ax3.set_ylabel('$ / MWh')

sns.lineplot(data=df_final,x='datetime_hourly',y='weighted_cong_c',hue='Location',ax=ax2,palette=custom_palette)
ax2.set_title('Weighted Congestion Cost')
ax2.set_ylabel('$ / MWh')

sns.lineplot(data=df_final,x='datetime_hourly',y='weighted_energyCost_c',hue='Location',ax=ax1,palette=custom_palette)
ax1.set_title('Weighted Energy Cost Component')
ax1.set_ylabel('$ / MWh');


# 5. FOUR GRAPH MATRIX

# LOAD DURATION CURVE
vt_ldc = df_final[(df_final.energy.notnull()) & (df_final.Location == 'VT')].sort_values('energy',ascending=False).energy.reset_index(drop=True)
ma_ldc = df_final[(df_final.energy.notnull()) & (df_final.Location == 'NEMA')].sort_values('energy',ascending=False).energy.reset_index(drop=True)

max_vt = max(vt_ldc)
max_ma = max(ma_ldc)

vt_ldc = vt_ldc / max_vt
ma_ldc = ma_ldc / max_ma

ldc_index_vt = np.linspace(0, 1, len(vt_ldc)) #np.arange(0,len(vt_ldc))
ldc_index_ma = np.linspace(0, 1, len(ma_ldc)) #np.arange(0,len(ma_ldc))

# REGR:
t_regr = temp_regression(df_final,'lmp')

cooling_x = np.arange(75,100,1)
heating_x = np.arange(0,40,1)

custom_palette = {'VT': '#FEB000', 'NEMA': '#212850'}

# Create a 2x2 grid of subplots and obtain axis handles
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))

# Plot data on each subplot using Seaborn
sns.scatterplot(data = df_final, x='t_f', y='lmp',hue='Location',alpha=.5, ax=ax1,palette=custom_palette)
sns.lineplot(x=cooling_x,y=cooling_x * t_regr.iloc[3,2] + t_regr.iloc[3,1],ax=ax1,color='#212850',linewidth=3,linestyle='dashed')
sns.lineplot(x=cooling_x,y=cooling_x * t_regr.iloc[2,2] + t_regr.iloc[2,1],ax=ax1,color='#ce9106ff',linewidth=3,linestyle='dashed')
sns.lineplot(x=heating_x,y=heating_x * t_regr.iloc[0,2] + t_regr.iloc[0,1],ax=ax1,color='#ce9106ff',linewidth=3,linestyle='dashed')
sns.lineplot(x=heating_x,y=heating_x * t_regr.iloc[1,2] + t_regr.iloc[1,1],ax=ax1,color='#212850',linewidth=3,linestyle='dashed')
ax1.set_title('Price vs. Temperature')

sns.scatterplot(data = df_final, x='t_f', y='energy_normalized',hue='Location',alpha=.5, ax=ax2,palette=custom_palette)
e_regr = temp_regression(df_final,'energy_normalized')
sns.lineplot(x=cooling_x,y=cooling_x * e_regr.iloc[3,2] + e_regr.iloc[3,1],ax=ax2,color='#212850',linewidth=3,linestyle='dashed')
sns.lineplot(x=cooling_x,y=cooling_x * e_regr.iloc[2,2] + e_regr.iloc[2,1],ax=ax2,color='#ce9106ff',linewidth=3,linestyle='dashed')
sns.lineplot(x=heating_x,y=heating_x * e_regr.iloc[0,2] + e_regr.iloc[0,1],ax=ax2,color='#ce9106ff',linewidth=3,linestyle='dashed')
sns.lineplot(x=heating_x,y=heating_x * e_regr.iloc[1,2] + e_regr.iloc[1,1],ax=ax2,color='#212850',linewidth=3,linestyle='dashed')
ax2.set_title('Demand vs. Temperature')

sns.scatterplot(data = df_final, x='lmp', y='energy',hue='Location',ax=ax3,palette=custom_palette)
ax3.set_title('Subplot 3: Demand vs. Price')

# sns.lineplot(data = vt_ldc, x='Index', y='Load',color='Blue', ax=ax4)
sns.lineplot(x=ldc_index_vt,y=vt_ldc,label='VT',ax=ax4,color = '#FEB000')
sns.lineplot(x=ldc_index_ma,y=ma_ldc,label='MA',ax=ax4,color = '#212850')
ax4.legend()
ax4.set_xlabel('Duration (%)')
ax4.set_ylabel('Demand (%)');
ax4.set_title('Subplot 4: Load Duration Curve')

# Adjust layout
plt.tight_layout()
plt.show()

########################################################################################################################
# 6. PEAK PROFILES

top_n = 5
max_index = df_final.groupby(['Location','season'])['energy'].idxmax()#indexid#.nlargest(top_n).index
result_df = df_final.iloc[max_index][['Location','day']].reset_index(drop=True)
top_season_peaks = result_df.merge(df_final)

height = 4
subplot_width = 6

aspect = subplot_width / height

custom_palette = {'summer': '#FEB000', 'winter': '#212850'}

grid = sns.FacetGrid(top_season_peaks, col="Location", hue="season",col_wrap=1, height=3,aspect=aspect,sharey=False,legend_out=True,
                    palette=custom_palette)

# Map a plot type (e.g., scatter plot) to the grid
grid.map(sns.lineplot, "hour", "energy", alpha=0.7)

grid.add_legend(title="Peak Season")

# Add a title for each subplot
grid.set_titles("{col_name}")

# grid.set(xlim=(1, 24))
x_ticks = [4, 8, 12, 16, 20, 24]  # Adjust based on your requirements
grid.set(xticks=x_ticks)

# Show the plot
plt.show()