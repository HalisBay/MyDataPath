import matplotlib.pyplot as plt
import pandas as pd
import folium
from folium.plugins import HeatMap

def allAnalysis(df):
    analyzData(df)
    analyzTrafficHours(df)
    analyzeMostCongestedAreas(df,3,5)
    createTrafficHeatMap(df,3,5)
    createCombinedTrafficMap(df,3,5)
    analyzeMonthlyTrends(df)
    analyzeWeekdayVsWeekend(df)
    analyzeSpeedCorrelation(df)
    createSpeedDensityCorrelationMap(df,3,5)

def analyzData(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['DATE_TIME'], df['AVERAGE_SPEED'], label='Average Speed')
    plt.xlabel("Date")
    plt.ylabel('Average Speed (km/h)')
    plt.title('Average Speed By Date')
    plt.xticks(rotation=45)
    plt.legend()
    plt.savefig('Outputs/AverageSpeed.png')

def analyzTrafficHours(df):
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
    df['HOUR'] = df['DATE_TIME'].dt.hour
    hourlyTraffic = df.groupby('HOUR')['NUMBER_OF_VEHICLES'].mean()

    plt.figure(figsize=(12,8))
    bars = plt.bar(hourlyTraffic.index, hourlyTraffic.values, color='skyblue', alpha=0.7)
    plt.title('Trafik Density By Hour', fontsize=18, fontweight='bold')
    plt.xlabel('Hours', fontsize=14)
    plt.ylabel('Average Number Of Vehicles', fontsize=14)
    plt.xticks(hourlyTraffic.index, rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, round(yval, 2), ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('Outputs/TrafficByHour.png')

def analyzeMostCongestedAreas(df, start_hour, end_hour):    
    df_selected_hour = df[(df['HOUR'] >= start_hour) & (df['HOUR'] <= end_hour)]
    grouped = df_selected_hour.groupby(['LATITUDE', 'LONGITUDE'])['NUMBER_OF_VEHICLES'].sum().reset_index()

    most_congested_areas = grouped.sort_values(by='NUMBER_OF_VEHICLES', ascending=False).head(10)
    print(most_congested_areas)

    map_obj = folium.Map(location=[41.015137, 28.979530], zoom_start=10)

    for idx, row in most_congested_areas.iterrows():
        folium.CircleMarker(
            location=[row['LATITUDE'], row['LONGITUDE']],
            radius=10,
            color='red',
            fill=True,
            fill_color='red',
            popup=f"Araç Sayısı: {row['NUMBER_OF_VEHICLES']}"
        ).add_to(map_obj)

    map_obj.save('Outputs/MostCongestedAreas.html')

def createTrafficHeatMap(df,start_hour,end_hour):
    df['HOUR'] = df['DATE_TIME'].dt.hour
    df_selected_hour = df[(df['HOUR'] >= start_hour) & (df['HOUR'] <= end_hour)]

    heat_data = [[row['LATITUDE'], row['LONGITUDE'], row['NUMBER_OF_VEHICLES']] for index, row in df_selected_hour.iterrows()]
    map_obj = folium.Map(location=[41.015137, 28.979530], zoom_start=10)
    HeatMap(heat_data, radius=15).add_to(map_obj)
    map_obj.save('Outputs/TrafficHeatMap.html')

def createCombinedTrafficMap(df, start_hour, end_hour):
    df['HOUR'] = df['DATE_TIME'].dt.hour
    df_selected_hour = df[(df['HOUR'] >= start_hour) & (df['HOUR'] <= end_hour)]

    # Isı haritası verileri
    heat_data = [[row['LATITUDE'], row['LONGITUDE'], row['NUMBER_OF_VEHICLES']] for index, row in df_selected_hour.iterrows()]

    # En yoğun noktaların verileri
    grouped = df_selected_hour.groupby(['LATITUDE', 'LONGITUDE'])['NUMBER_OF_VEHICLES'].sum().reset_index()
    most_congested_areas = grouped.sort_values(by='NUMBER_OF_VEHICLES', ascending=False).head(10)

    # Harita oluşturma
    map_obj = folium.Map(location=[41.015137, 28.979530], zoom_start=10)

    # Isı haritasını ekleme
    HeatMap(heat_data, radius=15).add_to(map_obj)

    # En yoğun noktaları ekleme
    for idx, row in most_congested_areas.iterrows():
        folium.CircleMarker(
            location=[row['LATITUDE'], row['LONGITUDE']],
            radius=10,
            color='red',
            fill=True,
            fill_color='red',
            popup=f"Araç Sayısı: {row['NUMBER_OF_VEHICLES']}"
        ).add_to(map_obj)
    
    map_obj.save('Outputs/CombinedTrafficMap.html')

def analyzeMonthlyTrends(df):
    df['MONTH'] = df['DATE_TIME'].dt.to_period('M')
    monthly_traffic = df.groupby('MONTH')['NUMBER_OF_VEHICLES'].sum()
    plt.figure(figsize=(12, 6))
    monthly_traffic.plot(kind='bar')
    plt.title('Monthly Traffic Trends')
    plt.xlabel('Month')
    plt.ylabel('Number of Vehicles')
    plt.xticks(rotation=25)
    plt.grid()
    plt.savefig('Outputs/MonthlyTrafficTrends.png')

def analyzeWeekdayVsWeekend(df):
    df['WEEKDAY'] = df['DATE_TIME'].dt.weekday
    weekday_traffic = df[df['WEEKDAY'] < 5]['NUMBER_OF_VEHICLES'].mean()
    weekend_traffic = df[df['WEEKDAY'] >= 5]['NUMBER_OF_VEHICLES'].mean()

    plt.figure(figsize=(8, 6))
    plt.bar(['Weekday', 'Weekend'], [weekday_traffic, weekend_traffic], color=['blue', 'orange'])
    plt.title('Weekday vs Weekend Traffic')
    plt.xlabel('Day Type')
    plt.ylabel('Average Number of Vehicles')
    plt.xticks(rotation=25)
    plt.grid()
    plt.savefig('Outputs/WeekdayVsWeekendTraffic.png')

def analyzeSpeedCorrelation(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['NUMBER_OF_VEHICLES'], df['AVERAGE_SPEED'], alpha=0.5, c='r')
    plt.title('Correlation between Traffic Density and Average Speed')
    plt.xlabel('Number of Vehicles')
    plt.ylabel('Average Speed (km/h)')
    plt.xticks(rotation = 25)
    plt.grid()
    plt.savefig('Outputs/SpeedCorrelation.png')

def createSpeedDensityCorrelationMap(df, start_hour, end_hour):
    df['HOUR'] = df['DATE_TIME'].dt.hour
    df_selected_hour = df[(df['HOUR'] >= start_hour) & (df['HOUR'] <= end_hour)]
    
    # Isı haritası verileri
    heat_data = [[row['LATITUDE'], row['LONGITUDE'], row['NUMBER_OF_VEHICLES']] for index, row in df_selected_hour.iterrows()]

    # Ortalama hız bilgisi ve yoğunluğun gösterimi
    speed_density_data = df_selected_hour.groupby(['LATITUDE', 'LONGITUDE']).agg({
        'NUMBER_OF_VEHICLES': 'sum',
        'AVERAGE_SPEED': 'mean'
    }).reset_index()

    # Harita oluşturma
    map_obj = folium.Map(location=[41.015137, 28.979530], zoom_start=10)

    # En yoğun noktaları ekleme
    for idx, row in speed_density_data.iterrows():
         
        if row['AVERAGE_SPEED'] > 70:
            color = 'green'
        elif row['AVERAGE_SPEED'] > 50:
            color = 'yellow'
        elif row['AVERAGE_SPEED'] > 30:
            color = 'orange'
        else:
            color = 'red'
    
        folium.CircleMarker(
            location=[row['LATITUDE'], row['LONGITUDE']],
            radius=10,
            color=color,
            fill=True,
            fill_color=color,
            popup=f"Average Speed: {row['AVERAGE_SPEED']:.2f} km/h\nVehicles: {row['NUMBER_OF_VEHICLES']}"
        ).add_to(map_obj)
    
    # Haritayı kaydetme
    map_obj.save('Outputs/SpeedDensityCorrelationMap.html')
