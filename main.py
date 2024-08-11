import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('weather_classification_data.csv')
weathavg = {}
#print(df.head()) #prints top of dataset

humidness = df['Humidity']
weathtype = df['Weather Type']

#print(humidness.values, weathtype.values)

for i in range(len(df)):
    print(humidness.values[i])
    if df['Weather Type'].values[i] not in weathavg:
        weathavg[df['Weather Type'].values[i]] = [0, 0]
    weathavg[df['Weather Type'].values[i]][0] += int(df['Humidity'].values[i])
    weathavg[df['Weather Type'].values[i]][1] += 1
print(weathavg)

weather_types = list(weathavg.keys())
average_temps = [weathavg[weather][0] / weathavg[weather][1] for weather in weather_types]

# Create the histogram (bar chart)
plt.bar(weather_types, average_temps, color='skyblue', width=0.5)

# Add labels and title
plt.xlabel('Weather Type')
plt.ylabel('Average Humidity')
plt.title('Average Humidity by Weather Type')
for i, temp in enumerate(average_temps):
    plt.text(i, temp + 0.1, f'{temp:.2f}', ha='center', va='bottom')
plt.grid(True, which='both', axis='y', linestyle='-', linewidth=0.7)  # Adds gridlines only to the y-axis
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability if needed

# Display the plot
plt.show()

#average humidity on rainy cloudy sunny days

#make dict of weather types, get corresponding humidity, add to it and get avg

