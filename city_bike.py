import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date
from scipy.stats import pearsonr, spearmanr

def read_and_modify_data():
    bikeRaw = pd.read_csv("data/od-trips-2016/2016-05.csv")
    weatherRaw = pd.read_csv("data/weather-kaisaniemi-2016.csv")
    for year in range(2016, 2020):
        if year > 2016:
            bikeRaw = pd.concat([bikeRaw, pd.read_csv(
                "data/od-trips-"+str(year)+"/"+str(year)+"-05.csv")], ignore_index=True)
        bikeRaw = pd.concat([bikeRaw, pd.read_csv(
            "data/od-trips-"+str(year)+"/"+str(year)+"-06.csv")], ignore_index=True)
        bikeRaw = pd.concat([bikeRaw, pd.read_csv(
            "data/od-trips-"+str(year)+"/"+str(year)+"-07.csv")], ignore_index=True)
        bikeRaw = pd.concat([bikeRaw, pd.read_csv(
            "data/od-trips-"+str(year)+"/"+str(year)+"-08.csv")], ignore_index=True)
        bikeRaw = pd.concat([bikeRaw, pd.read_csv(
            "data/od-trips-"+str(year)+"/"+str(year)+"-09.csv")], ignore_index=True)
        bikeRaw = pd.concat([bikeRaw, pd.read_csv(
            "data/od-trips-"+str(year)+"/"+str(year)+"-10.csv")], ignore_index=True)
        weatherRaw = pd.concat([weatherRaw, pd.read_csv(
            "data/weather-kaisaniemi-"+str(year)+".csv")])

    # KAISANIEMI BIKE DATA:
    bikeRaw_departures = bikeRaw[(bikeRaw['Departure station name'] == 'Kaisaniemenpuisto')]
    bikeKaisaData_departures = bikeRaw_departures.drop(['Return', 'Departure station id', 'Return station name','Covered distance (m)', 'Duration (sec.)', 'Return station id'], axis=1)
    bikeKaisaData_departures['DateTime'] = [d.split("T")[0] + " " + d.split("T")[1][:2]+":00" for d in bikeKaisaData_departures['Departure']]
    bikeKaisaData_departures = bikeKaisaData_departures[["DateTime"]]
    bikeKaisaData_departures = pd.DataFrame(
    bikeKaisaData_departures[['DateTime']].value_counts())
    bikeKaisaData_departures.reset_index(level=0, inplace=True)
    bikeKaisaData_departures.columns = ['DateTime', 'Departures']
    bikeKaisaData_departures.sort_values('DateTime')

    bikeRaw_returns = bikeRaw[(bikeRaw['Return station name'] == 'Kaisaniemenpuisto')]
    bikeKaisaData_returns = bikeRaw_returns.drop(['Departure', 'Departure station id', 'Return station name','Covered distance (m)', 'Duration (sec.)', 'Return station id'], axis=1)
    bikeKaisaData_returns['DateTime'] = [d.split("T")[0] + " " + d.split("T")[1][:2]+":00" for d in bikeKaisaData_returns['Return']]
    bikeKaisaData_returns = bikeKaisaData_returns[["DateTime"]]
    bikeKaisaData_returns = pd.DataFrame(bikeKaisaData_returns[['DateTime']].value_counts())
    bikeKaisaData_returns.reset_index(level=0, inplace=True)
    bikeKaisaData_returns.columns = ['DateTime', 'Returns']
    bikeKaisaData_returns.sort_values('DateTime')

    bikeKaisaData = bikeKaisaData_departures.merge(bikeKaisaData_returns, how='outer')

    # KAISANIEMI WEATHER DATA:
    weatherRaw = weatherRaw[(weatherRaw.m >= 5) & (weatherRaw.m <= 10)]
    weatherRaw['d'][weatherRaw['d'] < 10] = '0'+weatherRaw['d'].astype(str)
    weatherRaw['m'][weatherRaw['m'] < 10] = '0'+weatherRaw['m'].astype(str)
    dateTimeColumn = weatherRaw["Year"].astype(str)+'-'+weatherRaw["m"].astype(
        str)+'-'+weatherRaw["d"].astype(str)+" "+weatherRaw["Time"].astype(str)
    weatherRaw.insert(0, "DateTime", dateTimeColumn)
    weatherRaw.drop(['Year', 'm', 'd', 'Time', 'Time zone'],
                    axis=1, inplace=True)

    weatherRaw.columns = ["DateTime", "Clouds", "Rain amount", "Rain intensity", "Air temperature"]

    data = weatherRaw.merge(
        bikeKaisaData, on=["DateTime"], how='outer').fillna(0.0)
    data['Date'] = [d.split()[0] for d in data['DateTime']]
    data['Hour'] = [d.split()[1] for d in data['DateTime']]
    data.drop("DateTime", axis=1, inplace=True)
    data = data[["Date", "Hour", "Departures", "Returns",
                 "Air temperature", "Rain amount", "Rain intensity"]]

    # data["Weekday"] = [date.fromisoformat(str(d)).strftime('%A') for d in data.Date]  # Add weekday column to data
    data["Weekday"] = [date.fromisoformat(str(d)).weekday() for d in data.Date]  # Add weekday column to data
    data["Month"] = [date.fromisoformat(str(d)).month for d in data.Date]  # Add month column to data
    data["Hour"] = [d.split(":")[0] for d in data["Hour"]]  # Add hour column
    data["Date"] = [d[5:] for d in data["Date"]]  # Remove year from date

    # Categorize the rain intensity
    labels = [0, 1, 2]
    cut_values = [0.0, 2.5, 7.6, 50]
    categorized_rain = pd.cut(data["Rain intensity"], bins=cut_values, labels=labels, include_lowest=True)
    data["Rain intensity class"] = categorized_rain

    return data

def write_data(data):
    data.to_csv('./bike_data.csv')
    
def plot_data(X, y, title, cdata=None):
    fig, axs = plt.subplots(3, 2, num=title)
    axs[0, 0].set_xlabel("Date")
    axs[0, 0].set_ylabel(title)
    for tick in axs[0, 0].get_xticklabels(): tick.set_rotation(45)
    axs[0, 0].scatter(X[:, 0], y, c=cdata, cmap='winter')
    axs[0, 1].set_xlabel("Hour")
    axs[0, 1].set_ylabel(title)
    axs[0, 1].scatter(X[:, 1], y, c=cdata, cmap='winter')
    axs[1, 0].set_xlabel("Air temperature")
    axs[1, 0].set_ylabel(title)
    axs[1, 0].scatter(X[:, 2], y, c=cdata, cmap='winter')
    axs[1, 1].set_xlabel("Rain intensity")
    axs[1, 1].set_ylabel(title)
    axs[1, 1].scatter(X[:, 3], y, c=cdata, cmap='winter')
    axs[2, 0].set_xlabel("Weekday")
    axs[2, 0].set_ylabel(title)
    axs[2, 0].scatter(X[:, 4], y, c=cdata, cmap='winter')
    axs[2, 1].set_xlabel("Rain intensity class")
    axs[2, 1].set_ylabel(title)
    axs[2, 1].scatter(X[:, 5], y, c=cdata, cmap='winter')


def main():

    data = read_and_modify_data()
    write_data(data)
    #data = pd.read_csv('bike_data.csv')

    #Filter here if needed
    data_1 = data[ (data.Month==6) & (data.Weekday==5) ]
    data_2 = data[ (data.Month==6) & (data.Weekday==0) ]

    # Split into features and labels
    y_dep = data_1["Departures"].to_numpy()
    X = data_1[["Date", "Hour", "Air temperature", "Rain intensity", "Weekday", "Rain intensity class"]].to_numpy()

    plot_data(X, y_dep, "Sunday")

    y_dep_1 = data_2["Departures"].to_numpy()
    X_1 = data_2[["Date", "Hour", "Air temperature", "Rain intensity", "Weekday", "Rain intensity class"]].to_numpy()

    plot_data(X_1, y_dep_1, "Monday")
    
    plt.show()

if __name__=="__main__":
    main()
