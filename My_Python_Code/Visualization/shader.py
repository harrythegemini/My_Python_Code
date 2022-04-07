import pandas as pd
import hvplot.pandas
import bokeh
import datashader as ds


# read in data
dat = pd.read_csv("data/household_power_consumption.txt", sep = ";")
dat.head()

# Munging
## Datetime
dat["datetime"] = pd.to_datetime(dat["Date"] + ' ' + dat["Time"].astype(str))

list(dat.columns)

## Remove null value
dat = dat.dropna(axis = 0)


## String to numeric
dat["Global_active_power"] = pd.to_numeric(dat["Global_active_power"],errors="coerce")
dat["Global_reactive_power"] = pd.to_numeric(dat["Global_reactive_power"])
dat["Voltage"] = pd.to_numeric(dat["Voltage"])
dat["Global_intensity"] = pd.to_numeric(dat["Global_intensity"])
dat["Sub_metering_1"] = pd.to_numeric(dat["Sub_metering_1"])
dat["Sub_metering_2"] = pd.to_numeric(dat["Sub_metering_2"])
dat["Sub_metering_3"] = pd.to_numeric(dat["Sub_metering_3"])

# Plotting
f = dat.hvplot.scatter(x = "datetime",  y = "Global_active_power", datashade = True， title = "Global Active Power by datetime")
hvplot.show(f)

# Save the dataset to csv
dat.to_csv("data/power.csv")

# Save the dataset to parquet
dat.to_parquet("power.parquet")




