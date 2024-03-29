# Functions for UNSEEN work

# Local imports
import os
import sys
import glob
import random

# Third party imports
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Path to modules
import dictionaries as dicts

# Function for loading each of the ensemble members for a given model
def load_model_data(model_variable: str,
                    model: str,
                    experiment: str,
                    start_year: int,
                    end_year: int,
                    avg_period: int,
                    grid: dict):
    """
    Function for loading each of the ensemble members for a given model
    
    Parameters
    ----------
    
    model_variable: str
        The variable to load from the model data
        E.g. 'pr' for precipitation
        
    model: str
        The model to load the data from
        E.g. 'HadGEM3-GC31-MM'
    
    experiment: str
        The experiment to load the data from
        E.g. 'historical' or 'dcppA-hindcast'

    start_year: int
        The start year for the data
        E.g. 1961

    end_year: int
        The end year for the data
        E.g. 1990

    avg_period: int
        The number of years to average over
        E.g. 1 for 1-year, 5 for 5-year, etc.

    grid: dict
        The grid to load the data over

    Returns
    -------

    model_data dict[str, xr.DataArray]
        A dictionary of the model data for each ensemble member
        E.g. model_data['r1i1p1f1'] = xr.DataArray
    """

    # Set up the years
    years = np.arange(start_year, end_year + 1)

    # Set the n years
    n_years = len(years)

    # Extract the lon and lat bounds
    lon1, lon2 = grid['lon1'], grid['lon2'] ; lat1, lat2 = grid['lat1'], grid['lat2']

    # Set up the directory where the csv files are stored
    csv_dir = "/home/users/benhutch/multi_year_unseen/paths"

    # Assert that the folder exists
    assert os.path.exists(csv_dir), "The csv directory does not exist"

    # Assert that the folder is not empty
    assert os.listdir(csv_dir), "The csv directory is empty"

    # Extract the csv file for the model and experiment
    csv_file = glob.glob(f"{csv_dir}/*.csv")[0]

    # Verify that the csv file exists
    assert csv_file, "The csv file does not exist"

    # Load the csv file
    csv_data = pd.read_csv(csv_file)

    # Extract the path for the model and experiment and variable
    model_path = csv_data.loc[(csv_data['model'] == model) & (csv_data['experiment'] == experiment) & (csv_data['variable'] == model_variable), 'path'].values[0]

    print(model_path)

    # Assert that the model path exists
    assert os.path.exists(model_path), "The model path does not exist"

    # Assert that the model path is not empty
    assert os.listdir(model_path), "The model path is empty"

    # List the files in the model path
    model_files = os.listdir(model_path)

    # Create an empty list of files
    model_file_list = []

    no_members = 0

    # Loop over the years
    for year in years:
        # Find all of the files for the given year
        year_files = [file for file in model_files if f"s{year}" in file]

        # # Print the year and the number of files
        # print(year, len(year_files))
        if year == years[0]:
            # Set the no members
            no_members = len(year_files)

        # Assert that the number of files is the same as the number of members
        assert len(year_files) == no_members, "The number of files is not the same as the number of members"

        # Append the year files to the model file list
        model_file_list.append(year_files)

    # Flatten the model file list
    model_file_list = [file for sublist in model_file_list for file in sublist]

    # Print the number of files
    print("Number of files:", len(model_file_list))

    # From the first file extract the number of lats and lons
    ds = xr.open_dataset(f"{model_path}/{model_file_list[0]}")

    # Extract the time series for the gridbox
    ds = ds.sel(lat=slice(lat1, lat2),
                lon=slice(lon1, lon2)).mean(dim=('lat','lon'))

    # Print the first time of the first file
    print("First time:", ds['time'][0].values)

    # Extract the first year from the first file
    first_year = int(str(ds['time'][0].values)[:4])

    # Print the first year
    print("First year:", first_year)

    # Assert that the first year is the same as the start year
    assert first_year == start_year, "The first year is not the same as the start year"

    # Print the window over which we are slicing the time
    print("Slicing over:", f"{first_year}-12-01", f"{first_year + avg_period}-11-30")

    # Extract the time slice between
    ds_slice = ds.sel(time=slice(f"{first_year}-12-01", f"{first_year + avg_period}-11-30"))

    # Extract the nmonths
    n_months = len(ds_slice['time'])

    # Print the number of months
    print("Number of months:", n_months)

    # Form the empty array to store the data
    model_data = np.zeros([n_years, no_members, n_months])

    # Print the shape of the model data
    print("Shape of model data:", model_data.shape)

    # Loop over the years
    for year in tqdm(years, desc="Processing years"):
        for member in tqdm(range(no_members), desc=f"Processing members for year {year}", leave=False):
            # Find the file for the given year and member
            file = [file for file in model_file_list if f"s{year}" in file and f"r{member + 1}i" in file][0]

            # Load the file
            ds = xr.open_dataset(f"{model_path}/{file}")

            # Extract the time series for the gridbox
            ds = ds.sel(lat=slice(lat1, lat2),
                        lon=slice(lon1, lon2)).mean(dim=('lat','lon'))

            # Extract the time slice between
            ds_slice = ds.sel(time=slice(f"{year}-12-01", f"{year + avg_period}-11-30"))

            # Extract the data
            model_data[year - start_year, member, :] = ds_slice[model_variable].values

    # p[rint the shape of the model data
    print("Shape of model data:", model_data.shape)

    # Return the model data
    return model_data


# Function for loading the observations
def load_obs_data(obs_variable: str,
                  regrid_obs_path: str,
                  start_year: int,
                  end_year: int,
                  avg_period: int,
                  grid: dict):
    """
    Function for loading the observations
    
    Parameters
    ----------

    obs_variable: str
        The variable to load from the model data
        E.g. 'si10' for sfcWind

    regrid_obs_path: str
        The path to the regridded observations

    start_year: int
        The start year for the data
        E.g. 1961

    end_year: int
        The end year for the data
        E.g. 1990

    avg_period: int
        The number of years to average over
        E.g. 1 for 1-year, 5 for 5-year, etc.

    grid: dict
        The grid to load the data over

    Returns

    obs_data: np.array
        The observations
    """

    # Set up the years
    years = np.arange(start_year, end_year + 1)

    # Set up the new years
    new_years = []

    # Set the n years
    n_years = len(years)

    # Extract the lon and lat bounds
    lon1, lon2 = grid['lon1'], grid['lon2'] ; lat1, lat2 = grid['lat1'], grid['lat2']

    # Open the obs
    obs = xr.open_mfdataset(regrid_obs_path,
                            combine='by_coords',
                              parallel=True)[obs_variable]
    
    # Combine the first two expver variables
    obs = obs.sel(expver=1).combine_first(obs.sel(expver=5))

    # Extract the time series for the gridbox
    obs = obs.sel(lat=slice(lat1, lat2),
                  lon=slice(lon1, lon2)).mean(dim=('lat','lon'))
    
    # Convert numpy.datetime64 to datetime
    final_time = obs['time'][-1].values.astype(str)

    # Extract the year and month
    final_year = int(final_time[:4])
    final_month = int(final_time[5:7])

    # If the final time is not november or december
    if not (final_month == 11 or final_month == 12):
        # Check that the final year - avg_period is not less than the end year
        if (final_year - 1) - avg_period < end_year:
            # Set the end year to the final year - avg_period
            end_year = (final_year - 1) - avg_period
    else:
        print("The final year has november or december")

    # Set the new years
    new_years = np.arange(start_year, end_year + 1)
    
    # Print the first time of the new years
    print("First time:", new_years[0])
    print("Last time:", new_years[-1])

    # Print the years we are slicing over
    print("Slicing over:", f"{start_year}-12-01", f"{start_year + avg_period}-11-30")

    # Extract the time slice between
    obs_slice = obs.sel(time=slice(f"{start_year}-12-01",
                                   f"{start_year + avg_period}-11-30"))
    
    # Extract the nmonths
    n_months = len(obs_slice['time'])

    # Print the number of months
    print("Number of months:", n_months)

    # Form the empty array to store the data
    obs_data = np.zeros([len(new_years), n_months])

    # Print the shape of the obs data
    print("Shape of obs data:", obs_data.shape)

    # Loop over the years
    for year in tqdm(new_years, desc="Processing years"):
        # We only have obs upt to jjuly 2023
        
        # Extract the time slice between
        obs_slice = obs.sel(time=slice(f"{year}-12-01",
                                       f"{year + avg_period}-11-30"))

        # Extract the data
        obs_data[year - start_year, :] = obs_slice.values

    # Print the shape of the obs data
    print("Shape of obs data:", obs_data.shape)

    # Set up the obs years
    obs_years = np.arange(new_years[0], new_years[-1] + 1)

    # Return the obs data
    return obs_data, obs_years

# Function for calculating the obs_stats
def calculate_obs_stats(obs_data: np.ndarray,
                        start_year: int,
                        end_year: int,
                        avg_period: int,
                        grid: dict):
    """
    Calculate the observations stats
    
    Parameters
    ----------
        
        obs_data: np.ndarray
            The observations data
            With shape (nyears, nmonths)

        start_year: int
            The start year for the data
            E.g. 1961

        end_year: int
            The end year for the data
            E.g. 1990

        avg_period: int
            The number of years to average over
            E.g. 1 for 1-year, 5 for 5-year, etc.

        grid: dict
            The grid to load the data over
        
    Returns
    -------
        
        obs_stats: dict
            A dictionary containing the obs stats
    
    """

    # Define the mdi
    mdi = -9999.0

    # Define the obs stats
    obs_stats = {
        'avg_period_mean': [],
        'mean': mdi,
        'sigma': mdi,
        'skew': mdi,
        'kurt': mdi,
        'start_year': mdi,
        'end_year': mdi,
        'avg_period': mdi,
        'grid': mdi,
        'min_20': mdi,
        'max_20': mdi,
        'min_10': mdi,
        'max_10': mdi,
        'min_5': mdi,
        'max_5': mdi,
        'min': mdi,
        'max': mdi,
        'sample_size': mdi
    }

    # Set the start year
    obs_stats['start_year'] = start_year

    # Set the end year
    obs_stats['end_year'] = end_year

    # Set the avg period
    obs_stats['avg_period'] = avg_period

    # Set the grid
    obs_stats['grid'] = grid

    # Process the obs
    obs_copy = obs_data.copy()

    # Take the mean over the 1th axis (i.e. over the 12 months)
    obs_year = np.mean(obs_copy, axis=1)

    # Set the average period mean
    obs_stats['avg_period_mean'] = obs_year

    # Get the sample size
    obs_stats['sample_size'] = len(obs_year)

    # Take the mean over the 0th axis (i.e. over the years)
    obs_stats['mean'] = np.mean(obs_year)

    # Take the standard deviation over the 0th axis (i.e. over the years)
    obs_stats['sigma'] = np.std(obs_year)

    # Take the skewness over the 0th axis (i.e. over the years)
    obs_stats['skew'] = stats.skew(obs_year)

    # Take the kurtosis over the 0th axis (i.e. over the years)
    obs_stats['kurt'] = stats.kurtosis(obs_year)

    # Take the min over the 0th axis (i.e. over the years)
    obs_stats['min'] = np.min(obs_year)

    # Take the max over the 0th axis (i.e. over the years)
    obs_stats['max'] = np.max(obs_year)

    # Take the min over the 0th axis (i.e. over the years)
    obs_stats['min_5'] = np.percentile(obs_year, 5)

    # Take the max over the 0th axis (i.e. over the years)
    obs_stats['max_5'] = np.percentile(obs_year, 95)

    # Take the min over the 0th axis (i.e. over the years)
    obs_stats['min_10'] = np.percentile(obs_year, 10)

    # Take the max over the 0th axis (i.e. over the years)
    obs_stats['max_10'] = np.percentile(obs_year, 90)

    # Take the min over the 0th axis (i.e. over the years)
    obs_stats['min_20'] = np.percentile(obs_year, 20)

    # Take the max over the 0th axis (i.e. over the years)
    obs_stats['max_20'] = np.percentile(obs_year, 80)

    # Return the obs stats
    return obs_stats

# Write a function which does the plotting
def plot_events(model_data: np.ndarray,
                obs_data: np.ndarray,
                obs_stats: dict,
                start_year: int,
                end_year: int,
                bias_adjust: bool = True,
                figsize_x: int = 10,
                figsize_y: int = 10):
    """
    Plots the events on the same axis.

    Parameters
    ----------

    model_data: np.ndarray
        The model data
        With shape (nyears, nmembers, nmonths)

    obs_data: np.ndarray
        The observations data
        With shape (nyears, nmonths)

    obs_stats: dict
        A dictionary containing the obs stats

    start_year: int
        The start year for the data
        E.g. 1961

    end_year: int
        The end year for the data
        E.g. 1990

    bias_adjust: bool
        Whether to bias adjust the model data
        Default is True

    figsize_x: int
        The figure size in the x direction
        Default is 10

    figsize_y: int
        The figure size in the y direction
        Default is 10

    Returns
    -------
    None
    """

    # Set up the years
    years = np.arange(start_year, end_year + 1)

    # Take the mean over the 2th axis (i.e. over the months)
    # For the model data
    model_year = np.mean(model_data, axis=2)

    # Take the mean over the 1th axis (i.e. over the members)
    # For the obs data
    obs_year = np.mean(obs_data, axis=1)    

    # if the bias adjust is True
    if bias_adjust:
        print("Bias adjusting the model data")
        
        # Flatten the model data
        model_flat = model_year.flatten()

        # Find the difference between the model and obs
        bias = np.mean(model_flat) - np.mean(obs_year)

        # Add the bias to the model data
        model_year = model_year - bias

    # Set the figure size
    plt.figure(figsize=(figsize_x, figsize_y))

    # Plot the obs
    plt.scatter(years, obs_year, color='k', label='ERA5')

    # Plot the model data
    for i in range(model_year.shape[1]):

        # Separate data into two groups based on the condition
        below_20th = model_year[:, i] < obs_stats['min_20']
        above_20th = ~below_20th
        
        # Plot points below the 20th percentile with a label
        plt.scatter(years[below_20th], model_year[below_20th, i],
                    color='blue', alpha=0.8, label='model wind drought' if i == 0 else None)
        
        # Plot points above the 20th percentile without a label
        plt.scatter(years[above_20th], model_year[above_20th, i],
                    color='grey', alpha=0.8, label='HadGEM3-GC31-MM' if i == 0 else None)
        
    # Plot the 20th percentile
    plt.axhline(obs_stats['min_20'],
                 color='black', linestyle='-')
    
    # Plot the min
    plt.axhline(obs_stats['min'],
                 color='black', linestyle='--')
    
    # Add a legend in the upper left
    plt.legend(loc='upper left')

    # Add the axis labels
    plt.xlabel('Year')

    # Add the axis labels
    plt.ylabel('Wind speed (m/s)')

    # Show the plot
    plt.show()


# Write a function which does the bootstrapping to calculate the statistics
def model_stats_bs(model: np.ndarray,
                   nboot: int = 10000) -> dict:
    """
    Repeatedly samples the model data with replacement across its members to
    produce many samples equal in length to the reanalysis time series. This 
    gives a single pseudo-time series from which the moments of the distribution
    can be calculated. The process is repeated to give a distribution of the
    moments.

    Parameters
    ----------

    model: np.ndarray
        The model data
        With shape (nyears, nmembers, nmonths)

    nboot: int
        The number of bootstrap samples to take
        Default is 10000

    Returns
    -------

    model_stats: dict
        A dictionary containing the model stats with the following keys:
        'mean', 'sigma', 'skew', 'kurt'
    """

    # Set up the model stats
    model_stats = {
        'mean': [],
        'sigma': [],
        'skew': [],
        'kurt': []
    }

    # Set up the number of years
    n_years = model.shape[0]

    # Set up the number of members
    n_members = model.shape[1]

    # TODO: Does autocorrelation need to be accounted for?
    # If so, use a block bootstrap

    # Set up the arrays
    mean_boot = np.zeros(nboot) ; sigma_boot = np.zeros(nboot)

    skew_boot = np.zeros(nboot) ; kurt_boot = np.zeros(nboot)

    # Create the indexes for the ensemble members
    index_ens = range(n_members)

    # Loop over the number of bootstraps
    for iboot in tqdm(np.arange(nboot)):
        print(f"Bootstrapping {iboot + 1} of {nboot}")

        # Create the index for time
        ind_time_this = range(0, n_years)

        # Create an empty array to store the data
        model_boot = np.zeros([n_years])

        # Set the year index
        year_index = 0

        # Loop over the years
        for itime in ind_time_this:

            # Select a random ensemble member
            ind_ens_this = random.choices(index_ens)

            # Logging
            print(f"itime is {itime} of {n_years}")
            print(f"year_index is {year_index} of {n_years} "
                  f"iboot is {iboot} of {nboot} "
                  f"ind_ens_this is {ind_ens_this}")
            
            # Extract the data
            model_boot[year_index] = model[itime, ind_ens_this]

            # Increment the year index
            year_index += 1

        
        # Calculate the mean
        mean_boot[iboot] = np.mean(model_boot)

        # Calculate the sigma
        sigma_boot[iboot] = np.std(model_boot)

        # Calculate the skew
        skew_boot[iboot] = stats.skew(model_boot)

        # Calculate the kurtosis
        kurt_boot[iboot] = stats.kurtosis(model_boot)

    # Append the mean to the model stats
    model_stats['mean'] = mean_boot

    # Append the sigma to the model stats
    model_stats['sigma'] = sigma_boot

    # Append the skew to the model stats
    model_stats['skew'] = skew_boot

    # Append the kurt to the model stats
    model_stats['kurt'] = kurt_boot

    # Return the model stats
    return model_stats
    

# Write a function which plots the four moments
def plot_moments(model_stats: dict,
                 obs_stats: dict,
                 figsize_x: int = 10,
                 figsize_y: int = 10,
                 save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots/") -> None:
    """
    Plot the four moments of the distribution of the model data and the
    observations.
    
    Parameters
    ----------
    
    model_stats: dict
        A dictionary containing the model stats with the following keys:
        'mean', 'sigma', 'skew', 'kurt'

    obs_stats: dict
        A dictionary containing the obs stats

    figsize_x: int
        The figure size in the x direction
        Default is 10

    figsize_y: int
        The figure size in the y direction
        Default is 10

    save_dir: str
        The directory to save the plots to
        Default is "/gws/nopw/j04/canari/users/benhutch/plots/"

    Output
    ------

    None
    """
    
    # Set up the figure as a 2x2
    fig, axs = plt.subplots(2, 2, figsize=(figsize_x, figsize_y))

    ax1, ax2, ax3, ax4 = axs.ravel()

    # Plot the mean
    ax1.hist(model_stats['mean'], bins=100, density=True,
            color='red', label='model')
    
    # Plot the mean of the obs
    ax1.axvline(obs_stats['mean'], color='black', linestyle='-',
                label='ERA5')
    
    # Calculate the position of the obs mean in the distribution
    obs_mean_pos = stats.percentileofscore(model_stats['mean'], obs_stats['mean'])

    # Add a title
    ax1.set_title(f'Mean, {obs_mean_pos:.2f}%')

    # Include a textbox in the top right corner
    ax1.text(0.95, 0.95, "a)", transform=ax1.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='right',
            bbox=dict(boxstyle='square', facecolor='white', alpha=0.5),
            zorder=100)

    # Plot the skewness
    ax2.hist(model_stats['skew'], bins=100, density=True,
            color='red', label='model')

    # Plot the skewness of the obs
    ax2.axvline(obs_stats['skew'], color='black', linestyle='-',
                label='ERA5')
    
    # Calculate the position of the obs skewness in the distribution
    obs_skew_pos = stats.percentileofscore(model_stats['skew'], obs_stats['skew'])

    # Add a title
    ax2.set_title(f'Skewness, {obs_skew_pos:.2f}%')

    # Include a textbox in the top right corner
    ax2.text(0.95, 0.95, "b)", transform=ax2.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='right',
            bbox=dict(boxstyle='square', facecolor='white', alpha=0.5),
            zorder=100)

    # Plot the kurtosis
    ax3.hist(model_stats['kurt'], bins=100, density=True,
            color='red', label='model')
    
    # Plot the kurtosis of the obs
    ax3.axvline(obs_stats['kurt'], color='black', linestyle='-',
                label='ERA5')
    
    # Calculate the position of the obs kurtosis in the distribution
    obs_kurt_pos = stats.percentileofscore(model_stats['kurt'], obs_stats['kurt'])

    # Add a title
    ax3.set_title(f'Kurtosis, {obs_kurt_pos:.2f}%')

    # Include a textbox in the top right corner
    ax3.text(0.95, 0.95, "c)", transform=ax3.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='right',
            bbox=dict(boxstyle='square', facecolor='white', alpha=0.5),
            zorder=100)

    # Plot the sigma
    ax4.hist(model_stats['sigma'], bins=100, density=True,
            color='red', label='model')
    
    # Plot the sigma of the obs
    ax4.axvline(obs_stats['sigma'], color='black', linestyle='-',
                label='ERA5')
    
    # Calculate the position of the obs sigma in the distribution
    obs_sigma_pos = stats.percentileofscore(model_stats['sigma'], obs_stats['sigma'])

    # Add a title
    ax4.set_title(f'Standard deviation, {obs_sigma_pos:.2f}%')

    # Include a textbox in the top right corner
    ax4.text(0.95, 0.95, "d)", transform=ax4.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='right',
            bbox=dict(boxstyle='square', facecolor='white', alpha=0.5),
            zorder=100)
    
    return

# Write a function to plot the distribution of the model and obs data
def plot_distribution()


