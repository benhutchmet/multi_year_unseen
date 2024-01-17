# Functions for UNSEEN work

# Local imports
import os
import sys
import glob

# Third party imports
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
from scipy import stats

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
    obs_data = np.zeros([n_years, n_months])

    # Print the shape of the obs data
    print("Shape of obs data:", obs_data.shape)

    # Loop over the years
    for year in tqdm(years, desc="Processing years"):
        # Extract the time slice between
        obs_slice = obs.sel(time=slice(f"{year}-12-01",
                                       f"{year + avg_period}-11-30"))

        # Extract the data
        obs_data[year - start_year, :] = obs_slice.values

    # Print the shape of the obs data
    print("Shape of obs data:", obs_data.shape)

    # Return the obs data
    return obs_data

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
