import numpy as np
import xarray as xr
from itertools import groupby
from scipy.stats import pearson3

# count consecutive 1 elements in a 1D array
myCount = lambda ar: [sum(val for _ in group) for val, group in groupby(ar) if val==1]
# function version
def count_consecutive_ones(arr):
    return [sum(1 for _ in group) for val, group in groupby(arr) if val == 1]


def fit_pearson3(values):
    """Fit Pearson III distribution and return shape, loc, scale."""
    shape, loc, scale = pearson3.fit(values)
    return shape, loc, scale
def ppf_pearson3(prob, shape, loc, scale):
    return pearson3.ppf(prob, shape, loc=loc, scale=scale)
def cdf_pearson3(flow, shape, loc, scale):
    return pearson3.cdf(flow, shape, loc=loc, scale=scale)
    
#error metrics
def nse(obs, sim):
    """
    Calculates The Nash–Sutcliffe efficiency (NSE) between two time series arrays.

    Arguments
    ---------
    sim: array-like
        Simulated time series array.
    obs: array-like
        Observed time series array.

    Returns
    -------
    nse: float
        nse calculated between the two arrays.
    """
    mask = np.logical_and(~np.isnan(obs), ~np.isnan(sim))
    return 1.0 - (np.sum((sim[mask]-obs[mask])**2)/np.sum((obs[mask]-np.mean(obs[mask]))**2))

def corr(obs, sim):
    mask = np.logical_and(~np.isnan(obs), ~np.isnan(sim))
    return np.corrcoef(sim[mask],obs[mask])[0,1]

def alpha(obs, sim):
    """
    Calculates ratio of variabilities of two time series arrays.

    Arguments
    ---------
    sim: array-like
        Simulated time series array.
    obs: array-like
        Observed time series array.

    Returns
    -------
    alpha: float
        variability ratio calculated between the two arrays.
    """
    mask = np.logical_and(~np.isnan(obs), ~np.isnan(sim))
    m_sim=np.mean(sim[mask])
    m_obs=np.mean(obs[mask])
    std_sim = np.std(sim[mask], ddof=1)
    std_obs = np.std(obs[mask], ddof=1)
    return (std_sim/m_sim)/(std_obs/m_obs)

def beta(obs, sim):
    """
    Calculates ratio of means of two time series arrays.

    Arguments
    ---------
    sim: array-like
        Simulated time series array.
    obs: array-like
        Observed time series array.

    Returns
    -------
    beta: float
        mean ratio calculated between the two arrays.
    """
    mask = np.logical_and(~np.isnan(obs), ~np.isnan(sim))
    return np.mean(sim[mask])/np.mean(obs[mask])

def kge(obs, sim):
    """
    Calculates the Kling-Gupta Efficiency (KGE) between two flow arrays.
    Arguments
    ---------
    sim: array-like
        Simulated time series array.
    obs: array-like
        Observed time series array.

    Returns
    -------
    kge: float
        Kling-Gupta Efficiency calculated between the two arrays.
    """
    kge = 1.0 - np.sqrt((corr(obs, sim)-1.0)**2 + (alpha(obs, sim)-1.0)**2 + (beta(obs, sim)-1.0)**2)
    return kge

def rmse(obs, sim):
    """
    Calculates root mean squared of error (rmse) between two flow arrays.
    Arguments
    ---------
    sim: array-like
        Simulated time series array.
    obs: array-like
        Observed time series array.

    Returns
    -------
    rmse: float
        rmse calculated between the two arrays.
    """
    mask = np.logical_and(~np.isnan(obs), ~np.isnan(sim))
    return np.sqrt(np.nanmean((sim[mask]-obs[mask])**2))


def pbias(obs, sim):
    """
    Calculates percentage bias between two flow arrays.
    Arguments
    ---------
    sim: array-like
        Simulated time series array.
    obs: array-like
        Observed time series array.

    Returns
    -------
    pbial: float
        percentage bias calculated between the two arrays.
    """
    mask = np.logical_and(~np.isnan(obs), ~np.isnan(sim))
    m_sim=np.mean(sim[mask])
    m_obs=np.mean(obs[mask])
    return np.sum((m_sim-m_obs))/np.sum(m_obs)

def bias(obs, sim):
    """
    Calculates percentage bias between two flow arrays.
    Arguments
    ---------
    sim: array-like
        Simulated time series array.
    obs: array-like
        Observed time series array.

    Returns
    -------
    pbial: float
        percentage bias calculated between the two arrays.
    """
    mask = np.logical_and(~np.isnan(obs), ~np.isnan(sim))
    return np.mean((sim[mask]-obs[mask]))

def mae(obs,sim):
    """
    Calculates mean absolute error (mae) two flow arrays.
    Arguments
    ---------
    sim: array-like
        Simulated time series array.
    obs: array-like
        Observed time series array.

    Returns
    -------
    mae: float
        mean absolute error calculated between the two arrays.
    """
    mask = np.logical_and(~np.isnan(obs), ~np.isnan(sim))
    return np.mean(np.absolute((sim[mask] - obs[mask])))


def month_mean_flow_err(obs, sim, sim_time):
    obs = np.asarray(obs)
    sim = np.asarray(sim)
    sim = np.reshape(sim, np.shape(obs))

    #month = [dt.month for dt in sim_time]
    month = sim_time.dt.month

    data = {'sim':sim, 'obs':obs, 'month':month}
    #df = pd.DataFrame(data, index = sim_time)
    df = pd.DataFrame(data)

    gdf = df.groupby(['month'])
    sim_month_mean = gdf.aggregate({'sim':np.nanmean})
    obs_month_mean = gdf.aggregate({'obs':np.nanmean})

    mth_err = np.nanmean(np.absolute(obs_month_mean['obs'] - sim_month_mean['sim']))
    return mth_err


# flow metrics

def FHV(dr: xr.DataArray, percent=0.98):
    """
    Calculates Flow duration curve high segment volume.
    Arguments
    ---------
    dr: xr.DataArray
        2D DataArray containing daily time series with coordinates of 'site', and 'time'
    Returns
    -------
    ds_FLV: xr.Dataset
        Dataset containing two 2D DataArrays 'ann_max_flow' and 'ann_max_day' with coordinate of 'year', and 'site'
    Notes
    -------
    None
    """
    prob=np.arange(1,float(len(dr['time']+1)))/(1+len(dr['time'])) #probability
    for d in range(len(prob)):
        idx=d
        if prob[d] > percent: break

    t_axis = dr.dims.index('time')
    flow_array_sort = np.sort(dr.values, axis=t_axis)
    if t_axis==0:
        FHV = np.mean(flow_array_sort[idx:,:], axis=t_axis)
    elif t_axis==1:
        FHV = np.mean(flow_array_sort[:,idx:], axis=t_axis)

    ds_FHV = xr.Dataset(data_vars=dict(FHV=(["site"], FHV)), coords=dict(site=dr['site']))

    return ds_FHV


def FLV(dr: xr.DataArray, percent=0.1):

    #Calculates Flow duration curve low segment volume. default is < 0.1
    # Yilmaz, K. K., et al. (2008), A process-based diagnostic approach to model evaluation: Applicationto the NWS distributed hydrologic model, Water Resour. Res., 44, W09417, doi:10.1029/2007WR006716

    prob=np.arange(1,float(len(dr['time']+1)))/(1+len(dr['time'])) #probability
    for d in range(len(prob)):
        idx=d
        if prob[d] > percent: break

    t_axis = dr.dims.index('time')
    flow_array_sort = np.sort(dr.values, axis=t_axis)
    if t_axis==0:
        FLV = np.sum(flow_array_sort[:idx,:], axis=t_axis)
    elif t_axis==1:
        FLV = np.sum(flow_array_sort[:,:idx], axis=t_axis)

    ds_FLV = xr.Dataset(data_vars=dict(FLV=(["site"], FLV)), coords=dict(site=dr['site']))

    return ds_FLV


def FMS(dr: xr.DataArray, percent_low=0.3, percent_high=0.7):

    #Calculate Flow duration curve midsegment slope (default between 30 and 70 percentile)

    prob=np.arange(1,float(len(dr['time']+1)))/(1+len(dr['time'])) #probability
    for d in range(len(prob)):
        idx_l=d
        if prob[d] > percent_low: break
    for d in range(len(prob)):
        idx_h=d
        if prob[d] > percent_high: break

    t_axis = dr.dims.index('time')
    flow_array_sort = np.sort(dr.values, axis=t_axis)

    if t_axis==0:
        high = flow_array_sort[idx_h,:]
        low  = flow_array_sort[idx_l,:]
    elif t_axis==1:
        high = flow_array_sort[:,idx_h]
        low  = flow_array_sort[:,idx_l]

    high_log = np.log10(high, where=0<high, out=np.nan*high)
    low_log  = np.log10(low,  where=0<low,  out=np.nan*low)

    FMS = (high_log-low_log)/(percent_high-percent_low)

    ds_FMS = xr.Dataset(data_vars=dict(FMS=(["site"], FMS)), coords=dict(site=dr['site']))

    return ds_FMS


def BFI(dr: xr.DataArray, alpha=0.925, npass=3, skip_time=30):

    #Calculate digital filter based Baseflow Index
    # Ladson, A. R., et al. (2013). A Standard Approach to Baseflow Separation Using The Lyne and Hollick Filter. Australasian Journal of Water Resources, 17(1), 25–34.
    # https://doi.org/10.7158/13241583.2013.11465417

    t_axis = dr.dims.index('time')
    tlen = len(dr['time'])

    q_total = dr.values
    if t_axis==1:
        q_total = q_total.T

    q_total_diff = q_total - np.roll(q_total, 1, axis=t_axis) # q(i)-q(i-1)
    q_fast = np.tile(q_total[skip_time+1,:], (tlen,1))

    count=1
    while count <= npass:
        for tix in np.arange(1, tlen):
            q_fast[tix,:] = alpha*q_fast[tix-1,:]+(1.0+alpha)/2.0*q_total_diff[tix,:]
            q_fast[tix,:] = np.where(q_fast[tix,:]>=0, q_fast[tix,:], 0)
        count+=1

    q_base = q_total-q_fast

    q_total_sum = np.nansum(q_total[skip_time:,:], axis=t_axis)
    BFI = np.nansum(q_base[skip_time:,:], axis=t_axis)/np.where(q_total_sum>0, q_total_sum, np.nan)

    ds_BFI = xr.Dataset(data_vars=dict(BFI=(["site"], BFI)), coords=dict(site=dr['site']))

    return BFI


def high_q_freq_dur(dr: xr.DataArray, p_thresh_mult=5, dayofyear='wateryear'):
    """
    freq_high_q: frequency of high-flow days (> 9 times the median daily flow) day/yr
    mean_high_q_dur: average duration of high-flow events over yr (number of consecutive days > 9 times the median daily flow) day
    """
    
    if dayofyear=='wateryear':
        smon=10; sday=1; emon=9; eday=30; yr_adj=1
    elif dayofyear=='calendar':
        smon=1; sday=1; emon=12; eday=31; yr_adj=0
    else:
        raise ValueError('Invalid argument for "dayofyear"')

    years = np.unique(dr.time.dt.year.values)[:-1]
    sites = dr['site'].values
    n_years, n_sites = len(years), len(sites)
    q_thresh = np.nanmedian(dr.values, axis=dr.get_axis_num('time')) * p_thresh_mult

    mean_dur = np.full((n_years, n_sites), 0.0, dtype='float32')
    freq = np.full((n_years, n_sites), 0.0, dtype='float32')

    for y_idx, yr in enumerate(years):
        q_array = dr.sel(time=slice(f'{yr}-{smon:02d}-{sday:02d}', f'{yr + yr_adj}-{emon:02d}-{eday:02d}')).values
        high_flow = q_array > q_thresh  # shape: (time, site)

        for s_idx in range(n_sites):
            dur = count_consecutive_ones(high_flow[:, s_idx])
            if dur:
                mean_dur[y_idx, s_idx] = np.mean(dur)
                freq[y_idx, s_idx] = len(dur)

    return xr.Dataset(
        data_vars=dict(
            mean_high_q_dur=(["year", "site"], mean_dur),
            freq_high_q=(["year", "site"], freq),
        ),
        coords=dict(
            year=years,
            site=sites,
        )
    )

def low_q_freq_dur(dr: xr.DataArray, percent=0.7, dayofyear='wateryear'):
    # : frequency of low-flow days (< 0.2 times the mean daily flow) day/yr
    # : average duration of low-flow events (number of consecutive days < 0.2 times the mean daily flow) day

    dayofyear='wateryear'
    if dayofyear=='wateryear':
        smon=10; sday=1; emon=9; eday=30; yr_adj=1
    elif dayofyear=='calendar':
        smon=1; sday=1; emon=12; eday=31; yr_adj=0
    else:
        raise ValueError('Invalid argument for "dayofyear"')

    years = np.unique(dr.time.dt.year.values)[:-1]
    sites = dr['site'].values
    n_years, n_sites = len(years), len(sites)
    q_thresh = np.nanmedian(dr.values, axis=dr.get_axis_num('time')) *0.2

    mean_dur = np.full((n_years, n_sites), 0.0, dtype='float32')
    freq = np.full((n_years, n_sites), 0.0, dtype='float32')

    for y_idx, yr in enumerate(years):
        q_array = dr.sel(time=slice(f'{yr}-{smon:02d}-{sday:02d}', f'{yr + yr_adj}-{emon:02d}-{eday:02d}')).values
        low_flow = q_array < q_thresh  # shape: (time, site)

        for s_idx in range(n_sites):
            dur = count_consecutive_ones(low_flow[:, s_idx])
            if dur:
                mean_dur[y_idx, s_idx] = np.mean(dur)
                freq[y_idx, s_idx] = len(dur)

    return xr.Dataset(
        data_vars=dict(
            mean_low_q_dur=(["year", "site"], mean_dur),
            freq_low_q=(["year", "site"], freq),
        ),
        coords=dict(
            year=years,
            site=sites,
        )
    )


def annual_max(dr: xr.DataArray, dayofyear='wateryear'):
    """
    Calculates annual maximum value and dayofyear.
    Arguments
    ---------
    dr: xr.DataArray
        2D DataArray containing daily time series with coordinates of 'site', and 'time'
    Returns
    -------
    ds_ann_max: xr.Dataset
        Dataset containing two 2D DataArrays 'ann_max_flow' and 'ann_max_day' with coordinate of 'year', and 'site'
    Notes
    -------
    dayofyear start with October 1st with dayofyear="wateryear" or January 1st with dayofyear="calendar".
    """
    if dayofyear=='wateryear':
        smon=10; sday=1; emon=9; eday=30; yr_adj=1
    elif dayofyear=='calendar':
        smon=1; sday=1; emon=12; eday=31; yr_adj=0
    else:
        raise ValueError('Invalid argument for "dayofyear"')

    years = np.unique(dr.time.dt.year.values)[:-1]

    ds_ann_max = xr.Dataset(data_vars=dict(
                    ann_max_flow =(["year", "site"], np.full((len(years),len(dr['site'])), np.nan, dtype='float32')),
                    ann_max_day  =(["year", "site"], np.full((len(years),len(dr['site'])), np.nan, dtype='float32')),
                    ),
                    coords=dict(year=years,
                                site=dr['site'],),
                    )

    for yr in years:
        time_slice=slice(f'{yr}-{smon}-{sday}',f'{yr+yr_adj}-{emon}-{eday}')

        max_flow_array = dr.sel(time=time_slice).max(dim='time').values # find annual max flow
        ix = np.argwhere(np.isnan(max_flow_array)) # if whole year data is missing, it is nan, so find that site
        max_day_array = dr.sel(time=time_slice).argmax(dim='time', skipna=False).values.astype('float') # find annual max flow day
        max_day_array[ix]=np.nan

        ds_ann_max['ann_max_flow'].loc[dict(year=yr)] = max_flow_array
        ds_ann_max['ann_max_day'].loc[dict(year=yr)]  = max_day_array

    return ds_ann_max


def annual_min(dr: xr.DataArray, dayofyear='wateryear'):
    """
    Calculates annual minimum value and dayofyear.
    Arguments
    ---------
    dr: xr.DataArray
        2D DataArray containing daily time series with coordinates of 'site', and 'time'
    Returns
    -------
    ds_ann_max: xr.Dataset
        Dataset containing two 2D DataArrays 'ann_min_flow' and 'ann_min_day' with coordinate of 'year', and 'site'
    """
    if dayofyear=='wateryear':
        smon=10; sday=1; emon=9; eday=30; yr_adj=1
    elif dayofyear=='calendar':
        smon=1; sday=1; emon=12; eday=31; yr_adj=0
    else:
        raise ValueError('Invalid argument for "dayofyear"')

    years = np.unique(dr.time.dt.year.values)[:-1]

    ds_ann_min = xr.Dataset(data_vars=dict(
                    ann_min_flow =(["year", "site"], np.full((len(years),len(dr['site'])), np.nan, dtype='float32')),
                    ann_min_day  =(["year", "site"], np.full((len(years),len(dr['site'])), np.nan, dtype='float32')),
                    ),
                    coords=dict(year=years,
                                site=dr['site'],)
                    )

    for yr in years:
        time_slice=slice(f'{yr}-{smon}-{sday}',f'{yr+yr_adj}-{emon}-{eday}')
        min_flow_array = dr.sel(time=time_slice).min(dim='time').values
        ix = np.argwhere(np.isnan(min_flow_array)) # if whole year data is missing, it is nan, so find that site
        min_day_array = dr.sel(time=time_slice).argmin(dim='time', skipna=False).values.astype('float') # find annual max flow day
        min_day_array[ix]=np.nan
        ds_ann_min['ann_min_flow'].loc[dict(year=yr)] = min_flow_array
        ds_ann_min['ann_min_day'].loc[dict(year=yr)] = min_day_array

    return ds_ann_min


def annual_centroid(dr: xr.DataArray, dayofyear='wateryear'):
    """
    Calculates annual time series centroid (in dayofyear).
    Arguments
    ---------
    dr: xr.DataArray
        2D DataArray containing daily time series with coordinates of 'site', and 'time'
    Returns
    -------
    ds_ann_max: xr.Dataset
        Dataset containing one 2D DataArrays 'ann_centroid_day' with coordinate of 'year', and 'site'
    """
    if dayofyear=='wateryear':
        smon=10; sday=1; emon=9; eday=30; yr_adj=1
    elif dayofyear=='calendar':
        smon=1; sday=1; emon=12; eday=31; yr_adj=0
    else:
        raise ValueError('Invalid argument for "dayofyear"')

    years = np.unique(dr.time.dt.year.values)[:-1]
    sites = dr['site'].values
    n_years, n_sites = len(years), len(sites)
    centroids = np.full((n_years, n_sites), np.nan, dtype='float32')

    for y_idx, yr in enumerate(years):
        q_array = dr.sel(time=slice(f'{yr}-{smon:02d}-{sday:02d}', f'{yr + yr_adj}-{emon:02d}-{eday:02d}'))
        if q_array.time.size == 0:
            continue
        weights = np.arange(q_array.time.size)
        q_vals = q_array.values  # shape: (time, site)

        valid_mask = ~np.isnan(q_vals)
        sums = np.where(valid_mask, q_vals, 0).sum(axis=0)
        weighted = np.where(valid_mask, q_vals * weights[:, None], 0).sum(axis=0)

        with np.errstate(divide='ignore', invalid='ignore'):
            centroids[y_idx, :] = np.where(sums != 0, weighted / sums, np.nan)

    return xr.Dataset(
        data_vars=dict(
            ann_centroid_day=(["year", "site"], centroids),
        ),
        coords=dict(year=years, site=sites)
    )


def season_mean(ds: xr.Dataset, calendar="standard"):
    # Make a DataArray with the number of days in each month, size = len(time)
    month_length = ds.time.dt.days_in_month

    # Calculate the weights by grouping by 'time.season'
    weights = (month_length.groupby("time.season") / month_length.groupby("time.season").sum())

    # Test that the sum of the weights for each season is 1.0
    np.testing.assert_allclose(weights.groupby("time.season").sum().values, np.ones(4))

    # Calculate the weighted average
    return (ds * weights).groupby("time.season").sum(dim="time")


def lp3_flood(dr: xr.DataArray):
    """
    Calculates annual time series centroid (in dayofyear).
    Arguments
    ---------
    dr: xr.DataArray
        2D DataArray containing daily time series with coordinates of 'site', and 'time'
    Returns
    -------
    ds_ann_max: xr.Dataset
        Dataset containing one 2D DataArrays 'ann_centroid_day' with coordinate of 'year', and 'site'
    """
    ann_max = dr.resample(time="YS").max()
    log_ann_max = np.log(ann_max)

    # Fit the Log-Pearson Type III distribution
    # Note: scipy.stats.pearson3 uses the skew as the shape parameter
    # and loc and scale are used to shift and scale the distribution
    # The parameters are calculated from the log-transformed data
    
    # Calculate parameters
    skew_log, mean_log, std_log = pearson3.fit(log_ann_max)
    fitted_dist = pearson3(skew_log, loc=mean_log, scale=std_log)

    # Apply function along 'year' axis for each 'site'
    shape, loc, scale = xr.apply_ufunc(
        fit_pearson3,
        log_ann_max,
        input_core_dims=[["time"]],
        output_core_dims=[[], [], []],
        vectorize=True,
        dask="parallelized",  # If da is a Dask array
        output_dtypes=[float, float, float],
    )
    # Calculate quantiles for different return periods
    probabilities = [0.01, 0.5, 0.8, 0.9, 0.95, 0.96, 0.98, 0.99]
    return_periods = [1 /(1-p) for p in probabilities]
    probs_da = xr.DataArray(probabilities, dims="return_period", coords={"return_period": return_periods})
    
    # compute values in log scale at quantiles from fitted 
    quantiles_log = xr.apply_ufunc(
        ppf_pearson3,
        probs_da,        # (return_period,)
        shape,           # (site,)
        loc,
        scale,
        input_core_dims=[["return_period"], [], [], []],
        output_core_dims=[["return_period"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    # Exponentiate to get flood magnitudes
    return xr.Dataset(dict(aff=np.exp(quantiles_log)))

def lp3_flood_return_period(dr: xr.DataArray, dr_ref_flow:xr.DataArray):
    """
    Calculates probability of flow based on reference LP3 AFF. flow is given by ds_ref_flow
    Arguments
    ---------
    dr: xr.DataArray
        2D DataArray containing daily time series with coordinates of 'site', and 'time'
    dr_ref_flow: xr.DataArray
        1D DataArray containing flow magnitude with coordinates of 'site'
    Returns
    -------
    ds_ann_max: xr.Dataset
        Dataset containing one 2D DataArrays 'ann_centroid_day' with coordinate of 'year', and 'site'
    """
    ann_max = dr.resample(time="YS").max()
    log_ann_max = np.log(ann_max)

    # Fit the Log-Pearson Type III distribution
    # Note: scipy.stats.pearson3 uses the skew as the shape parameter
    # and loc and scale are used to shift and scale the distribution
    # The parameters are calculated from the log-transformed data
    
    # Calculate parameters
    skew_log, mean_log, std_log = pearson3.fit(log_ann_max)
    fitted_dist = pearson3(skew_log, loc=mean_log, scale=std_log)

    # Apply function along 'time' axis for each 'site'
    shape, loc, scale = xr.apply_ufunc(
        fit_pearson3,
        log_ann_max,
        input_core_dims=[["time"]],
        output_core_dims=[[], [], []],
        vectorize=True,
        dask="parallelized",  # If da is a Dask array
        output_dtypes=[float, float, float],
    )

    # compute values in log scale at quantiles from fitted 
    cdf_log = xr.apply_ufunc(
        cdf_pearson3,
        np.log(dr_ref_flow),     # (site,)
        shape,           # (site,)
        loc,
        scale,
        input_core_dims=[[],[], [], []],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    # Exponentiate to get flood magnitudes
    return xr.Dataset(dict(cdf=cdf_log))