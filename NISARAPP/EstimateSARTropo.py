#!/usr/bin/env python3
# Created by Kunyi Chen on 2026-09-02

from datetime import datetime, timedelta
import numpy as np
import os
import time
import pyaps3 as pa
import isce,isceobj
from isceobj.Alos2Proc.Alos2ProcPublic import create_xml
import h5py
from multiprocessing import Pool

def auto_download_ERA5_data(rslc_h5: str, op_dir: str, snweDownload: list):
    '''
    Automatically download ERA5 data for Tropospheric correction

    Parameters
     ----------
     rslc_h5: str
        NISAR RSLC hdf5 file
     op_dir: str
        output directory for ERA5 data and troposphere delay datacube
    snweDownload: list
        list of [south, north, west, east] boundaries for ERA5 data download

    Returns
     -------
     None
    '''
    era5_folder = os.path.join(op_dir, 'ERA5')
    os.makedirs(era5_folder,exist_ok=True)
    with h5py.File(rslc_h5, 'r') as h5_obj:

        zeroDopplerEndTime = h5_obj['/science/LSAR/identification/zeroDopplerEndTime'][()].decode('utf-8')[:26]
        zeroDopplerEndTime = datetime.strptime(zeroDopplerEndTime, '%Y-%m-%dT%H:%M:%S.%f')
        zeroDopplerStartTime = h5_obj['/science/LSAR/identification/zeroDopplerStartTime'][()].decode('utf-8')[:26]
        zeroDopplerStartTime = datetime.strptime(zeroDopplerStartTime, '%Y-%m-%dT%H:%M:%S.%f')
        e_t = zeroDopplerEndTime.timestamp()
        s_t = zeroDopplerStartTime.timestamp()
        zeroDopplerMidTime = datetime.fromtimestamp((e_t+s_t)/2)
        yr = zeroDopplerMidTime.year
        mn = zeroDopplerMidTime.month
        dy = zeroDopplerMidTime.day
        hr = zeroDopplerMidTime.hour

        time = datetime(yr, mn, dy, hr, zeroDopplerMidTime.minute, zeroDopplerMidTime.second)
        time1 = datetime(yr, mn, dy, hr, 0, 0)
        time2 = time1 + timedelta(hours=1)

        #pyaps format
        date1 = '%4d%02d%02d'%(time1.year, time1.month, time1.day)
        date2 = '%4d%02d%02d'%(time2.year, time2.month, time2.day)
        date = '%4d%02d%02d'%(time.year, time.month, time.day)
        #hour format: HH
        hour1 = '%02d'%time1.hour
        hour2 = '%02d'%time2.hour

        flist1 = pa.ECMWFdload([date1],hour1,era5_folder, model='ERA5', snwe=snweDownload)
        flist2 = pa.ECMWFdload([date2],hour2,era5_folder, model='ERA5', snwe=snweDownload)


        weight1 = np.abs((time-time2).total_seconds())/3600
        weight2 = np.abs((time-time1).total_seconds())/3600
        h5_obj.close()

    return [flist1[0],flist2[0]], [weight1, weight2], date

def compute_troposphere(args:dict):

    try:
        wavelength = args['wavelength']
        rslc_fp = args['rslc_h5']
        temp = rslc_fp.split('/')[-1]
        if 'Track' in temp and 'Frame' in temp:
            date = temp.split('_')[6][:8]
        else:
            date = temp.split('_')[12][:8]
        print(f"Processing date: {date}...")
        slat_fp = args['slat']
        slon_fp = args['slon']
        shgt_fp = args['shgt']
        sinc_fp = args['sinc']
        InSARRangeLooks = args['InSARRangeLooks']
        InSARAzimuthLooks = args['InSARAzimuthLooks']
        op_dir = args['op_dir']

        img = isceobj.createImage()
        img.load(slat_fp + '.xml')
        width = img.width
        length = img.length
        ml_lat = np.fromfile(slat_fp, dtype=np.float32).reshape(length, width)
        ml_lon = np.fromfile(slon_fp, dtype=np.float32).reshape(length, width)
        ml_hgt = np.fromfile(shgt_fp, dtype=np.float32).reshape(length, width)
        inc = np.fromfile(sinc_fp, dtype=np.float32).reshape(length, width)
        snwe = [int(np.floor(ml_lat.min())-1),
                int(np.ceil(ml_lat.max())+1),
                int(np.floor(ml_lon.min())-1),
                int(np.ceil(ml_lon.max())+1)]

        print('Range of incidence angle: [{:.2f}, {:.2f}] degrees'.format(inc.min(), inc.max()))

        if not ml_hgt.shape == ml_lat.shape == ml_lon.shape == inc.shape:
            raise ValueError('Shape of multlooked height, latitude, longitude and incidence angle arrays do not match. Please check the input files and parameters.')
        
        print(f"Downloading ERA5 data for the following SNWE box: {snwe}...")
        flist, weight, date = auto_download_ERA5_data(rslc_fp, op_dir, snwe)
        print(f"Finished downloading ERA5 data. Computing troposphere phase delay...")

        tropo_dir = os.path.join(op_dir, 'tropo_delay')
        os.makedirs(tropo_dir, exist_ok=True)

        aps1 = pa.PyAPS(flist[0], dem=ml_hgt, inc=inc, lat=ml_lat, lon=ml_lon, grib='ERA5', verb=True)
        aps2 = pa.PyAPS(flist[1], dem=ml_hgt, inc=inc, lat=ml_lat, lon=ml_lon, grib='ERA5', verb=True)

        phs1 = np.zeros((aps1.ny, aps1.nx), dtype=np.float32)
        phs2 = np.zeros((aps2.ny, aps2.nx), dtype=np.float32)
        aps1.getdelay(phs1, wvl=wavelength)
        aps2.getdelay(phs2, wvl=wavelength)
        phs = weight[0] * phs1 + weight[1] * phs2
        looks = f"{InSARRangeLooks}rlks_{InSARAzimuthLooks}alks"
        tro_fp = os.path.join(tropo_dir, f'{date}_{looks}.tro')
        phs.astype(np.float32).tofile(tro_fp)
        create_xml(tro_fp, width, length, 'float')

        return (date,"SUCCESS","")
    except Exception as e:
        return (date,"FAILURE",str(e))

def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Estimate the Tropospheric Phase Delay for NISAR Time Series Processing with ERA5.')
    parser.add_argument('-config', dest='config', type=str, required=True,
            help = 'config yaml containing the input paths and parameters')
    parser.add_argument('-n', dest='n', type=str, required=False, default=4,
            help = 'number of parallels to run, default is 4')

    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        return parser.parse_args()

if __name__ == '__main__':
    inps = cmdLineParse()
    config_fp = inps.config
    n = int(inps.n)

    import yaml,json
    with open(config_fp, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    ref_fp = config['Path of Input']['Super Reference']
    ref_fp_fn = ref_fp.split('/')[-1]
    if 'Track' in ref_fp_fn and 'Frame' in ref_fp_fn:
        ref_date = ref_fp_fn.split('_')[6][:8]
    else:
        ref_date = ref_fp_fn.split('_')[12][:8]
    sed_fps = config['Path of Input']['Secondary List']
    sed_dates = []
    for sed_fp in sed_fps:
        sed_fp_fn = sed_fp.split('/')[-1]
        if 'Track' in sed_fp_fn and 'Frame' in sed_fp_fn:
            sed_date = sed_fp_fn.split('_')[6][:8]
        else:
            sed_date = sed_fp_fn.split('_')[12][:8]
        sed_dates.append(sed_date)
    rslc_fps = [ref_fp] + sed_fps
    dates = [ref_date] + sed_dates
    output_fp = config['Path of Output']['Output Dict']
    subsequent_num = config['Number of subsequent dates']
    InSARRangeLooks = config['Multilook Parameters']['InSAR Range Looks']
    InSARAzimuthLooks = config['Multilook Parameters']['InSAR Azimuth Looks']
    os.chdir(output_fp)
    dem_fp = os.path.join('dates', ref_date, f'hgt_{InSARRangeLooks}rlks_{InSARAzimuthLooks}alks.flt')
    lat_fp = os.path.join('dates', ref_date, f'lat_{InSARRangeLooks}rlks_{InSARAzimuthLooks}alks.flt')
    lon_fp = os.path.join('dates', ref_date, f'lon_{InSARRangeLooks}rlks_{InSARAzimuthLooks}alks.flt')
    inc_fp = os.path.join('dates', ref_date, f'inc_{InSARRangeLooks}rlks_{InSARAzimuthLooks}alks.flt')
    para_json_fp = os.path.join('dates', ref_date, 'slc_parameters.json')
    paras = json.load(open(para_json_fp, 'r', encoding='utf-8'))
    wavelength = paras['wavelength']

    tropo_op_dir = './dates_tro'
    os.makedirs(tropo_op_dir, exist_ok=True)

    tasks = []
    for i,date in enumerate(dates):
        rslc_fp = rslc_fps[i]
        args = {
            'wavelength': wavelength,
            'rslc_h5': rslc_fp,
            'slat': lat_fp,
            'slon': lon_fp,
            'shgt': dem_fp,
            'sinc': inc_fp,
            'InSARRangeLooks': InSARRangeLooks,
            'InSARAzimuthLooks': InSARAzimuthLooks,
            'op_dir': tropo_op_dir
        }
        tasks.append(args)
    
    results = []
    for task in tasks:
        result = compute_troposphere(task)
        results.append(result)
    for date, status, msg in results:
        print(f"Tropospheric Correction for Date: {date}, Status: {status}, Message: {msg}")
