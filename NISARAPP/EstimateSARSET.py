#!/usr/bin/env python3
# Created by Kunyi Chen on 2026-09-02

import pysolid
import h5py
from datetime import datetime, timedelta
import numpy as np
import os
from scipy.interpolate import interpn
import isce,isceobj
from isceobj.Alos2Proc.Alos2ProcPublic import create_xml
from multiprocessing import Pool

def interpolate_to_rdr(data, ml_lat, ml_lon, meta):
    """
    data: 2D array [LENGTH, WIDTH]
    ml_lat, ml_lon: 2D arrays same shape as output
    """
    y = meta['Y_FIRST'] + meta['Y_STEP'] * np.arange(meta['LENGTH'])
    x = meta['X_FIRST'] + meta['X_STEP'] * np.arange(meta['WIDTH'])
    points = (y, x)
    xi = np.stack([ml_lat.ravel(), ml_lon.ravel()], axis=1)
    data_interp = interpn(
        points,
        data,
        xi,
        method='linear',
        bounds_error=False,
        fill_value=np.nan
    )
    return data_interp.reshape(ml_lat.shape)

def get_time(rslc_fp: str):

    with h5py.File(rslc_fp, 'r') as h5_obj:

        zeroDopplerEndTime = h5_obj['/science/LSAR/identification/zeroDopplerEndTime'][()].decode('utf-8')[:26]
        zeroDopplerEndTime = datetime.strptime(zeroDopplerEndTime, '%Y-%m-%dT%H:%M:%S.%f')
        zeroDopplerStartTime = h5_obj['/science/LSAR/identification/zeroDopplerStartTime'][()].decode('utf-8')[:26]
        zeroDopplerStartTime = datetime.strptime(zeroDopplerStartTime, '%Y-%m-%dT%H:%M:%S.%f')
        e_t = zeroDopplerEndTime.timestamp()
        s_t = zeroDopplerStartTime.timestamp()
        zeroDopplerMidTime = datetime.fromtimestamp((e_t+s_t)/2)
        yr = zeroDopplerMidTime.year
        mo = zeroDopplerMidTime.month
        dy = zeroDopplerMidTime.day
        hr = zeroDopplerMidTime.hour
        mn = zeroDopplerMidTime.minute
        sc = zeroDopplerMidTime.second

        time = datetime(yr, mo, dy, hr, mn, sc)

        h5_obj.close()

    return time

def compute_solid_earth_tide(args):
    # Placeholder function for computing solid earth tide correction
    # You can implement the actual computation logic here using pysolid or any other library
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
        saza_fp = args['saza']
        InSARRangeLooks = args['InSARRangeLooks']
        InSARAzimuthLooks = args['InSARAzimuthLooks']
        op_dir = args['op_dir']

        set_fd = os.path.join(op_dir, 'dates_set')
        os.makedirs(set_fd, exist_ok=True)
        looks = f"{InSARRangeLooks}rlks_{InSARAzimuthLooks}alks"

        img = isceobj.createImage()
        img.load(slat_fp + '.xml')
        width = img.width
        length = img.length
        ml_lat = np.fromfile(slat_fp, dtype=np.float32).reshape(length, width)
        ml_lon = np.fromfile(slon_fp, dtype=np.float32).reshape(length, width)
        ml_hgt = np.fromfile(shgt_fp, dtype=np.float32).reshape(length, width)
        inc = np.fromfile(sinc_fp, dtype=np.float32).reshape(length, width)
        aza = np.fromfile(saza_fp, dtype=np.float32).reshape(length, width)
        time = get_time(rslc_fp)
        date = '%4d%02d%02d'%(time.year, time.month, time.day)

        snwe = [np.min(ml_lat), np.max(ml_lat), np.min(ml_lon), np.max(ml_lon)]
        edge = .5
        delta = 0.1
        s, n, w, e = snwe

        meta = {
            'LENGTH' : int((n - s + 2.0 * edge) / delta + 1),    # number of rows
            'WIDTH'  : int((e - w + 2.0 * edge) / delta + 1),    # number of columns
            'X_FIRST': w - edge,                                 # min longitude in degree (upper left corner of the upper left pixel)
            'Y_FIRST': n + edge,                                 # max latitude   in degree (upper left corner of the upper left pixel)
            'X_STEP' : delta,                                    # output resolution in degree
            'Y_STEP' : -delta,                                   # output resolution in degree
        }
    
        # compute SET via pysolid
        tide_e, tide_n, tide_u = pysolid.calc_solid_earth_tides_grid(
            time, meta,
            display=False,
            verbose=True,
        )

        tide_rdr = []

        tide_rdr = [interpolate_to_rdr(d, ml_lat, ml_lon, meta)
                    for d in [tide_e, tide_n, tide_u]]
        tide_rdr_e = tide_rdr[0]
        tide_rdr_n = tide_rdr[1]
        tide_rdr_u = tide_rdr[2]
        #convert to LOS
        tide_los = tide_rdr_e * np.sin(np.deg2rad(inc))*np.sin(np.deg2rad(aza)) +\
                   tide_rdr_n * np.sin(np.deg2rad(inc))*np.cos(np.deg2rad(aza)) -\
                   tide_rdr_u * np.cos(np.deg2rad(inc))
        tide_los *= -4.0 * np.pi / wavelength
        tide_los = tide_los.astype(np.float32)
        set_fp = os.path.join(set_fd, f'{date}_{looks}.set')
        tide_los.tofile(set_fp)
        create_xml(set_fp, width, length, 'float')

        return (date,"SUCCESS","")
    except Exception as e:
        return (date,"FAILURE",str(e))


def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Estimate the Solid Earth Tide for NISAR Time Series Processing with pysolid.')
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
    aza_fp = os.path.join('dates', ref_date, f'aza_{InSARRangeLooks}rlks_{InSARAzimuthLooks}alks.flt')
    para_json_fp = os.path.join('dates', ref_date, 'slc_parameters.json')
    paras = json.load(open(para_json_fp, 'r', encoding='utf-8'))
    wavelength = paras['wavelength']

    tasks = []
    for rslc_fp in rslc_fps:
        args = {
            'wavelength': wavelength,
            'rslc_h5': rslc_fp,
            'slat': lat_fp,
            'slon': lon_fp,
            'shgt': dem_fp,
            'sinc': inc_fp,
            'saza': aza_fp,
            'InSARRangeLooks': InSARRangeLooks,
            'InSARAzimuthLooks': InSARAzimuthLooks,
            'op_dir': output_fp
        }
        tasks.append(args)

    num_workers = min(n*4, os.cpu_count())

    print(f"Total pairs: {len(tasks)}")
    print(f"Using {num_workers} workers")

    with Pool(processes=num_workers) as pool:
        results = pool.map(compute_solid_earth_tide, tasks)
    for date, status, msg in results:
        print(f"Compute the solid earth tide for {date}: {status} {msg}")