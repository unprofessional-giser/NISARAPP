#!/usr/bin/env python3
# Created by Kunyi Chen on 2026-09-02

import numpy as np
import isce, isceobj
from isceobj.Alos2Proc.Alos2ProcPublic import create_xml
import os, json
from multiprocessing import Pool

def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Estimate the Ionospheric Phase Screen for each date with Least Square Estimation. The first date is set as the reference.')
    parser.add_argument('-config', dest='config', type=str, required=True,
            help = 'config yaml containing the input paths and parameters')
    parser.add_argument('-n', dest='n', type=str, required=False, default=4,
            help = 'number of parallels to run, default is 4')
    parser.add_argument('-excluded_dates', dest='excluded_dates', type=str, required=False, default=[],
            help = 'dates to be excluded from the estimation, separated by comma, e.g. 20200101,20200102')
    parser.add_argument('-excluded_pairs', dest='excluded_pairs', type=str, required=False, default=[],
            help = 'pairs to be excluded from the estimation, separated by comma, e.g. 20200101_20200102,20200102_20200103')

    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        return parser.parse_args()

def LSE(dates:list, used_pairs:list, InSARRangeLooks:int, InSARAzimuthLooks:int, blocks = 100):
    try:
        dates_ion_fp = 'dates_ion'
        os.makedirs(dates_ion_fp, exist_ok=True)
        dates.sort()
        num_dates = len(dates)
        num_pairs = len(used_pairs)
        H = np.zeros((num_pairs, num_dates), dtype=np.float32)
        for i, pair in enumerate(used_pairs):
            date1, date2 = pair.split('_')
            idx1 = dates.index(date1)
            idx2 = dates.index(date2)
            H[i, idx1] = -1
            H[i, idx2] = 1
        H = H[:, 1:]  # remove the first column to set the first date as reference
        temp = np.linalg.pinv(H.T @ H) @ H.T

        img = isceobj.createImage()
        img.load(os.path.join('pairs_ion', used_pairs[0], f'filt_{used_pairs[0]}_{InSARRangeLooks}rlks_{InSARAzimuthLooks}alks.ion.xml'))
        width = img.width
        length = img.length

        for i in range(0, length, blocks):
            start_i = i
            end_i = min(i + blocks, length)
            print(f"Processing lines {start_i} to {end_i}...")
            block_length = end_i - start_i
            InSARIonDataset = np.zeros((num_pairs, block_length, width), dtype=np.float32)
            zero_mask = 1
            for j, pair in enumerate(used_pairs):
                ion_fp = os.path.join('pairs_ion', pair, f'filt_{pair}_{InSARRangeLooks}rlks_{InSARAzimuthLooks}alks.ion')
                InSARIonDataset[j, :, :] = np.memmap(ion_fp, dtype=np.float32, mode='r', shape=(length, width))[start_i:end_i, :]
                zero_mask = zero_mask * (np.abs(InSARIonDataset[j, :, :]) >= 1e-8)
            InSARIonDataset = InSARIonDataset.reshape(num_pairs, -1)  # reshape to (num_pairs, block_length * width)
            SARIonDataset = temp @ InSARIonDataset  # shape: (num_dates-1, block_length * width)
            SARIonDataset = SARIonDataset.reshape(num_dates-1, block_length, width)  # reshape back to (num_dates-1, block_length, width)
            SARIonDataset = SARIonDataset * zero_mask  # apply the zero mask

            for k in range(1, num_dates):
                date = dates[k]
                SARIon_fp = os.path.join(dates_ion_fp, f'{date}_{InSARRangeLooks}rlks_{InSARAzimuthLooks}alks.ion')
                if i == 0:
                    SARIonDataset[k-1, :, :].astype(np.float32).tofile(SARIon_fp)
                else:
                    with open(SARIon_fp, 'ab') as f:
                        SARIonDataset[k-1, :, :].astype(np.float32).tofile(f)

        for k in range(1, num_dates):
            date = dates[k]
            SARIon_fp = os.path.join(dates_ion_fp, f'{date}_{InSARRangeLooks}rlks_{InSARAzimuthLooks}alks.ion')
            create_xml(SARIon_fp, width, length, 'float')

        return (used_pairs, "SUCCESS", "")
    except Exception as e:
        return (used_pairs, "FAILED", str(e))

if __name__ == '__main__':
    inps = cmdLineParse()
    config_fp = inps.config
    n = int(inps.n)
    excluded_dates = inps.excluded_dates.split(',') if inps.excluded_dates else []
    excluded_pairs = inps.excluded_pairs.split(',') if inps.excluded_pairs else []

    import yaml
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
    dates = [ref_date] + sed_dates
    dates.sort()
    output_fp = config['Path of Output']['Output Dict']
    subsequent_num = config['Number of subsequent dates']
    InSARRangeLooks = config['Multilook Parameters']['InSAR Range Looks']
    InSARAzimuthLooks = config['Multilook Parameters']['InSAR Azimuth Looks']

    os.chdir(output_fp)
    all_pairs = []
    used_pairs = []
    for i in range(len(dates)):
        for j in range(i+1, min(i+subsequent_num+1, len(dates))):
            pair = dates[i]+'_'+dates[j]
            all_pairs.append(pair)
            if pair not in excluded_pairs and dates[i] not in excluded_dates and dates[j] not in excluded_dates:
                used_pairs.append(pair)
    for e_pair in excluded_pairs:
        if e_pair not in all_pairs:
            raise ValueError(f"Excluded pair {e_pair} is not in the list of all pairs.")
    print(f"All pairs: {len(all_pairs)}, Used pairs: {len(used_pairs)}")
    print(f"Used pairs: {used_pairs}")

    results = LSE(dates, used_pairs, InSARRangeLooks, InSARAzimuthLooks)
    for used_pairs, status, msg in [results]:
        print(f"Estimate ionospheric phase screen for {len(used_pairs)} pairs: {status} {msg}")
