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

    parser = argparse.ArgumentParser(description='Correct the InSAR Ionospheric Phase Screen for NISAR Time Series Processing.')
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

def CorrectInSARIon(args):
    try:
        InSARRangeLooks = args['InSARRangeLooks']
        InSARAzimuthLooks = args['InSARAzimuthLooks']
        coregistration = args['coregistration']
        pair = args['pair']
        IonRefDate = args['IonRefDate']

        print(f"Correcting InSAR ionospheric phase screen for pair {pair} with Ionospheric reference date {IonRefDate}...")
        looks = f'{InSARRangeLooks}rlks_{InSARAzimuthLooks}alks'
        OriInt_fp = os.path.join('pairs',pair,f'filt_{pair}_{looks}_{coregistration}.int')
        if not os.path.exists(OriInt_fp + '.xml'):
            OriInt_fp = os.path.join('pairs',pair,f'diff_{pair}_{looks}_{coregistration}.int')
            if not os.path.exists(OriInt_fp + '.xml'):
                raise ValueError(f"Error: {pair} does not have a valid .int file for correction")

        date1, date2 = pair.split('_')
        date1_ion = 0
        date2_ion = 0
        if date1 != IonRefDate:
            date1_ion_fp = os.path.join('dates_ion', f'{date1}_{InSARRangeLooks}rlks_{InSARAzimuthLooks}alks.ion')
            img = isceobj.createImage()
            img.load(date1_ion_fp + '.xml')
            width = img.width
            length = img.length
            date1_ion = np.fromfile(date1_ion_fp, dtype=np.float32).reshape(length, width)
        if date2 != IonRefDate:
            date2_ion_fp = os.path.join('dates_ion', f'{date2}_{InSARRangeLooks}rlks_{InSARAzimuthLooks}alks.ion')
            img = isceobj.createImage()
            img.load(date2_ion_fp + '.xml')
            width = img.width
            length = img.length
            date2_ion = np.fromfile(date2_ion_fp, dtype=np.float32).reshape(length, width)
        diff_ion = date2_ion - date1_ion
        zero_mask = diff_ion !=0
        ori_int = np.fromfile(OriInt_fp, dtype=np.complex64).reshape(length, width)
        boundary_mask = np.zeros((length, width), dtype=np.int8)
        factor = 0.057
        boundary_mask[:,int(width*factor):width-int(width*factor)] = True
        ori_int = ori_int * boundary_mask
        diff_ion = diff_ion * boundary_mask
        new_int = zero_mask * np.abs(ori_int) * np.exp(1j*np.angle(ori_int)*zero_mask) * np.exp(-1j * diff_ion)
        new_int_fd_fp = os.path.join('pairs',pair,'rm_ion')
        os.makedirs(new_int_fd_fp, exist_ok=True)
        flag = 'diff' if 'diff' in OriInt_fp else 'filt'
        new_int_fp = os.path.join(new_int_fd_fp, f'{flag}_{pair}_{looks}_{coregistration}_rm_ion.int')
        new_int.astype(np.complex64).tofile(new_int_fp)
        create_xml(new_int_fp, width, length, 'int')

        return (pair, "SUCCESS", "")
    except Exception as e:
        return (pair, "FAILED", str(e))

if __name__ == '__main__':
    inps = cmdLineParse()
    config_fp = inps.config
    n = int(inps.n)

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
    pairs = []
    for i in range(len(dates)):
        for j in range(i+1, min(i+subsequent_num+1, len(dates))):
            pairs.append(dates[i]+'_'+dates[j])

    tasks = []
    for pair in pairs:
        task = {
            'InSARRangeLooks': InSARRangeLooks,
            'InSARAzimuthLooks': InSARAzimuthLooks,
            'coregistration': config['Coregistration'].lower(),
            'pair': pair,
            'IonRefDate': ref_date
        }
        tasks.append(task)
    num_workers = min(n*4, os.cpu_count())
    print(f"Total pairs: {len(tasks)}")
    print(f"Using {num_workers} workers")

    with Pool(processes=num_workers) as pool:
        results = pool.map(CorrectInSARIon, tasks)
    for pair, status, msg in results:
        print(f"Correct ionospheric phase screen for {pair}: {status} {msg}")
