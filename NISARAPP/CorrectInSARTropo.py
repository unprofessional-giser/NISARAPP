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

    parser = argparse.ArgumentParser(description='Correct the InSAR Tropospheric Phase Delay for NISAR Time Series Processing.')
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

def CorrectInSARTropo(args):
    try:
        InSARRangeLooks = args['InSARRangeLooks']
        InSARAzimuthLooks = args['InSARAzimuthLooks']
        coregistration = args['coregistration']
        doIon = args['doIon']
        pair = args['pair']

        print(f"Correcting InSAR tropospheric phase delay for pair {pair}")
        looks = f'{InSARRangeLooks}rlks_{InSARAzimuthLooks}alks'

        pre_flag = 'filt'
        if not os.path.exists(os.path.join('pairs',pair,f'{pre_flag}_{pair}_{looks}_{coregistration}.int.xml')):
            pre_flag = 'diff'
            if not os.path.exists(os.path.join('pairs',pair,f'{pre_flag}_{pair}_{looks}_{coregistration}.int.xml')):
                raise ValueError(f"Error: {pair} does not have a valid .int file for correction")

        rm_in_extent = 'rm'
        if doIon:
            rm_in_extent +='_ion'
        rm_in_extent = '' if rm_in_extent == 'rm' else rm_in_extent

        rm_ion_fp = os.path.join('pairs',pair,'rm_ion')
        if doIon and not os.path.exists(rm_ion_fp):
            raise ValueError(f"Error: need to finish the ionospheric correction for pair {pair} before correcting the tropospheric phase delay")
        rm_in_name = '' if not doIon else '_rm_ion'
        OriInt_fp = os.path.join('pairs',pair,rm_in_extent,f'{pre_flag}_{pair}_{looks}_{coregistration}{rm_in_name}.int')

        date1, date2 = pair.split('_')
        date1_tro_fp = os.path.join('dates_tro/tropo_delay', f'{date1}_{looks}.tro')
        date2_tro_fp = os.path.join('dates_tro/tropo_delay', f'{date2}_{looks}.tro')
        img = isceobj.createImage()
        img.load(date1_tro_fp + '.xml')
        width = img.width
        length = img.length
        date1_tro = np.fromfile(date1_tro_fp, dtype=np.float32).reshape(length, width)
        date2_tro = np.fromfile(date2_tro_fp, dtype=np.float32).reshape(length, width)
        diff_tro = date2_tro - date1_tro
        ori_int = np.fromfile(OriInt_fp, dtype=np.complex64).reshape(length, width)
        zero_mask = np.abs(ori_int)>0.01
        new_int = zero_mask * np.abs(ori_int) * np.exp(1j*np.angle(ori_int)*zero_mask) * np.exp(-1j * diff_tro)

        if doIon:
            new_int_fd = os.path.join('pairs',pair,'rm_ion_tro')
            os.makedirs(new_int_fd, exist_ok=True)
        else:
            new_int_fd = os.path.join('pairs',pair,'rm_tro')
            os.makedirs(new_int_fd, exist_ok=True)
 
        flag = 'diff' if 'diff' in OriInt_fp else 'filt'
        rm_out_extent = '_rm_ion_tro' if doIon else '_rm_tro'
        new_int_fp = os.path.join(new_int_fd, f'{flag}_{pair}_{looks}_{coregistration}{rm_out_extent}.int')
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
    doIon = config['Error Correction Parameters']['Do Ionospheric Correction']
    doTro = config['Error Correction Parameters']['Do Tropospheric Correction']
    doSET = config['Error Correction Parameters']['Do Soild Earth Tide Correction']
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
            'doIon': doIon,
            'doTro': doTro,
            'doSET': doSET,
        }
        tasks.append(task)
    num_workers = min(n*4, os.cpu_count())
    print(f"Total pairs: {len(tasks)}")
    print(f"Using {num_workers} workers")

    with Pool(processes=num_workers) as pool:
        results = pool.map(CorrectInSARTropo, tasks)
    for pair, status, msg in results:
        print(f"Correct tropospheric phase delay for {pair}: {status} {msg}")
