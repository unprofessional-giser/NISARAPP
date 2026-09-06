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

    parser = argparse.ArgumentParser(description='Correct the InSAR Soild Earth Tide Error for NISAR Time Series Processing.')
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

def CorrectInSARSET(args):
    try:
        InSARRangeLooks = args['InSARRangeLooks']
        InSARAzimuthLooks = args['InSARAzimuthLooks']
        coregistration = args['coregistration']
        doIon = args['doIon']
        doTro = args['doTro']
        pair = args['pair']

        print(f"Correcting InSAR Solid Earth Tide for pair {pair}")
        looks = f'{InSARRangeLooks}rlks_{InSARAzimuthLooks}alks'

        flag = 'filt'
        if not os.path.exists(os.path.join('pairs',pair,f'{flag}_{pair}_{looks}_{coregistration}.int.xml')):
            flag = 'diff'
            if not os.path.exists(os.path.join('pairs',pair,f'{flag}_{pair}_{looks}_{coregistration}.int.xml')):
                raise ValueError(f"Error: {pair} does not have a valid .int file for correction")


        rm_in_extent = 'rm'
        if doIon:
            rm_in_extent += '_ion'
        if doTro:
            rm_in_extent += '_tro'
        sec_fd = '' if rm_in_extent == 'rm' else f'{rm_in_extent}'
        rm_extent_name = '' if rm_in_extent == 'rm' else f'_{rm_in_extent}'
        OriInt_fp = os.path.join('pairs',pair,sec_fd,f'{flag}_{pair}_{looks}_{coregistration}{rm_extent_name}.int')

        date1, date2 = pair.split('_')
        date1_set_fp = os.path.join('dates_set', f'{date1}_{looks}.set')
        date2_set_fp = os.path.join('dates_set', f'{date2}_{looks}.set')
        img = isceobj.createImage()
        img.load(date1_set_fp + '.xml')
        width = img.width
        length = img.length
        date1_set = np.fromfile(date1_set_fp, dtype=np.float32).reshape(length, width)
        date2_set = np.fromfile(date2_set_fp, dtype=np.float32).reshape(length, width)
        diff_set = date2_set - date1_set
        ori_int = np.fromfile(OriInt_fp, dtype=np.complex64).reshape(length, width)
        zero_mask = np.abs(ori_int)>0.01
        new_int = zero_mask * np.abs(ori_int) * np.exp(1j*np.angle(ori_int)*zero_mask) * np.exp(1j * diff_set)

        rm_out_extent = os.path.join("pairs",pair,rm_in_extent + '_set')
        os.makedirs(rm_out_extent, exist_ok=True)
        if rm_in_extent == '':
            rm_extent_name = '_rm_set'
        else:
            rm_extent_name = '_' + rm_in_extent + '_set'
        new_int_fp = os.path.join(rm_out_extent, f'{flag}_{pair}_{looks}_{coregistration}{rm_extent_name}.int')
        new_int.astype(np.complex64).tofile(new_int_fp)
        create_xml(new_int_fp, width, length, 'int')

        return (pair,"SUCCESS","")
    except Exception as e:
        return (pair,"FAILURE",str(e))

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
        results = pool.map(CorrectInSARSET, tasks)
    for pair, status, msg in results:
        print(f"Correct Solid Earth Tide for {pair}: {status} {msg}")

