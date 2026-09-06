#!/usr/bin/env python3
# Created by Kunyi Chen on 2026-09-02

import numpy as np
import isce, isceobj
import os, json
from multiprocessing import Pool
from isceobj.Alos2Proc.Alos2ProcPublic import snaphuUnwrapOriginal

def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Unwrap the InSAR interferogram for NISAR Time Series Processing.')
    parser.add_argument('-config', dest='config', type=str, required=True,
            help = 'config yaml containing the input paths and parameters')
    parser.add_argument('-n', dest='n', type=str, required=False, default=4,
            help = 'number of parallels to run, default is 4')
    parser.add_argument('-sec_fd',dest='sec_fd', type=str, required=False, default=None,
                        help = "the folder containing the wrapped interferograms to be unwrapped. " \
                        "For example: ['./', './rm_ion', './rm_ion_azshift']. If not provided, the script will use the default folder based on the errors correstion processing.")

    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        return parser.parse_args()
    
def unwrap(args):
    try:
        doIon = args['doIon']
        doTro = args['doTro']
        doSET = args['doSET']
        sec_fd = args['sec_fd']
        InSARRangeLooks = args['InSARRangeLooks']
        InSARAzimuthLooks = args['InSARAzimuthLooks']
        coregistration = args['coregistration']
        pair = args['pair']

        print(f"Unwrapping InSAR interferogram for pair {pair}")
        looks = f'{InSARRangeLooks}rlks_{InSARAzimuthLooks}alks'
        rm_in_extent = 'rm'
        if doIon:
            rm_in_extent += '_ion'
        if doTro:
            rm_in_extent += '_tro'
        if doSET:
            rm_in_extent += '_set'
        if sec_fd is None:
            sec_fd = '' if rm_in_extent == 'rm' else rm_in_extent
        if './' in sec_fd:
            sec_fd = sec_fd.replace('./','')

        rm_extent_name = '' if sec_fd == '' else f'_{sec_fd}'
        os.chdir(os.path.join('pairs',pair))
        int_fp = os.path.join(sec_fd, f'filt_{pair}_{looks}_{coregistration}{rm_extent_name}.int')
        flag = 'filt'
        if not os.path.exists(int_fp):
            int_fp = os.path.join(sec_fd, f'diff_{pair}_{looks}_{coregistration}{rm_extent_name}.int')
            flag = 'diff'
            print(int_fp)
            if not os.path.exists(int_fp):
                raise FileNotFoundError(f"Interferogram file not found for pair {pair} in folder {sec_fd}")
        unw_fp = os.path.join(sec_fd, f'{flag}_{pair}_{looks}_{coregistration}{rm_extent_name}.unw')
        cor_fp = os.path.join(f'{pair}_{looks}_{coregistration}.cor')
        amp_fp = os.path.join(f'{pair}_{looks}_{coregistration}.amp')
        
        try:
            snaphuUnwrapOriginal(int_fp, cor_fp, amp_fp, unw_fp)
            os.system(f'rm -rf ./snaphu.conf')
            os.system(f'rm -rf ./isce.log')
            os.chdir('../..')
            return (pair, "Success", "")
        except Exception as e:
            os.system(f'rm -rf ./snaphu.conf')
            os.system(f'rm -rf ./isce.log')
            os.chdir('../..')
            return (pair, "Failed", str(e))
    except Exception as e:
        os.system(f'rm -rf ./snaphu.conf')
        os.system(f'rm -rf ./isce.log')
        os.chdir('../..')
        return (pair, "Failed", str(e))

if __name__ == '__main__':
    inps = cmdLineParse()
    config_fp = inps.config
    n = int(inps.n)
    sec_fd = inps.sec_fd

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
            'sec_fd': sec_fd
        }
        tasks.append(task)
        # break
    num_workers = min(n*4, os.cpu_count())
    print(f"Total pairs: {len(tasks)}")
    print(f"Using {num_workers} workers")

    with Pool(processes=num_workers) as pool:
        results = pool.map(unwrap, tasks)
    os.system('rm -rf ./snaphu.conf ./isce.log')
    for pair, status, msg in results:
        print(f"Unwrap for {pair}: {status} {msg}")
