#!/usr/bin/env python3
# Created by Kunyi Chen on 2026-08-29
import os, json

def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Small Baseline Subset (SBAS) InSAR processing')
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

def build_plist(pairs, op_fp):
    with open(os.path.join(op_fp, 'plist.txt'), 'w', encoding='utf-8') as f:
        for pair in pairs:
            if '_' in pair:
                mdate, sdate = pair.split('_')
            elif '-' in pair:
                mdate, sdate = pair.split('-')
            if len(mdate) == 8:
                mdate = mdate[2:]
            if len(sdate) == 8:
                sdate = sdate[2:]
            pair = mdate + '-' + sdate
            f.write(pair + '\n')
    return os.path.abspath(os.path.join(op_fp, 'plist.txt'))

if __name__ == '__main__':
    inps = cmdLineParse()
    config_fp = inps.config
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
    App_fp = config['Path of NISAR Workflow Software']['NISARAPP']
    InSARRangeLooks = config['Multilook Parameters']['InSAR Range Looks']
    InSARAzimuthLooks = config['Multilook Parameters']['InSAR Azimuth Looks']
    coregistration = config['Coregistration'].lower()
    do_ion = config['Error Correction Parameters']['Do Ionospheric Correction']
    do_tropo = config['Error Correction Parameters']['Do Tropospheric Correction']
    do_set = config['Error Correction Parameters']['Do Soild Earth Tide Correction']
    isce2_path = config['Path of Python Interpreter']['isce2']
    pairs = []
    for i in range(len(dates)):
        for j in range(i+1, min(i+subsequent_num+1, len(dates))):
            pairs.append(dates[i]+'_'+dates[j])

    os.chdir(output_fp)
    para_json_fp = os.path.join('dates', ref_date, 'slc_parameters.json')
    paras = json.load(open(para_json_fp, 'r', encoding='utf-8'))
    wavelength = paras['wavelength']
    rm_in_extent = 'rm'
    if do_ion:
        rm_in_extent +='_ion'
    if do_tropo:
        rm_in_extent +='_tro'
    if do_set:
        rm_in_extent +='_set'
    rm_in_extent = '' if rm_in_extent == 'rm' else rm_in_extent

    looks = f'{InSARRangeLooks}rlks_{InSARAzimuthLooks}alks'
    flag = 'filt'
    if not os.path.exists(f'pairs/{pairs[0]}/{flag}_{pairs[0]}_{looks}_{coregistration}.unw'):
        flag = 'diff'
        if not os.path.exists(f'pairs/{pairs[0]}/{flag}_{pairs[0]}_{looks}_{coregistration}.unw'):
            raise Exception(f'Cannot find {flag}_{pairs[0]}_{looks}_{coregistration}.unw in {output_fp}/pairs/{pairs[0]}/')

    rm_in_name = '' if rm_in_extent == '' else f'_{rm_in_extent}'
    used_unw_fp = f'{output_fp}/pairs/*_*/{rm_in_extent}/{flag}_*_*_{looks}_{coregistration}{rm_in_name}.unw'
    output_fp = f'{output_fp}/ts/velocity.unw'
    os.makedirs('./ts', exist_ok=True)
    build_plist(pairs, './ts')

    sbas_cmd_fp = os.path.join('./ts', f'cmd_sbas.sh')
    with open(sbas_cmd_fp, 'w', encoding='utf-8') as f:
        f.write(f'#!/bin/bash\n')
        f.write(f'cd {os.path.abspath("./ts")}\n')
        f.write(f'{isce2_path} {os.path.join(App_fp, "insar_ts_lse.py")}  -plist {os.path.abspath("./ts/plist.txt")}  -unw "{used_unw_fp}"  -output {output_fp} -wvl {wavelength}\n')
    f.close()
    os.system(f'chmod +x {sbas_cmd_fp}')
    print(f'Created SBAS InSAR processing command file: {os.path.abspath(sbas_cmd_fp)}')
    print(f'Run the following command to perform SBAS InSAR processing.')
