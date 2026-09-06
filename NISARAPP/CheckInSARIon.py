#!/usr/bin/env python3
# Created by Kunyi Chen on 2026-09-02

import numpy as np
from matplotlib import pyplot as plt
import isce, isceobj
import os, json
from multiprocessing import Pool

def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Check the InSAR Ionospheric Phase Screen for NISAR Time Series Processing')
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

def plot(args):
    try:
        InSARRangeLooks = args['InSARRangeLooks']
        InSARAzimuthLooks = args['InSARAzimuthLooks']
        coregistration = args['coregistration']
        pair = args['pair']
        looks = f'{InSARRangeLooks}rlks_{InSARAzimuthLooks}alks'
        FiltIon_fp = os.path.join('pairs_ion',pair,f'filt_{pair}_{looks}.ion')
        OriInt_fp = os.path.join('pairs',pair,f'diff_{pair}_{looks}_{coregistration}.int')

        img = isceobj.createImage()
        img.load(FiltIon_fp + '.xml')
        IonWidth = img.width
        IonLength = img.length
        img = isceobj.createImage()
        img.load(OriInt_fp + '.xml')
        OriIntWidth = img.width
        OriIntLength = img.length
        if IonWidth != OriIntWidth or IonLength != OriIntLength:
            raise ValueError(f"Error: {pair} FiltIon and OriInt have different dimensions")
        FiltIon = np.fromfile(FiltIon_fp, dtype=np.float32).reshape(IonLength, IonWidth)
        OriInt = np.fromfile(OriInt_fp, dtype=np.complex64).reshape(OriIntLength, OriIntWidth)
        mask = np.where((np.abs(OriInt) > 1e-6)&(FiltIon != 0) , 1, 0)
        FiltIon = np.exp(1j*FiltIon*mask)
        avg = np.angle(np.mean(FiltIon))
        FiltIon = FiltIon * np.exp(-1j*avg)
        OriInt = np.exp(1j*mask*np.angle(OriInt))
        avg = np.angle(np.mean(OriInt))
        OriInt = OriInt * np.exp(-1j*avg)

        op_fp = './figures_ion'
        os.makedirs(op_fp, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(3 * 3, IonLength/IonWidth*4))
        # print(4 * 3, IonLength/IonWidth*4)
        fontsize = 10 + IonLength//200
        fig.subplots_adjust(bottom=0.1)
        im1 = axes[0].imshow(np.angle(OriInt), cmap='jet', vmin=-np.pi, vmax=np.pi)
        axes[0].set_title('Ori', fontsize=fontsize)
        axes[0].axis('off')
        im2 = axes[1].imshow(np.angle(FiltIon), cmap='jet', vmin=-np.pi, vmax=np.pi)
        axes[1].set_title('Ion', fontsize=fontsize)
        axes[1].axis('off')
        im3 = axes[2].imshow(np.angle(OriInt*FiltIon.conj()), cmap='jet', vmin=-np.pi, vmax=np.pi)
        axes[2].set_title('rm Ion', fontsize=fontsize)
        axes[2].axis('off')
        cbar1 = fig.colorbar(im1, ax=axes[0], orientation='horizontal', pad=0.05)
        cbar2 = fig.colorbar(im2, ax=axes[1], orientation='horizontal', pad=0.05)
        cbar3 = fig.colorbar(im3, ax=axes[2], orientation='horizontal', pad=0.05)
        fig.savefig(os.path.join(op_fp, f'{pair}_{looks}.png'), dpi=300, bbox_inches='tight',pad_inches=0.05)
        plt.close(fig)

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
    freq = config['Selected Frequency and Polarization']['Freq']
    pol = config['Selected Frequency and Polarization']['Pol']
    IonRangeLooks = config['Multilook Parameters']['Ion Range Looks']
    IonAzimuthLooks = config['Multilook Parameters']['Ion Azimuth Looks']
    InSARRangeLooks = config['Multilook Parameters']['InSAR Range Looks']
    InSARAzimuthLooks = config['Multilook Parameters']['InSAR Azimuth Looks']
    coregistration = config['Coregistration'].lower()

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
            'coregistration': coregistration,
            'pair': pair
        }
        tasks.append(task)
    num_workers = min(n*4, os.cpu_count())

    print(f"Total pairs: {len(tasks)}")
    print(f"Using {num_workers} workers")

    with Pool(processes=num_workers) as pool:
        results = pool.map(plot, tasks)
    for pair, status, msg in results:
        print(f"Check ionospheric phase screen for {pair}: {status} {msg}")


    
