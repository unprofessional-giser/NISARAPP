#!/usr/bin/env python3
# Created by Kunyi Chen on 2026-09-02

import numpy as np
import isce, isceobj
from isceobj.Alos2Proc.Alos2ProcPublic import create_xml
from isceobj.Alos2Proc.runIonFilt import computeIonosphere, adaptive_gaussian, gaussian, polyfit_2d
from contrib.alos2proc_f.alos2proc_f import rect
from isceobj.Alos2Proc.Alos2ProcPublic import filterStdPolyIon
import scipy.signal as ss
from scipy.interpolate import interp1d
from GenerateInSARCoh import process_pair
import h5py
from matplotlib import pyplot as plt
import os, json
from multiprocessing import Pool

def get_hdf_metadata(filepath:str):
    if filepath.endswith(('.rdr', '.slc')):
        f_hdf = filepath[:-4] + '.hdr'
    else:
        f_hdf = filepath + '.hdr'
    with open(f_hdf, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'samples' in line:
                width = int(line.split('=')[1].strip())
            if 'lines' in line:
                length = int(line.split('=')[1].strip())
            if 'data type' in line:
                dtype = int(line.split('=')[1].strip())
                if dtype == 4:
                    dtype = np.float32
                elif dtype == 6:
                    dtype = np.complex64
                elif dtype == 5:
                    dtype = np.float64
    return width, length, dtype

def ion_std(fl, fu, numberOfLooks, cor):
        '''
        compute standard deviation of ionospheric phase
        fl:  lower band center frequency
        fu:  upper band center frequency
        cor: coherence, must be numpy array
        '''
        f0 = (fl + fu) / 2.0
        interferogramVar = (1.0 - cor**2) / (2.0 * numberOfLooks * cor**2 + (cor==0))
        std = fl*fu/f0/(fu**2-fl**2)*np.sqrt(fu**2*interferogramVar+fl**2*interferogramVar)
        std[np.nonzero(cor==0)] = 0
        return std

def EstimateFiltIon(args):
    pair = args['pair']
    pairs_fp = args['pair_fp']
    IonRangeLooks = args['IonRangeLooks']
    IonAzimuthLooks = args['IonAzimuthLooks']
    InSARRangeLooks = args['InSARRangeLooks']
    InSARAzimuthLooks = args['InSARAzimuthLooks']
    high_wavelength = args['high_subband_wavelength']
    low_wavelength = args['low_subband_wavelength']
    dem_fp = args['dem_fp']
    subbandNumberOfLooks = args['subbandNumberOfLooks']
    rg_bd = args['RangeBandWidth']
    IonStdOut = args['IonStdOut'] if 'IonStdOut' in args else 0.1
    speed_of_light = 299792458.0
    fl = speed_of_light / low_wavelength
    fu = speed_of_light / high_wavelength

    try:
        # 1. unwrapped interferogram
        print(f"[START] {pair}")
        from isceobj.Alos2Proc.Alos2ProcPublic import snaphuUnwrapOriginal
        highband_ifgm_fp = os.path.abspath(os.path.join(pairs_fp, f'high_diff_{pair}_{IonRangeLooks}rlks_{IonAzimuthLooks}alks.int'))
        highband_cor_fp = os.path.abspath(os.path.join(pairs_fp, f'high_{pair}_{IonRangeLooks}rlks_{IonAzimuthLooks}alks.cor'))
        highband_amp_fp = os.path.abspath(os.path.join(pairs_fp, f'high_{pair}_{IonRangeLooks}rlks_{IonAzimuthLooks}alks.amp'))
        highband_unw_fp = os.path.abspath(os.path.join(pairs_fp, f'high_diff_{pair}_{IonRangeLooks}rlks_{IonAzimuthLooks}alks.unw'))
        lowband_ifgm_fp = os.path.abspath(os.path.join(pairs_fp, f'low_diff_{pair}_{IonRangeLooks}rlks_{IonAzimuthLooks}alks.int'))
        lowband_cor_fp = os.path.abspath(os.path.join(pairs_fp, f'low_{pair}_{IonRangeLooks}rlks_{IonAzimuthLooks}alks.cor'))
        lowband_amp_fp = os.path.abspath(os.path.join(pairs_fp, f'low_{pair}_{IonRangeLooks}rlks_{IonAzimuthLooks}alks.amp'))
        lowband_unw_fp = os.path.abspath(os.path.join(pairs_fp, f'low_diff_{pair}_{IonRangeLooks}rlks_{IonAzimuthLooks}alks.unw'))
        try:
            os.chdir(pairs_fp)
            snaphuUnwrapOriginal(highband_ifgm_fp, highband_cor_fp, highband_amp_fp, highband_unw_fp, snaphuConfFile = './highband_snaphu.conf')
            os.system(f'rm -rf ./highband_snaphu.conf')
            os.chdir('../..')
        except Exception as e:
            print(f"{pair}: {e}")
        try:
            os.chdir(pairs_fp)
            snaphuUnwrapOriginal(lowband_ifgm_fp, lowband_cor_fp, lowband_amp_fp, lowband_unw_fp, snaphuConfFile = './lowband_snaphu.conf')
            os.system(f'rm -rf ./lowband_snaphu.conf')
            os.chdir('../..')
        except Exception as e:
            print(f"{pair}: {e}")

        print(f"Step 1 Finished: Unwrapped interferograms for pair {pair} generated.")

        # 2. Estimate ionospheric phase screen
        img = isceobj.createImage()
        img.load(highband_unw_fp + '.xml')
        width = img.width
        length = img.length
        Ion_width = width
        Ion_length = length
        highband_unw = np.fromfile(highband_unw_fp, dtype=np.float32).reshape(length,2, width)
        lowband_unw = np.fromfile(lowband_unw_fp, dtype=np.float32).reshape(length,2, width)
        highband_amp_mask = np.abs(highband_unw[:,0,:]) > 1e-3
        lowband_amp_mask = np.abs(lowband_unw[:,0,:]) > 1e-3
        highband_unw_phase = highband_unw[:,1,:]
        lowband_unw_phase = lowband_unw[:,1,:]
        highband_cor = np.fromfile(highband_cor_fp, dtype=np.float32).reshape(length,2, width)
        highband_cor = highband_cor[:,1,:]
        lowband_cor = np.fromfile(lowband_cor_fp, dtype=np.float32).reshape(length,2, width)
        lowband_cor = lowband_cor[:,1,:]
        cor = (highband_cor + lowband_cor) / 2.0
        cor[cor<0.0] = 0.0
        cor[cor>1.0] = 1.0
        wgt = cor ** 20 * highband_amp_mask * lowband_amp_mask
        ion = computeIonosphere(lowband_unw_phase, highband_unw_phase, wgt, fl, fu, 0, 0)
        ion = ion * highband_amp_mask * lowband_amp_mask
        ion_fp = os.path.join(pairs_fp, f'{pair}_{IonRangeLooks}rlks_{IonAzimuthLooks}alks.ion')
        ion.astype(np.float32).tofile(ion_fp)
        create_xml(ion_fp, width, length, 'float')
        print(f"Step 2 Finished: Ionospheric phase screen for pair {pair} generated.")
        
        # 3. Filter ionospheric phase screen
        dem = None
        if InSARRangeLooks == IonRangeLooks and InSARAzimuthLooks == IonAzimuthLooks:
            img = isceobj.createImage()
            img.load(dem_fp + '.xml')
            dem_width = img.width
            dem_length = img.length
            dem = np.fromfile(dem_fp, dtype=np.float32).reshape(dem_length, dem_width)
        else:
            dem_fp = os.path.abspath(dem_fp)
            res_dem_fp = dem_fp.replace(f'{InSARRangeLooks}rlks_{InSARAzimuthLooks}alks', f'{IonRangeLooks}rlks_{IonAzimuthLooks}alks')
            if os.path.exists(res_dem_fp):
                img = isceobj.createImage()
                img.load(res_dem_fp + '.xml')
                dem_width = img.width
                dem_length = img.length
                dem = np.fromfile(res_dem_fp, dtype=np.float32).reshape(dem_length, dem_width)
            else:
                img = isceobj.createImage()
                img.load(dem_fp + '.xml')
                dem_width = img.width
                dem_length = img.length
                dem = np.fromfile(dem_fp, dtype=np.float32).reshape(dem_length, dem_width)
                # resample dem
                nrli = InSARRangeLooks
                nali = InSARAzimuthLooks
                nrlo = IonRangeLooks
                nalo = IonAzimuthLooks
                dem_index = np.linspace(0, dem_width-1, num=dem_width, endpoint=True)
                res_dem_index = np.linspace(0, Ion_width-1, num=Ion_width, endpoint=True) * nrlo/nrli + (nrlo-nrli)/(2.0*nrli)
                res_dem1 = np.zeros((max(Ion_length, dem_length), Ion_width), dtype=np.float32)
                res_dem = np.zeros((Ion_length, Ion_width), dtype=np.float32)
                # print(dem.shape, res_dem1.shape)
                for i in range(dem_length):
                    # print(f"Resampling DEM: {i+1}/{dem_length}")
                    f = interp1d(dem_index, dem[i,:], kind='cubic', fill_value="extrapolate")
                    res_dem1[i, :] = f(res_dem_index)
                dem_index = np.linspace(0, dem_length-1, num=dem_length, endpoint=True)
                res_dem_index = np.linspace(0, Ion_length-1, num=Ion_length, endpoint=True) * nalo/nali + (nalo-nali)/(2.0*nali)
                for j in range(Ion_width):
                    # print(f"Resampling DEM: {j+1}/{Ion_width}")
                    f = interp1d(dem_index, res_dem1[0:dem_length, j], kind='cubic', fill_value="extrapolate")
                    res_dem[:, j] = f(res_dem_index)
                res_dem.astype(np.float32).tofile(res_dem_fp)
                dem = res_dem
                create_xml(res_dem_fp, Ion_width, Ion_length, 'float')
        mask = dem > 0
        # highband_cor = np.fromfile(highband_cor_fp, dtype=np.float32).reshape(length,2, width)
        # lowband_cor = np.fromfile(lowband_cor_fp, dtype=np.float32).reshape(length,2, width)
        # highband_cor = highband_cor[:,1,:]
        # lowband_cor = lowband_cor[:,1,:]
        # cor = (highband_cor + lowband_cor) / 2.0
        index = np.nonzero(np.logical_or(lowband_cor==0, highband_cor==0))
        cor[index] = 0
        del highband_cor, lowband_cor, index
        cor = cor * mask
        cor[np.isnan(cor)] = 0.0
        cor[np.nonzero(cor<0)] = 0.0
        cor[np.nonzero(cor>1)] = 0.0
        cor[np.nonzero(cor<0.05)] = 0.05
        cor[np.nonzero(cor>0.95)] = 0.95
        print('cor min, max, mean, std: {}, {}, {}, {}'.format(np.min(cor), np.max(cor), np.mean(cor), np.std(cor)))

        std_out0 = IonStdOut
        std_out0_test = np.polyval(filterStdPolyIon, rg_bd/(1e6))
        print('std_out0 = {}'.format(std_out0_test))
        std = ion_std(fl, fu, subbandNumberOfLooks, cor)
        cor2 = np.linspace(0.1, 0.9, num=9, endpoint=True)
        std2 = ion_std(fl, fu, subbandNumberOfLooks, cor2)
        std_out2 = np.zeros(cor2.size)
        win2 = np.zeros(cor2.size, dtype=np.int32)
        for i in range(cor2.size):
            for size in range(9, 10001, 2):
                #this window must be the same as those used in adaptive_gaussian!!!
                gw = gaussian(size, size/2.0, scale=1.0)
                scale = 1.0 / np.sum(gw / std2[i]**2)
                std_out2[i] = scale * np.sqrt(np.sum(gw**2 / std2[i]**2))
                win2[i] = size
                if std_out2[i] <= std_out0:
                    break
        print('if ionospheric phase standard deviation <= {} rad, minimum filtering window size required:'.format(std_out0))
        print('coherence   window size')
        print('************************')
        for x, y in zip(cor2, win2):
            print('  %5.2f       %5d'%(x, y))

        ion = ion * mask
        dofilt = True
        size_secondary = 5
        size_min = win2[-1]
        size_max = win2[0]
        if dofilt:
            corThresholdFit = 0.25
            fit = False
            filt = True
            filtSecondary = True
            fitAdaptive = True
            fitAdaptiveOrder = 2
            rmOutliers = True
            if fit:
                #prepare weight
                wgt = std**2
                wl,ww = wgt.shape
                print("Mean weight:", np.mean(wgt))
                wgt[np.nonzero(cor<corThresholdFit)] = 0
                index = np.nonzero(wgt!=0)
                wgt[index] = 1.0/(wgt[index])
                #fit
                ion_fit, coeff = polyfit_2d(ion, wgt, 2)
                ion -= ion_fit * (ion!=0)
            #filter the rest of the ionosphere
            if filt:
                (ion_filt, std_out, window_size_out) = adaptive_gaussian(ion, std, size_min, size_max, std_out0, fit=fitAdaptive, order=fitAdaptiveOrder, rm_outliers=rmOutliers)
                if filtSecondary:
                    print('applying secondary filtering with window size {}'.format(size_secondary))
                    g2d = gaussian(size_secondary, size_secondary/2.0, scale=1.0)
                    scale = ss.fftconvolve((ion_filt!=0), g2d, mode='same')
                    ion_filt = (ion_filt!=0) * ss.fftconvolve(ion_filt, g2d, mode='same') / (scale + (scale==0))

            #get final results
            if (fit == True) and (filt == True):
                ion_final = ion_filt + ion_fit * (ion_filt!=0)
            elif (fit == True) and (filt == False):
                ion_final = ion_fit
            elif (fit == False) and (filt == True):
                ion_final = ion_filt
            else:
                ion_final = ion

            ionfiltfile = os.path.join(pairs_fp, f'filt_{pair}_{IonRangeLooks}rlks_{IonAzimuthLooks}alks.ion')
            stdfiltfile = os.path.join(pairs_fp, f'filt_{pair}_{IonRangeLooks}rlks_{IonAzimuthLooks}alks.std')
            windowsizefiltfile = os.path.join(pairs_fp, f'filt_{pair}_{IonRangeLooks}rlks_{IonAzimuthLooks}alks.ws')
            #output results
            ion_final = ion_final * mask * highband_amp_mask * lowband_amp_mask
            new_mask = np.zeros((length, width), dtype=np.int8)
            factor = 0.057
            new_mask[:,int(width*factor):width-int(width*factor)] = True
            ion_final = ion_final * new_mask
            ion_final.astype(np.float32).tofile(ionfiltfile)
            create_xml(ionfiltfile, width, length, 'float')
            if filt == True:
                std_out.astype(np.float32).tofile(stdfiltfile)
                create_xml(stdfiltfile, width, length, 'float')
                window_size_out.astype(np.float32).tofile(windowsizefiltfile)
                create_xml(windowsizefiltfile, width, length, 'float')
            print(f"Step 3 Finished: Filtered ionospheric phase screen for pair {pair} generated.")

        # 4. resample the filtered ionospheric phase screen to match the InSAR looks
        if (InSARRangeLooks != IonRangeLooks) or (InSARAzimuthLooks != IonAzimuthLooks):
            img = isceobj.createImage()
            img.load(dem_fp + '.xml')
            dem_width = img.width
            dem_length = img.length
            ml2 = '{}rlks_{}alks'.format(IonRangeLooks, IonAzimuthLooks)
            ml3 = '{}rlks_{}alks'.format(InSARRangeLooks, InSARAzimuthLooks)
            ionfiltfile = os.path.join(pairs_fp, f'filt_{pair}_{ml2}.ion')
            ionrectfile = os.path.join(pairs_fp, f'filt_{pair}_{ml3}.ion')
            img = isceobj.createImage()
            img.load(ionfiltfile + '.xml')
            nlro = InSARRangeLooks
            nalo = InSARAzimuthLooks
            nlri = IonRangeLooks
            nali = IonAzimuthLooks
            Ion_index = np.linspace(0, Ion_width-1, num=Ion_width, endpoint=True)
            InSAR_index = np.linspace(0, dem_width-1, num=dem_width, endpoint=True) * nlro/nlri + (nlro-nlri)/(2.0*nlri)
            ion_rect = np.zeros((dem_length, dem_width), dtype=np.float32)
            for i in range(Ion_length):
                f = interp1d(Ion_index, ion_final[i,:], kind='cubic', fill_value="extrapolate")
                ion_rect[i, :] = f(InSAR_index)
            Ion_index = np.linspace(0, Ion_length-1, num=Ion_length, endpoint=True)
            InSAR_index = np.linspace(0, dem_length-1, num=dem_length, endpoint=True) * nalo/nali + (nalo-nali)/(2.0*nali)
            for j in range(dem_width):
                f = interp1d(Ion_index, ion_rect[0:Ion_length, j], kind='cubic', fill_value="extrapolate")
                ion_rect[:, j] = f(InSAR_index)
            ion_rect.astype(np.float32).tofile(ionrectfile)
            create_xml(ionrectfile, dem_width, dem_length, 'float')
            
            print(f"Step 4 Finished: Resampled filtered ionospheric phase screen for pair {pair} generated.")

        print(f"[DONE]  {pair}")
        return (pair, "SUCCESS", "")
    except Exception as e:
        print(f"[FAIL]  {pair} -> {e}")
        return (pair, "FAILED", str(e))

def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Estimate and Filter InSAR Ionospheric Phase Screen for NISAR Time Series Processing')
    parser.add_argument('-config', dest='config', type=str, required=True,
            help = 'config yaml containing the input paths and parameters')
    parser.add_argument('-IonStdOut', dest='IonStdOut', type=float, required=False, default=0.0,
            help = 'the standard deviation of the ionospheric phase screen after filtering.\n\
                Example: 0.05 rad for 40 range looks and 50 azimuth looks in 20+5 MHz Mode.')
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
    
    if IonAzimuthLooks < 50:
        IonRangeLooks = int(IonRangeLooks * 50 / IonAzimuthLooks)
        IonAzimuthLooks = 50

    os.chdir(output_fp)
    dem_fp = os.path.join('dates', ref_date, f'hgt_{InSARRangeLooks}rlks_{InSARAzimuthLooks}alks.flt')
    para_json_fp = os.path.join('dates', ref_date, 'slc_parameters.json')
    paras = json.load(open(para_json_fp, 'r', encoding='utf-8'))
    length = paras['length']
    width = paras['width']
    range_pixel_spacing = paras['range_pixel_spacing']
    wavelength = paras['wavelength']
    rg_bd = paras['range_bandwidth']
    az_bd = paras['azimuth_bandwidth']
    prf = paras['prf']
    speed_of_light = 299792458.0
    rg_sr = speed_of_light / (2.0 * range_pixel_spacing)

    center_frequency = speed_of_light / wavelength
    high_subband_center_frequency = center_frequency + rg_bd/3
    low_subband_center_frequency = center_frequency - rg_bd/3
    high_subband_wavelength = speed_of_light / high_subband_center_frequency
    low_subband_wavelength = speed_of_light / low_subband_center_frequency

    os.makedirs('pairs_ion', exist_ok=True)
    pairs = []
    for i in range(len(dates)):
        for j in range(i+1, min(i+subsequent_num+1, len(dates))):
            pairs.append(dates[i]+'_'+dates[j])
    for i,pair in enumerate(pairs):
        pair_fp = os.path.join('pairs_ion', pair)
        os.makedirs(pair_fp, exist_ok=True)

    def build_tasks(pairs, ref_date, freq, pol,
                    _IonRangeLooks, _IonAzimuthLooks,
                    range_pixel_spacing, 
                    wavelength,subband='high'):
        tasks = []

        for i, pair in enumerate(pairs):
            pair_fp = os.path.join("pairs_ion", pair)
            ref, sed = pair.split("_")

            # ---- reference SLC ----
            if ref != ref_date:
                ref_subband_slc_fp = os.path.join('dates',ref,f'ionosphere/{subband}/fine_resample_slc/freq{freq}/{pol}/coregistered_secondary.slc')
                ref_range_off_fp = os.path.join(
                    "dates", ref,
                    f"geo2rdr/freq{freq}/range.off"
                )
            else:
                ref_subband_slc_fp = os.path.join('dates',sed,f'ionosphere/{subband}/crossmul/freq{freq}/{pol}/reference.slc')
                ref_range_off_fp = None

            # ---- secondary SLC ----
            if sed != ref_date:
                sed_subband_slc_fp = os.path.join('dates',sed,f'ionosphere/{subband}/fine_resample_slc/freq{freq}/{pol}/coregistered_secondary.slc')
                sed_range_off_fp = os.path.join(
                    "dates", sed,
                    f"geo2rdr/freq{freq}/range.off"
                )
            else:
                sed_subband_slc_fp = os.path.join('dates',ref,f'ionosphere/{subband}/crossmul/freq{freq}/{pol}/secondary.slc')
                sed_range_off_fp = None

            # ---- outputs ----
            ifgm_fp = os.path.join(
                pair_fp,
                f"{subband}_diff_{pair}_{_IonRangeLooks}rlks_{_IonAzimuthLooks}alks.int"
            )
            amp_fp = os.path.join(
                pair_fp,
                f"{subband}_{pair}_{_IonRangeLooks}rlks_{_IonAzimuthLooks}alks.amp"
            )
            cor_fp = os.path.join(
                pair_fp,
                f"{subband}_{pair}_{_IonRangeLooks}rlks_{_IonAzimuthLooks}alks.cor"
            )

            tasks.append({
                "pair": pair,
                "pair_fp": pair_fp,
                "ref": ref,
                "sed": sed,
                "ref_slc_fp": ref_subband_slc_fp,
                "sed_slc_fp": sed_subband_slc_fp,
                "ref_range_off_fp": ref_range_off_fp,
                "sed_range_off_fp": sed_range_off_fp,
                "ifgm_fp": ifgm_fp,
                "amp_fp": amp_fp,
                "cor_fp": cor_fp,
                "InSARRangeLooks": _IonRangeLooks,
                "InSARAzimuthLooks": _IonAzimuthLooks,
                "range_pixel_spacing": range_pixel_spacing,
                "wavelength": wavelength,
            })

        return tasks


    highsubband_tasks = build_tasks(
        pairs, ref_date, freq, pol,
        IonRangeLooks, IonAzimuthLooks,
        range_pixel_spacing,
        high_subband_wavelength, subband='high')

    lowsubband_tasks = build_tasks(
        pairs, ref_date, freq, pol,
        IonRangeLooks, IonAzimuthLooks,
        range_pixel_spacing,
        low_subband_wavelength, subband='low')

    tasks = highsubband_tasks + lowsubband_tasks

    num_workers = min(n*4, os.cpu_count())

    print(f"Total pairs: {len(tasks)}")
    print(f"Using {num_workers} workers")

    # with Pool(processes=num_workers) as pool:
    #     results = pool.map(process_pair, tasks)
    # for pair, status, msg in results:
    #     print(f"Generate subband data for {pair}: {status} {msg}")

    def build_ion_tasks(pairs, 
                    InSARRangeLooks, InSARAzimuthLooks,
                    IonRangeLooks, IonAzimuthLooks,IonStdOut,
                    high_wavelength, low_wavelength,
                    subbandNumberOfLooks, RangeBandWidth,
                    dem_fp=None):
        tasks = []

        for i, pair in enumerate(pairs):
            pair_fp = os.path.join("pairs_ion", pair)
            tasks.append({
                "pair": pair,
                "pair_fp": pair_fp,
                "InSARRangeLooks": InSARRangeLooks,
                "InSARAzimuthLooks": InSARAzimuthLooks,
                "IonRangeLooks": IonRangeLooks,
                "IonAzimuthLooks": IonAzimuthLooks,
                "IonStdOut": IonStdOut,
                "RangeBandWidth": RangeBandWidth,
                "high_subband_wavelength": high_wavelength,
                "low_subband_wavelength": low_wavelength,
                "subbandNumberOfLooks": subbandNumberOfLooks,
                "dem_fp": dem_fp
            })

        return tasks

    range_over_sampling_rate = rg_bd / rg_sr
    azimuth_over_sampling_rate = az_bd / prf
    subbandNumberOfLooks = IonRangeLooks * IonAzimuthLooks \
        / (range_over_sampling_rate * azimuth_over_sampling_rate * 3)
    IonStdOut = inps.IonStdOut 
    if IonStdOut == 0.0:
        IonStdOut = 0.05 * (2000) / (IonRangeLooks * IonAzimuthLooks)
    ion_tasks = build_ion_tasks(
        pairs, InSARRangeLooks, InSARAzimuthLooks,
        IonRangeLooks, IonAzimuthLooks,
        IonStdOut,high_subband_wavelength, low_subband_wavelength,
        subbandNumberOfLooks, rg_bd,
        dem_fp=dem_fp)
    with Pool(processes=num_workers) as pool:
        results = pool.map(EstimateFiltIon, ion_tasks)
    os.system('rm -rf ./snaphu.conf')
    for pair, status, msg in results:
        print(f"Estimate ionospheric phase screen for {pair}: {status} {msg}")