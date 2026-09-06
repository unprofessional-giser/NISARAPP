#!/usr/bin/env python3
# Created by Kunyi Chen on 2026-08-29

import numpy as np
import isce, isceobj
from isceobj.Alos2Proc.Alos2ProcPublic import create_xml
from isceobj.Alos2Proc.Alos2ProcPublic import multilook
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

def generate_ifgm(ref_path,sec_path,ifgm_path,amp_path,nrlks,nalks,
                  range_pixel_spacing, wavelength,
                  ref_range_off_fp = None, sec_range_off_fp = None):

    ref_width, ref_length, ref_dtype = get_hdf_metadata(ref_path)
    sec_width, sec_length, sec_dtype = get_hdf_metadata(sec_path)


    if ref_width!=sec_width or ref_length!=sec_length:
        print("Reference and Secondary SLC don't share the same shape!")
    
    else:
        print("*"*50)
        print(f"Processing Pairs: \n Ref: {ref_path} \n Sec: {sec_path}")
        print("*"*50)
        ref_range_off = 0
        sec_range_off = 0
        if ref_range_off_fp is not None:
            ref_range_off = np.memmap(ref_range_off_fp, dtype=np.float64, mode='r', shape=(ref_length, ref_width))
        if sec_range_off_fp is not None:
            sec_range_off = np.memmap(sec_range_off_fp, dtype=np.float64, mode='r', shape=(sec_length, sec_width))
        if ref_range_off_fp is None and sec_range_off_fp is None:
            print("No range offset correction applied.")
            raise ValueError("No range offset correction applied.")
        ref = np.fromfile(ref_path,dtype=np.complex64).reshape(ref_length,ref_width)
        sec = np.fromfile(sec_path,dtype=np.complex64).reshape(sec_length,sec_width)

        ml_width = int(ref_width/nrlks)
        ml_length = int(ref_length/nalks)

        ori_ifgm = ref * np.conjugate(sec)
        ori_ifgm_flatten = ori_ifgm * np.exp(1j*4*np.pi*range_pixel_spacing/wavelength*(ref_range_off-sec_range_off))
        ori_ifgm_flatten = np.complex64(ori_ifgm_flatten)
        if nalks ==1 and nrlks == 1:
            ifgm = ori_ifgm_flatten
            amp = np.sqrt(ref.real*ref.real + ref.imag*ref.imag) + 1j * np.sqrt(sec.real*sec.real + sec.imag*sec.imag)
        else:
            ifgm = multilook(ori_ifgm_flatten, nalks, nrlks, mean=False)
            amp = np.sqrt(multilook(ref.real*ref.real+ref.imag*ref.imag, nalks, nrlks, mean=False)) + 1j * \
                np.sqrt(multilook(sec.real*sec.real+sec.imag*sec.imag, nalks, nrlks, mean=False))
        index = np.nonzero( (np.real(amp)==0) + (np.imag(amp)==0) )
        amp[index]=0

        ifgm.tofile(ifgm_path)
        create_xml(ifgm_path,ml_width,ml_length,'int')
        amp = np.complex64(amp)
        amp.tofile(amp_path)
        create_xml(amp_path,ml_width,ml_length,'amp')

        print(f"Successfully generate interferogram and amplitude : \n{ifgm_path}\n")



def generate_ifgm_blocks(ref_path, sec_path, ifgm_path, amp_path, nrlks, nalks,
                  range_pixel_spacing, wavelength,
                  ref_range_off_fp=None, sec_range_off_fp=None,
                  chunk_lines=2048):

    ref_width, ref_length, ref_dtype = get_hdf_metadata(ref_path)
    sec_width, sec_length, sec_dtype = get_hdf_metadata(sec_path)

    if ref_width != sec_width or ref_length != sec_length:
        raise ValueError("Reference and Secondary SLC don't share the same shape!")
    chunk_lines = (chunk_lines//nalks)*nalks
    print("*" * 50)
    print(f"Processing Pairs: \n Ref: {ref_path} \n Sec: {sec_path}")
    print(f"Image size: {ref_width} x {ref_length}, chunk_lines={chunk_lines}")
    print(f'Number of looks: azlooks {nalks} and rglooks {nrlks}')
    print("*" * 50)

    ref = np.memmap(ref_path, dtype=np.complex64, mode='r',
                    shape=(ref_length, ref_width))
    sec = np.memmap(sec_path, dtype=np.complex64, mode='r',
                    shape=(sec_length, sec_width))
    ref_range_off = None
    sec_range_off = None
    if ref_range_off_fp is not None:
        ref_range_off = np.memmap(ref_range_off_fp, dtype=np.float64, mode='r',
                                   shape=(ref_length, ref_width))
    if sec_range_off_fp is not None:
        sec_range_off = np.memmap(sec_range_off_fp, dtype=np.float64, mode='r',
                                   shape=(sec_length, sec_width))
    if ref_range_off is None and sec_range_off is None:
        raise ValueError("No range offset correction applied.")

    ml_width = int(ref_width / nrlks)
    ml_length = int(ref_length / nalks)

    ifgm_out = np.memmap(ifgm_path, dtype=np.complex64, mode='w+',
                         shape=(ml_length, ml_width))
    amp_out = np.memmap(amp_path, dtype=np.complex64, mode='w+',
                        shape=(ml_length, ml_width))

    for y_start in range(0, ref_length, chunk_lines):
        y_end = min(y_start + chunk_lines, ref_length)
        
        print(f"  Processing lines {y_start}:{y_end} / {ref_length}", end='\r')
        ref_chunk = ref[y_start:y_end, :]
        sec_chunk = sec[y_start:y_end, :]
        if ref_range_off is not None:
            roff_chunk = ref_range_off[y_start:y_end, :]
        else:
            roff_chunk = 0
        if sec_range_off is not None:
            soff_chunk = sec_range_off[y_start:y_end, :]
        else:
            soff_chunk = 0
        ifgm_chunk = ref_chunk * np.conjugate(sec_chunk)
        if not (isinstance(roff_chunk, int) and isinstance(soff_chunk, int)):
            flatten_phase = np.exp(
                1j * 4 * np.pi * range_pixel_spacing / wavelength *
                (roff_chunk - soff_chunk)
            )
            ifgm_chunk = ifgm_chunk * flatten_phase
        amp_chunk = (
            np.sqrt(np.abs(ref_chunk)**2) +
            1j * np.sqrt(np.abs(sec_chunk)**2)
        )
        
        if nalks > 1 or nrlks > 1:
            ifgm_chunk = multilook(ifgm_chunk, nalks, nrlks, mean=False)
            amp_chunk = np.sqrt(multilook(ref_chunk.real**2 + ref_chunk.imag**2, nalks, nrlks, mean=False)) + \
                        np.sqrt(multilook(sec_chunk.real**2 + sec_chunk.imag**2, nalks, nrlks, mean=False)) * 1j

        out_y_start = y_start // nalks
        out_y_end = y_end // nalks 
        
        ifgm_out[out_y_start:out_y_end, :] = ifgm_chunk.astype(np.complex64)
        amp_out[out_y_start:out_y_end, :] = amp_chunk.astype(np.complex64)

        del ref_chunk, sec_chunk, ifgm_chunk, amp_chunk

    # mask = np.zeros((ml_length, ml_width), dtype=bool)
    # mask[:,1600//nrlks:ml_width-1600//nrlks] = True
    # ifgm_out = ifgm_out * mask
    # amp_out = amp_out * mask
    ifgm_out.flush()
    amp_out.flush()
    del ref, sec, ifgm_out, amp_out
    if ref_range_off is not None:
        del ref_range_off
    if sec_range_off is not None:
        del sec_range_off

    create_xml(ifgm_path, ml_width, ml_length, 'int')
    create_xml(amp_path, ml_width, ml_length, 'amp')

    print(f"Successfully generate interferogram and amplitude: \n{ifgm_path}\n")

def multilook_and_coherence(ref_slc, sec_slc, ifgm_file, amp_file, cor_file, rlks,alks, 
                            range_pixel_spacing, wavelength, coherence_win=11,
                            ref_range_off_fp = None, sec_range_off_fp = None):
    from isceobj.Alos2Proc.runCoherence import coherence

    generate_ifgm_blocks(ref_slc,sec_slc,ifgm_file,amp_file,rlks,alks,range_pixel_spacing,wavelength,ref_range_off_fp,sec_range_off_fp)
    coherence(amp_file, ifgm_file, cor_file, 
            method="cchz_wave", windowSize=coherence_win)
    return

def process_pair(args):
    pair = args["pair"]
    pair_fp = args["pair_fp"]
    ref = args["ref"]
    sed = args["sed"]
    ref_slc_fp = args["ref_slc_fp"]
    sed_slc_fp = args["sed_slc_fp"]
    ref_range_off_fp = args["ref_range_off_fp"]
    sed_range_off_fp = args["sed_range_off_fp"]
    ifgm_fp = args["ifgm_fp"]
    amp_fp = args["amp_fp"]
    cor_fp = args["cor_fp"]
    InSARRangeLooks = args["InSARRangeLooks"]
    InSARAzimuthLooks = args["InSARAzimuthLooks"]
    range_pixel_spacing = args["range_pixel_spacing"]
    wavelength = args["wavelength"]
    coh_win = args["coherence_win"] if "coherence_win" in args else 11

    try:
        print(f"[START] {pair}")
        multilook_and_coherence(
            ref_slc_fp, sed_slc_fp,
            ifgm_fp, amp_fp, cor_fp,
            InSARRangeLooks, InSARAzimuthLooks,
            range_pixel_spacing, wavelength, coh_win,
            ref_range_off_fp, sed_range_off_fp
        )
        print(f"[DONE]  {pair}")
        return (pair, "SUCCESS", "")
    except Exception as e:
        print(f"[FAIL]  {pair} -> {e}")
        return (pair, "FAILED", str(e))

def get_multlooked_lat_lon_hgt(slat_fp:str, slon_fp:str,shgt_fp:str, azlooks:int, rglooks:int, chunklines:int=2048):
    ori_width, ori_length, dtype = get_hdf_metadata(slat_fp)
    ml_width = ori_width // rglooks
    ml_length = ori_length // azlooks
    chunklines = (chunklines//azlooks)*azlooks
    slat = np.memmap(slat_fp, dtype=dtype, mode='r').reshape(ori_length, ori_width)
    slon = np.memmap(slon_fp, dtype=dtype, mode='r').reshape(ori_length, ori_width)
    shgt = np.memmap(shgt_fp, dtype=dtype, mode='r').reshape(ori_length, ori_width)
    ml_slat = np.zeros((ml_length, ml_width), dtype=np.float32)
    ml_slon = np.zeros((ml_length, ml_width), dtype=np.float32)
    ml_shgt = np.zeros((ml_length, ml_width), dtype=np.float32)
    for i in range(0, ori_length, chunklines):
        end_i = min(i + chunklines, ori_length)
        block_min_i_index = i // azlooks
        block_max_i_index = end_i // azlooks
        d_i = block_max_i_index - block_min_i_index
        slat_block = slat[block_min_i_index*azlooks:block_max_i_index*azlooks, :ml_width*rglooks]
        slon_block = slon[block_min_i_index*azlooks:block_max_i_index*azlooks, :ml_width*rglooks]
        shgt_block = shgt[block_min_i_index*azlooks:block_max_i_index*azlooks, :ml_width*rglooks]
        ml_slat[block_min_i_index:block_max_i_index, :] = slat_block.reshape(d_i, azlooks, ml_width, rglooks).mean(axis=(1,3))
        ml_slon[block_min_i_index:block_max_i_index, :] = slon_block.reshape(d_i, azlooks, ml_width, rglooks).mean(axis=(1,3))
        ml_shgt[block_min_i_index:block_max_i_index, :] = shgt_block.reshape(d_i, azlooks, ml_width, rglooks).mean(axis=(1,3))
    del slat, slon, shgt
    return ml_slat, ml_slon, ml_shgt

def ecef_to_enu(lat, lon, x, y, z):
    lat, lon = np.radians(lat), np.radians(lon)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)

    R = np.array([
        [-sin_lon,              cos_lon,               0],
        [-sin_lat*cos_lon, -sin_lat*sin_lon,      cos_lat],
        [ cos_lat*cos_lon,  cos_lat*sin_lon,      sin_lat]
    ])
    return R @ np.array([x, y, z])

def compute_los(ref_rslc_fp: str, lat: float = None, lon: float = None,
                rglooks: int = 1, azlooks: int = 1):
    """
    Compute the line-of-sight (LOS) vector for each pixel in the interferogram using the satellite orbit information from the RIFG file.

    Parameters
    ----------
    ref_rslc_fp : str
        File path to the reference SLC file containing orbit data.

    Returns
    -------
    los : numpy.ndarray
        Line-of-sight vector for each pixel in the interferogram.
    hgt : numpy.ndarray
        Digital elevation model (DEM) height for each pixel in the interferogram [meters].
    """
    pos = None
    time = None
    t_azi = None
    vel = None
    slantRange = None
    with h5py.File(ref_rslc_fp, 'r', libver='latest', swmr=True) as fid:

        ref_pos = fid['/science/LSAR/RSLC/metadata/orbit/position'][()]
        time = fid['/science/LSAR/RSLC/metadata/orbit/time'][()]
        ref_vel = fid['/science/LSAR/RSLC/metadata/orbit/velocity'][()]
        
        centroidDopplerTime = fid['/science/LSAR/RSLC/swaths/zeroDopplerTime'][()]
        centroidDopplerTime = np.array(centroidDopplerTime)
        slantRange = fid['/science/LSAR/RSLC/swaths/frequencyA/slantRange'][()]
        slantRange = np.array(slantRange)

        ori_ct_len = len(centroidDopplerTime)
        ori_sr_len = len(slantRange)
        ct_len = ori_ct_len // azlooks
        sr_len = ori_sr_len // rglooks
        centroidDopplerTime = centroidDopplerTime[:ct_len*azlooks].reshape(ct_len, azlooks).mean(axis=1)
        slantRange = slantRange[:sr_len*rglooks].reshape(sr_len, rglooks).mean(axis=1)
        t_azi = 0.0
        if ct_len % 2 == 0:
            mid_idx = ct_len // 2
            t_azi = (centroidDopplerTime[mid_idx - 1] + centroidDopplerTime[mid_idx]) / 2.0
        else:
            mid_idx = ct_len // 2
            t_azi = centroidDopplerTime[mid_idx]
    
    ref_full_v = np.array([np.interp(centroidDopplerTime, time, ref_vel[:, i]) for i in range(3)])
    ref_full_v = ecef_to_enu(lat, lon, ref_full_v[0], ref_full_v[1], ref_full_v[2])
    ref_azimuth_direction = np.rad2deg(np.arctan2(ref_full_v[1], ref_full_v[0]))
    
    # The radar flight direction is defined as the angle between the east direction and the satellite velocity vector, measured counterclockwise from east. 
    azimuth_dir = np.tile(ref_azimuth_direction.reshape(ct_len, 1), (1, sr_len))

    a = 6378137.0
    e2 = 6.69437999014e-3
    p = np.array([np.interp(t_azi, time, ref_pos[:, i]) for i in range(3)])

    hsat = np.linalg.norm(p)
    phi = np.radians(lat)
    # Heading Angle, defined as the angle between the north direction and the satellite velocity vector, measured clockwise from north. 
    heading = 90 - azimuth_dir[ct_len//2, sr_len//2]
    alpha = np.radians(heading)

    M = a * (1.0 - e2) / (1.0 - e2 * np.sin(phi)**2)**1.5   # meridional
    N = a / np.sqrt(1.0 - e2 * np.sin(phi)**2)              # prime vertical

    denom = (np.cos(alpha)**2 / M) + (np.sin(alpha)**2 / N)
    R = 1.0 / denom

    cosSupplementaryAngle = (R**2 + slantRange**2 - hsat**2) / (2.0 * R * slantRange)
    inc = np.pi - np.arccos(cosSupplementaryAngle)
    inc *= (180.0 / np.pi)
    inc = np.tile(inc.reshape(1, sr_len), (ct_len, 1))

    # Azimuth angle is defined as the angle between the north direction and the projection of the LOS vector onto the horizontal plane, measured clockwise from north.
    # NISAR is a left-looking radar, which means the LOS vector points to the left of the flight direction. Therefore, we need to negate the azimuth direction to get the correct LOS azimuth angle.
    aza = -1.0 * azimuth_dir

    return inc, aza

def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Generate InSAR Coherence and Interferogram for NISAR Time Series Processing')
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
    InSARRangeLooks = config['Multilook Parameters']['InSAR Range Looks']
    InSARAzimuthLooks = config['Multilook Parameters']['InSAR Azimuth Looks']
    coregistration = config['Coregistration'].lower()
    coherence_win = config['Coherence Estimation Parameters']['Window Size']

    os.chdir(output_fp)
    para_json_fp = os.path.join('dates', ref_date, 'slc_parameters.json')
    paras = json.load(open(para_json_fp, 'r', encoding='utf-8'))
    length = paras['length']
    width = paras['width']
    range_pixel_spacing = paras['range_pixel_spacing']
    wavelength = paras['wavelength']

    os.makedirs('pairs', exist_ok=True)

    pairs = []
    for i in range(len(dates)):
        for j in range(i+1, min(i+subsequent_num+1, len(dates))):
            pairs.append(dates[i]+'_'+dates[j])
    for i,pair in enumerate(pairs):
        pair_fp = os.path.join('pairs', pair)
        os.makedirs(pair_fp, exist_ok=True)

    slat_fp = os.path.join('dates', sed_dates[0], f'rdr2geo/freq{freq}/y.rdr')
    slon_fp = os.path.join('dates', sed_dates[0], f'rdr2geo/freq{freq}/x.rdr')
    shgt_fp = os.path.join('dates', sed_dates[0], f'rdr2geo/freq{freq}/z.rdr')
    ml_slat, ml_slon, ml_shgt = get_multlooked_lat_lon_hgt(slat_fp, slon_fp, shgt_fp, InSARAzimuthLooks, InSARRangeLooks)
    ml_slat_fp = os.path.join('dates', ref_date, f'lat_{InSARRangeLooks}rlks_{InSARAzimuthLooks}alks.flt')
    ml_slon_fp = os.path.join('dates', ref_date, f'lon_{InSARRangeLooks}rlks_{InSARAzimuthLooks}alks.flt')
    ml_shgt_fp = os.path.join('dates', ref_date, f'hgt_{InSARRangeLooks}rlks_{InSARAzimuthLooks}alks.flt')
    ml_slat.astype(np.float32).tofile(ml_slat_fp)
    ml_slon.astype(np.float32).tofile(ml_slon_fp)
    ml_shgt.astype(np.float32).tofile(ml_shgt_fp)
    create_xml(ml_slat_fp, ml_slon.shape[1], ml_slon.shape[0], 'float')
    create_xml(ml_slon_fp, ml_slon.shape[1], ml_slon.shape[0], 'float')
    create_xml(ml_shgt_fp, ml_slon.shape[1], ml_slon.shape[0], 'float')
    avg_lat = np.mean(ml_slat)
    avg_lon = np.mean(ml_slon)
    avg_hgt = np.mean(ml_shgt)
    del ml_slat, ml_slon, ml_shgt
    ref_rslc_fp = ref_fp
    inc, aza = compute_los(ref_rslc_fp, avg_lat, avg_lon, InSARRangeLooks, InSARAzimuthLooks)
    inc_fp = os.path.join('dates', ref_date, f'inc_{InSARRangeLooks}rlks_{InSARAzimuthLooks}alks.flt')
    aza_fp = os.path.join('dates', ref_date, f'aza_{InSARRangeLooks}rlks_{InSARAzimuthLooks}alks.flt')
    inc.astype(np.float32).tofile(inc_fp)
    aza.astype(np.float32).tofile(aza_fp)
    create_xml(inc_fp, inc.shape[1], inc.shape[0], 'float')
    create_xml(aza_fp, aza.shape[1], aza.shape[0], 'float')
    del inc, aza


    # for i, pair in enumerate(pairs):
    #     print("*"*50)
    #     print(f"Processing pair {i+1}/{len(pairs)}: {pair}")
    #     pair_fp = os.path.join('pairs', pair)
    #     ref, sed = pair.split('_')
    #     if ref != ref_date:
    #         ref_slc_fp = os.path.join('dates', ref, f'fine_resample_slc/freq{freq}/{pol}','coregistered_secondary.slc')
    #         ref_range_off_fp = os.path.join('dates', ref, f'geo2rdr/freq{freq}/range.off')
    #     else:
    #         ref_slc_fp = os.path.join('dates', sed, f'crossmul/freq{freq}/{pol}','reference.slc')
    #         ref_range_off_fp = None
    #     if sed != ref_date:
    #         sed_slc_fp = os.path.join('dates', sed, f'fine_resample_slc/freq{freq}/{pol}','coregistered_secondary.slc')
    #         sed_range_off_fp = os.path.join('dates', sed, f'geo2rdr/freq{freq}/range.off')
    #     else:
    #         sed_slc_fp = os.path.join('dates', ref, f'crossmul/freq{freq}/{pol}','secondary.slc')
    #         sed_range_off_fp = None
    #     ifgm_fp = os.path.join(pair_fp, f'diff_{pair}_{InSARRangeLooks}rlks_{InSARAzimuthLooks}alks.int')
    #     amp_fp = os.path.join(pair_fp, f'{pair}_{InSARRangeLooks}rlks_{InSARAzimuthLooks}alks.amp')
    #     cor_fp = os.path.join(pair_fp, f'{pair}_{InSARRangeLooks}rlks_{InSARAzimuthLooks}alks.cor')
    #     multilook_and_coherence(ref_slc_fp, sed_slc_fp, ifgm_fp, amp_fp, cor_fp, InSARRangeLooks, InSARAzimuthLooks,
    #                             range_pixel_spacing, wavelength,
    #                             ref_range_off_fp, sed_range_off_fp)
    #     print(f"Successfully generate interferogram, amplitude and coherence for pair {pair} \n")
    def build_tasks(pairs, ref_date, freq, pol,
                    InSARRangeLooks, InSARAzimuthLooks,
                    range_pixel_spacing, wavelength,
                    coherence_win=5):
        tasks = []

        for i, pair in enumerate(pairs):
            pair_fp = os.path.join("pairs", pair)
            ref, sed = pair.split("_")

            # ---- reference SLC ----
            if ref != ref_date:
                ref_slc_fp = os.path.join(
                    "dates", ref,
                    f"{coregistration}_resample_slc/freq{freq}/{pol}",
                    "coregistered_secondary.slc"
                )
                ref_range_off_fp = os.path.join(
                    "dates", ref,
                    f"geo2rdr/freq{freq}/range.off"
                )
            else:
                ref_slc_fp = os.path.join(
                    "dates", sed,
                    f"crossmul/freq{freq}/{pol}",
                    "reference.slc"
                )
                ref_range_off_fp = None

            # ---- secondary SLC ----
            if sed != ref_date:
                sed_slc_fp = os.path.join(
                    "dates", sed,
                    f"{coregistration}_resample_slc/freq{freq}/{pol}",
                    "coregistered_secondary.slc"
                )
                sed_range_off_fp = os.path.join(
                    "dates", sed,
                    f"geo2rdr/freq{freq}/range.off"
                )
            else:
                sed_slc_fp = os.path.join(
                    "dates", ref,
                    f"crossmul/freq{freq}/{pol}",
                    "secondary.slc"
                )
                sed_range_off_fp = None

            # ---- outputs ----
            ifgm_fp = os.path.join(
                pair_fp,
                f"diff_{pair}_{InSARRangeLooks}rlks_{InSARAzimuthLooks}alks_{coregistration}.int"
            )
            amp_fp = os.path.join(
                pair_fp,
                f"{pair}_{InSARRangeLooks}rlks_{InSARAzimuthLooks}alks_{coregistration}.amp"
            )
            cor_fp = os.path.join(
                pair_fp,
                f"{pair}_{InSARRangeLooks}rlks_{InSARAzimuthLooks}alks_{coregistration}.cor"
            )

            tasks.append({
                "pair": pair,
                "pair_fp": pair_fp,
                "ref": ref,
                "sed": sed,
                "ref_slc_fp": ref_slc_fp,
                "sed_slc_fp": sed_slc_fp,
                "ref_range_off_fp": ref_range_off_fp,
                "sed_range_off_fp": sed_range_off_fp,
                "ifgm_fp": ifgm_fp,
                "amp_fp": amp_fp,
                "cor_fp": cor_fp,
                "InSARRangeLooks": InSARRangeLooks,
                "InSARAzimuthLooks": InSARAzimuthLooks,
                "range_pixel_spacing": range_pixel_spacing,
                "wavelength": wavelength,
                "coherence_win": coherence_win,
            })

        return tasks

    tasks = build_tasks(
            pairs, ref_date, freq, pol,
            InSARRangeLooks, InSARAzimuthLooks,
            range_pixel_spacing, wavelength, coherence_win=5)

    num_workers = min(n, os.cpu_count())

    print(f"Total pairs: {len(tasks)}")
    print(f"Using {num_workers} workers")

    with Pool(processes=num_workers) as pool:
        results = pool.map(process_pair, tasks)
    for pair, status, msg in results:
        print(f"{pair}: {status} {msg}")