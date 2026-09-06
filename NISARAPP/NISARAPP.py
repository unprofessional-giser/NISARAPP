#!/usr/bin/env python3
# Created by Kunyi Chen on 2026-08-28
import os
import json
import copy
from nisar.products.readers import RSLC
import numpy as np
import h5py, json


def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Time Series InSAR Processing for NISAR')
    parser.add_argument('-config', dest='config', type=str, required=True,
            help = 'config yaml containing the input paths and parameters')
    parser.add_argument('-insar', dest='insar', type=str, required=False,
            help = 'insar yaml, i.e., NISAR DINSAR config file for ISCE3')

    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        return parser.parse_args()

def build_config_yaml_template(output_yaml_path='./config.yaml'):
    '''
    build config yaml template for NISAR DINSAR processing
    '''

    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.default_flow_style = False

    dict = {
            "Path of Input": {
            "Super Reference": "/path/to/super_reference",
            "Secondary List": ["/path/to/secondary1",
                "/path/to/secondary2",
                "/path/to/secondary3"
            ],
            "DEM TIFF File": "/path/to/dem.tif",},
            "Path of Output": {
            "Output Dict": "/path/to/output_dict/"},
            "Path of Python Interpreter": {
            "isce2": "/path/to/isce2/bin/python",
            "isce3": "/path/to/isce3/bin/python"
            },
            "Path of NISAR Workflow Software": {
            "NISARAPP": "/path/to/NISARAPP/"},
            "NumParallels": 8,
            "Coregistration": 'fine',
            "Number of subsequent dates": 2,
            "Selected Frequency and Polarization": {
            "Freq": "A",
            "Pol": "HH"},
            "Multilook Parameters": {
            "InSAR Range Looks": 4,
            "InSAR Azimuth Looks": 5,
            "Ion Range Looks": 40,
            "Ion Azimuth Looks": 50},
            "Coherence Estimation Parameters": {
            "Window Size": 11},
            "Filtering Parameters": {
            "Do Filtering": True,
            "Filter Type": "psflit",
            "Window Size": 64,
            "Window Step": 16,
            "Alpha": 0.7},
            "Error Correction Parameters": {
            "Do Ionospheric Correction": True,
            "Do Tropospheric Correction": True,
            "Do Soild Earth Tide Correction": True}
        }
    op_yaml = output_yaml_path
    data = CommentedMap()
    data["Path of Input"] = dict["Path of Input"]
    data.yaml_set_comment_before_after_key("Path of Input", before="Configs for the input paths and parameters")

    data["Path of Output"] = dict["Path of Output"]
    data.yaml_set_comment_before_after_key("Path of Output", before="\n")

    data["Selected Frequency and Polarization"] = dict["Selected Frequency and Polarization"]
    data.yaml_set_comment_before_after_key("Selected Frequency and Polarization", before="\n")

    data["Path of Python Interpreter"] = dict["Path of Python Interpreter"]
    data.yaml_set_comment_before_after_key("Path of Python Interpreter", before="\n")

    data["Path of NISAR Workflow Software"] = dict["Path of NISAR Workflow Software"]
    data.yaml_set_comment_before_after_key("Path of NISAR Workflow Software", before="\n")

    data["NumParallels"] = dict["NumParallels"]
    data.yaml_set_comment_before_after_key("NumParallels", before="\nNumber of parallel processes to run. Default is 8.\n")

    data["Coregistration"] = dict["Coregistration"]
    data.yaml_set_comment_before_after_key("Coregistration", before="\nOptions: 'coarse' or 'fine'.\n")

    data["Number of subsequent dates"] = dict["Number of subsequent dates"]
    data.yaml_set_comment_before_after_key("Number of subsequent dates", before="\n")

    data["Multilook Parameters"] = dict["Multilook Parameters"]
    data.yaml_set_comment_before_after_key("Multilook Parameters", before="\n")

    data["Coherence Estimation Parameters"] = dict["Coherence Estimation Parameters"]
    data.yaml_set_comment_before_after_key("Coherence Estimation Parameters", before="\n")

    data["Filtering Parameters"] = dict["Filtering Parameters"]
    data.yaml_set_comment_before_after_key("Filtering Parameters", before="\nWe recommend using the Goldstein filter (psfilt). For more aggressive noise reduction, use a larger window size, a smaller window step, and a lower alpha value.\n")

    data["Error Correction Parameters"] = dict["Error Correction Parameters"]
    data.yaml_set_comment_before_after_key("Error Correction Parameters", before="\nIon: Mainband Range Split Spectrum; \nTrop: pyaps with ERA5; \nSET: pysoild \n")
    with open(op_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(data, f)

def get_slc_parameters(slc_h5, freq, op_json_fp):
    '''
    Get parameters from SLC h5 file
    '''
    slc_obj = RSLC(hdf5file=os.fspath(slc_h5))
    doppler = slc_obj.getDopplerCentroid(frequency=freq)
    grid = slc_obj.getRadarGrid(frequency=freq)

    length = grid.length
    width = grid.width
    doppler_data = doppler.data
    doppler_x_start = doppler.x_start
    doppler_x_spacing = doppler.x_spacing
    doppler_x_end = doppler.x_end
    doppler_y_start = doppler.y_start
    doppler_y_spacing = doppler.y_spacing
    doppler_y_end = doppler.y_end
    dop_length = doppler.length
    dop_width = doppler.width
    starting_range = grid.starting_range
    range_pixel_spacing = grid.range_pixel_spacing
    ending_range = starting_range + range_pixel_spacing * (width - 1)
    sensing_start = grid.sensing_start
    az_time_interval = grid.az_time_interval
    sensing_stop = grid.sensing_stop
    prf = 1 / az_time_interval

    dop_x_index_min = int((starting_range - doppler_x_start) / doppler_x_spacing)
    dop_x_index_max = int((ending_range - doppler_x_start) / doppler_x_spacing)
    dop_y_index_min = int((sensing_start - doppler_y_start) / doppler_y_spacing)
    dop_y_index_max = int((sensing_stop - doppler_y_start) / doppler_y_spacing)
    slc_dop = doppler_data[dop_y_index_min:dop_y_index_max+1, dop_x_index_min:dop_x_index_max+1]
    avg_dop = np.mean(slc_dop)
    parameters = {
        'average_doppler_centroid': avg_dop,
        'starting_range': starting_range,
        'prf': prf,
        'length': length,
        'width': width
    }
    with h5py.File(slc_h5,'r') as f:
        ground_v = f['science/LSAR/RSLC/metadata/geolocationGrid/groundTrackVelocity'][:]
        sky_v = f['science/LSAR/RSLC/metadata/orbit/velocity'][:]
        frequency = f[f'science/LSAR/RSLC/swaths/frequency{freq}/processedCenterFrequency'][()]
        az_bd = f[f'/science/LSAR/RSLC/swaths/frequency{freq}/processedAzimuthBandwidth'][()]
        rg_bd = f[f'/science/LSAR/RSLC/swaths/frequency{freq}/processedRangeBandwidth'][()]
        rg_pixel_spacing = f[f'/science/LSAR/RSLC/swaths/frequency{freq}/slantRangeSpacing'][()]
        wavelength = 299792458 / frequency
        f.close()
    delta_f = az_bd / 3 * 2
    v_g = np.mean(ground_v)
    sky_v = np.sqrt(sky_v[:,0]**2 + sky_v[:,1]**2 + sky_v[:,2]**2)
    v_s = np.mean(sky_v)
    v_r = np.sqrt(v_s * v_g)
    slant_angle = np.arcsin(avg_dop * wavelength / (2 * v_s))
    K_a = 2 * v_r**2 * np.cos(slant_angle)**3 / (wavelength * starting_range)
    az_pixel_spacing = v_s / prf

    parameters['azimuth_bandwidth'] = az_bd
    parameters['average_ground_velocity'] = v_g
    parameters['average_sky_velocity'] = v_s
    parameters['average_radar_velocity'] = v_r
    parameters['slant_angle'] = slant_angle
    parameters['azimuth_frequency_modulation_rate'] = K_a
    parameters['azimuth_pixel_spacing'] = az_pixel_spacing
    parameters['range_bandwidth'] = rg_bd
    parameters['range_pixel_spacing'] = rg_pixel_spacing
    parameters['wavelength'] = wavelength

    with open(op_json_fp, 'w', encoding='utf-8') as f:
            json.dump(parameters, f, indent=4)
    
    return
    


if __name__ == '__main__':
    inps = cmdLineParse()
    config_fp = inps.config
    insar_fp = inps.insar
    if not os.path.exists(config_fp):
        build_config_yaml_template(config_fp)
        print(f'Config yaml template has been created at {config_fp}. Please fill in the required parameters and rerun the script.')
    else:
        from ruamel.yaml import YAML
        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.default_flow_style = False
        with open(config_fp, 'r', encoding='utf-8') as f:
            config = yaml.load(f)
        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.default_flow_style = False
        with open(insar_fp, 'r', encoding='utf-8') as f:
            insar_config = yaml.load(f)

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
        sed_dates.sort()
        dem_fp = config['Path of Input']['DEM TIFF File']
        freq = config['Selected Frequency and Polarization']['Freq']
        pol = config['Selected Frequency and Polarization']['Pol']
        isce2_path = config['Path of Python Interpreter']['isce2']
        isce3_path = config['Path of Python Interpreter']['isce3']
        software_fp = config['Path of NISAR Workflow Software']['NISARAPP']
        output_fp = config['Path of Output']['Output Dict']
        InSARRangeLooks = config['Multilook Parameters']['InSAR Range Looks']
        InSARAzimuthLooks = config['Multilook Parameters']['InSAR Azimuth Looks']
        IonRangeLooks = config['Multilook Parameters']['Ion Range Looks']
        IonAzimuthLooks = config['Multilook Parameters']['Ion Azimuth Looks']
        cor_win = config['Coherence Estimation Parameters']['Window Size']
        do_filter = config['Filtering Parameters']['Do Filtering']
        filter_type = config['Filtering Parameters']['Filter Type']
        filter_win = config['Filtering Parameters']['Window Size']
        filter_step = config['Filtering Parameters']['Window Step']
        filter_alpha = config['Filtering Parameters']['Alpha']
        do_ion = config['Error Correction Parameters']['Do Ionospheric Correction']
        do_tropo = config['Error Correction Parameters']['Do Tropospheric Correction']
        do_set = config['Error Correction Parameters']['Do Soild Earth Tide Correction']
        numParallels = config['NumParallels']
        coregistration = config['Coregistration'].lower()
        subsequent_num = config['Number of subsequent dates']

        ## 1. Preprocessing with ISCE3

        os.makedirs(output_fp, exist_ok=True)
        os.chdir(output_fp)
        os.makedirs('dates', exist_ok=True)
        os.makedirs('cmds', exist_ok=True)
        os.chdir('dates')
        os.makedirs(ref_date, exist_ok=True)
        slc_parameters_json_fp = os.path.join(ref_date, f'slc_parameters.json')
        get_slc_parameters(ref_fp, freq, slc_parameters_json_fp)
        for i,sed in enumerate(sed_dates):
            os.makedirs(sed_dates[i], exist_ok=True)
            _insar_yaml = copy.deepcopy(insar_config)
            _insar_yaml['runconfig']['groups']['input_file_group']['reference_rslc_file'] = ref_fp
            _insar_yaml['runconfig']['groups']['input_file_group']['secondary_rslc_file'] = sed_fps[i]
            _insar_yaml['runconfig']['groups']['dynamic_ancillary_file_group']['dem_file'] = dem_fp
            _insar_yaml['runconfig']['groups']['product_path_group']['scratch_path'] = '.'
            _insar_yaml['runconfig']['groups']['processing']['input_subset']['list_of_frequencies']['A'] = list([pol])
            _insar_yaml['runconfig']['groups']['processing']['ionosphere_phase_correction']['enabled'] = do_ion
            _insar_yaml['runconfig']['groups']['processing']['ionosphere_phase_correction']['spectral_diversity'] = 'main_diff_low_high_subband'
            _insar_yaml['runconfig']['groups']['processing']['dense_offsets']['enabled'] = True
            _insar_yaml['runconfig']['groups']['processing']['offsets_product']['enabled'] = False
            _insar_yaml['runconfig']['groups']['processing']['fine_resample']['enabled'] = True
            _insar_yaml['runconfig']['groups']['processing']['fine_resample']['offsets_dir'] = './'
            _insar_yaml['runconfig']['groups']['processing']['crossmul']['range_looks'] = InSARRangeLooks
            _insar_yaml['runconfig']['groups']['processing']['crossmul']['azimuth_looks'] = InSARAzimuthLooks
            _insar_yaml['runconfig']['groups']['processing']['filter_interferogram']['filter_type'] = 'no_filter'
            _insar_yaml_fp = os.path.join(sed_dates[i], f'InSAR_{ref_date}_{sed_dates[i]}.yaml')
            with open(_insar_yaml_fp, 'w', encoding='utf-8') as f:
                yaml.dump(_insar_yaml, f)
            run_cmd_fp = os.path.join(sed_dates[i], f'run_InSAR_{ref_date}_{sed_dates[i]}.sh')
            with open(run_cmd_fp, 'w', encoding='utf-8') as f:
                f.write(f'#!/bin/bash\n')
                f.write(f'{isce3_path} {os.path.join(software_fp, "PreProcessingISCE3.py")} {os.path.abspath(_insar_yaml_fp)}\n')
                f.write(f'rm -rf ./GUNW* ./RUNW.h5 \n')
            f.close()
            os.system(f'chmod +x {run_cmd_fp}')
            print(f'InSAR config yaml for {ref_date} and {sed_dates[i]} has been created at {os.path.abspath(_insar_yaml_fp)}')
        os.chdir('../')
        cmd_1_fp = os.path.join('cmds', f'cmd1_Preprossing.sh')
        with open(cmd_1_fp, 'w', encoding='utf-8') as f:
            flag = 0
            f.write(f'#!/bin/bash\n')
            for i, sed in enumerate(sed_dates):
                abs_date_fp = os.path.abspath(os.path.join('dates', sed))
                f.write(f'cd {abs_date_fp}\n')
                if flag < numParallels and i < len(sed_dates) - 1:
                    flag += 1
                    f.write(f'./run_InSAR_{ref_date}_{sed}.sh &\n')
                else:
                    flag = 0
                    f.write(f'./run_InSAR_{ref_date}_{sed}.sh\n')
                    f.write(f'wait\n\n')
                f.write(f'cd ../../\n')
        f.close()
        os.system(f'chmod +x {cmd_1_fp}')

        ## 2. Generate Multilooked Interferogram and Coherence
        cmd_2_fp = os.path.join('cmds', f'cmd2_Interferometry.sh')
        with open(cmd_2_fp, 'w', encoding='utf-8') as f:
            f.write(f'#!/bin/bash\n')
            f.write(f"{isce2_path} {os.path.join(software_fp, 'GenerateInSARCoh.py')} -config {os.path.abspath(config_fp)} -n {numParallels}\n")
            f.write('wait\n')
            f.write('cur_dir=$(pwd)\n')
            f.write('rm -rf $cur_dir/isce.log\n')
        f.close()
        os.system(f'chmod +x {cmd_2_fp}')

        ## 3. Ionospheric Correction
        cmd_flag = 3
        if do_ion:
            cmd_3_1_fp = os.path.join('cmds', f'cmd3_1_EstimateInSARIon.sh')
            with open(cmd_3_1_fp, 'w', encoding='utf-8') as f:
                f.write(f'#!/bin/bash\n')
                f.write(f"{isce2_path} {os.path.join(software_fp, 'EstimateInSARIon.py')} -config {os.path.abspath(config_fp)} -n {numParallels}\n")
                f.write('wait\n')
                f.write('cur_dir=$(pwd)\n')
                f.write('rm -rf $cur_dir/isce.log\n')
            f.close()
            os.system(f'chmod +x {cmd_3_1_fp}')
            cmd_3_2_fp = os.path.join('cmds', f'cmd3_2_CheckInSARIon.sh')
            with open(cmd_3_2_fp, 'w', encoding='utf-8') as f:
                f.write(f'#!/bin/bash\n')
                f.write(f"{isce2_path} {os.path.join(software_fp, 'CheckInSARIon.py')} -config {os.path.abspath(config_fp)} -n {numParallels}\n")
                f.write('wait\n')
                f.write('cur_dir=$(pwd)\n')
                f.write('rm -rf $cur_dir/isce.log\n')
            f.close()
            os.system(f'chmod +x {cmd_3_2_fp}')
            cmd_3_3_fp = os.path.join('cmds', f'cmd3_3_EstimateSARIon.sh')
            with open(cmd_3_3_fp, 'w', encoding='utf-8') as f:
                f.write(f'#!/bin/bash\n\n')
                f.write('# Check the original InSAR ionospheric phase results in ./figures_ion before running this step. \n')
                f.write('# Uncorrect results of some pairs or specific dates should be excluded.\n')
                f.write('# Example: -excluded_dates 20220101,20220115 -excluded_pairs 20220101_20220115,20220115_20220201\n\n')
                f.write(f"# {isce2_path} {os.path.join(software_fp, 'EstimateSARIon.py')} -config {os.path.abspath(config_fp)} -excluded_dates 20220101,20220115 -excluded_pairs 20220101_20220115,20220115_20220201 -n {numParallels} \n\n")
                f.write('wait\n')
                f.write('cur_dir=$(pwd)\n')
                f.write('rm -rf $cur_dir/isce.log\n')
            f.close()
            os.system(f'chmod +x {cmd_3_3_fp}')
            cmd_3_4_fp = os.path.join('cmds', f'cmd3_4_CorrectInSARIon.sh')
            with open(cmd_3_4_fp, 'w', encoding='utf-8') as f:
                f.write(f'#!/bin/bash\n')
                f.write(f"{isce2_path} {os.path.join(software_fp, 'CorrectInSARIon.py')} -config {os.path.abspath(config_fp)} -n {numParallels}\n")
                f.write('wait\n')
                f.write('cur_dir=$(pwd)\n')
                f.write('rm -rf $cur_dir/isce.log\n')
            f.close()
            os.system(f'chmod +x {cmd_3_4_fp}')
            cmd_flag +=1
        # 4. Tropospheric Correction
        if do_tropo:
            cmd_tropo_1_fp = os.path.join('cmds', f'cmd{cmd_flag}_1_EstimateSARTropo.sh')
            with open(cmd_tropo_1_fp, 'w', encoding='utf-8') as f:
                f.write(f'#!/bin/bash\n')
                f.write(f"{isce2_path} {os.path.join(software_fp, 'EstimateSARTropo.py')} -config {os.path.abspath(config_fp)} -n {numParallels}\n")
                f.write('wait\n')
                f.write('cur_dir=$(pwd)\n')
                f.write('rm -rf $cur_dir/isce.log\n')
            f.close()
            os.system(f'chmod +x {cmd_tropo_1_fp}')
            cmd_tropo_2_fp = os.path.join('cmds', f'cmd{cmd_flag}_2_CorrectInSARTropo.sh')
            with open(cmd_tropo_2_fp, 'w', encoding='utf-8') as f:
                f.write(f'#!/bin/bash\n')
                f.write(f"{isce2_path} {os.path.join(software_fp, 'CorrectInSARTropo.py')} -config {os.path.abspath(config_fp)} -n {numParallels}\n")
                f.write('wait\n')
                f.write('cur_dir=$(pwd)\n')
                f.write('rm -rf $cur_dir/isce.log\n')
            f.close()
            os.system(f'chmod +x {cmd_tropo_2_fp}')
            cmd_flag +=1

        # 5. Solid Earth Tide Correction
        if do_set:
            cmd_set_1_fp = os.path.join('cmds', f'cmd{cmd_flag}_1_EstimateSARSET.sh')
            with open(cmd_set_1_fp, 'w', encoding='utf-8') as f:
                f.write(f'#!/bin/bash\n')
                f.write(f"{isce2_path} {os.path.join(software_fp, 'EstimateSARSET.py')} -config {os.path.abspath(config_fp)} -n {numParallels}\n")
                f.write('wait\n')
                f.write('cur_dir=$(pwd)\n')
                f.write('rm -rf $cur_dir/isce.log\n')
            f.close()
            os.system(f'chmod +x {cmd_set_1_fp}')
            cmd_set_2_fp = os.path.join('cmds', f'cmd{cmd_flag}_2_CorrectInSARSET.sh')
            with open(cmd_set_2_fp, 'w', encoding='utf-8') as f:
                f.write(f'#!/bin/bash\n')
                f.write(f"{isce2_path} {os.path.join(software_fp, 'CorrectInSARSET.py')} -config {os.path.abspath(config_fp)} -n {numParallels}\n")
                f.write('wait\n')
                f.write('cur_dir=$(pwd)\n')
                f.write('rm -rf $cur_dir/isce.log\n')
            f.close()
            os.system(f'chmod +x {cmd_set_2_fp}')
            cmd_flag +=1

        # 6. Unwrap
        rm_in_extent = 'rm'
        if do_ion:
            rm_in_extent +='_ion'
        if do_tropo:
            rm_in_extent +='_tro'
        if do_set:
            rm_in_extent +='_set'
        rm_in_extent = '' if rm_in_extent == 'rm' else rm_in_extent
        cmd_unwrap_fp = os.path.join('cmds', f'cmd{cmd_flag}_Unwrap.sh')
        with open(cmd_unwrap_fp, 'w', encoding='utf-8') as f:
            f.write(f'#!/bin/bash\n')
            f.write(f"{isce2_path} {os.path.join(software_fp, 'UnwrapInSAR.py')} -config {os.path.abspath(config_fp)} -n {numParallels} -sec_fd ./{rm_in_extent}\n")
            f.write('wait\n')
            f.write('cur_dir=$(pwd)\n')
            f.write('rm -rf $cur_dir/isce.log\n')
        f.close()
        os.system(f'chmod +x {cmd_unwrap_fp}')
        cmd_flag +=1

        # 7. Time Series InSAR Processing
        cmd_ts_fp = os.path.join('cmds', f'cmd{cmd_flag}_SBAS.sh')
        with open(cmd_ts_fp, 'w', encoding='utf-8') as f:
            f.write(f'#!/bin/bash\n')
            f.write(f"{isce2_path} {os.path.join(software_fp, 'SBAS.py')} -config {os.path.abspath(config_fp)} -n {numParallels}\n")
            f.write('wait\n')
            f.write('cur_dir=$(pwd)\n')
            f.write('rm -rf $cur_dir/isce.log\n')
        f.close()
        os.system(f'chmod +x {cmd_ts_fp}')

        
