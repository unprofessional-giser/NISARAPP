#!/usr/bin/env python3
import time
import shutil
import pathlib

import journal
from nisar.workflows import (bandpass_insar, baseline, crossmul, dense_offsets,
                             filter_interferogram, geo2rdr, geocode_insar,
                             h5_prep, ionosphere, offsets_product,
                             prepare_insar_hdf5, rdr2geo, resample_slc_v2,
                             rubbersheet, solid_earth_tides, split_spectrum,
                             troposphere, unwrap)
from nisar.workflows.geocode_insar import InputProduct
from nisar.workflows.insar_runconfig import InsarRunConfig
from nisar.workflows.persistence import Persistence
from nisar.workflows.yaml_argparse import YamlArgparse


def run(cfg: dict, out_paths: dict, run_steps: dict):
    '''
    Run INSAR workflow with parameters in cfg dictionary
    '''
    info_channel = journal.info("insar.run")
    info_channel.log("starting INSAR preprocessing")
    t_all = time.time()

    if run_steps['bandpass_insar']:
        bandpass_insar.run(cfg)

    if run_steps['rdr2geo']:
        rdr2geo.run(cfg)

    if run_steps['geo2rdr']:
        geo2rdr.run(cfg)

    if run_steps['prepare_insar_hdf5']:
        prepare_insar_hdf5.run(cfg)

    if run_steps['coarse_resample']:
        resample_slc_v2.run(cfg, 'coarse')

    if (run_steps['dense_offsets']) and \
            (cfg['processing']['dense_offsets']['enabled']):
        dense_offsets.run(cfg)

    if (run_steps['offsets_product']) and \
            (cfg['processing']['offsets_product']['enabled']):
        offsets_product.run(cfg, out_paths['ROFF'])

    if run_steps['rubbersheet'] and \
            cfg['processing']['rubbersheet']['enabled'] and \
            'RIFG' in out_paths:
        rubbersheet.run(cfg, out_paths['RIFG'])

    # If enabled, run fine_resampling
    if (
        run_steps['fine_resample']
        and cfg['processing']['fine_resample']['enabled']
        and 'RIFG' in out_paths
    ):
        resample_slc_v2.run(cfg, 'fine')
    
    if run_steps['crossmul'] and 'RIFG' in out_paths:
        if cfg['processing']['fine_resample']['enabled']:
            print("Running crossmul with fine-coregistered SLC...")
            crossmul.run(cfg, out_paths['RIFG'], 'fine')
        else:
            print("Running crossmul with coarse-coregistered SLC...")
            crossmul.run(cfg, out_paths['RIFG'], 'coarse')
    # Run insar_filter only
    if run_steps['filter_interferogram'] and \
        cfg['processing']['filter_interferogram']['filter_type'] != 'no_filter' and \
            'RIFG' in out_paths:
        filter_interferogram.run(cfg, out_paths['RIFG'])

    if run_steps['ionosphere'] and \
            cfg['processing']['ionosphere_phase_correction']['enabled'] and \
            'RUNW' in out_paths:
        split_spectrum.run(cfg)
        ionosphere.run(cfg, out_paths['RUNW'])

    t_all_elapsed = time.time() - t_all
    info_channel.log(f"Successfully ran INSAR preprocessing in {t_all_elapsed:.3f} seconds")


if __name__ == "__main__":
    # parse CLI input
    yaml_parser = YamlArgparse()
    args = yaml_parser.parse()

    # convert CLI input to run configuration
    insar_runcfg = InsarRunConfig(args)

    # To allow persistence, a logfile is required. Raise exception
    # if logfile is None and persistence is requested
    logfile_path = insar_runcfg.cfg['logging']['path']
    if (logfile_path is None) and insar_runcfg.args.restart:
        raise ValueError('InSAR workflow persistence requires to specify a logfile')
    persist = Persistence(logfile_path, insar_runcfg.args.restart)

    # run InSAR workflow
    if persist.run:
        _, out_paths = h5_prep.get_products_and_paths(insar_runcfg.cfg)
        run(insar_runcfg.cfg, out_paths, persist.run_steps)
