# NISARAPP
Time-series analysis extension for NISAR LSAR Data

NISARAPP is a Python-based time-series analysis extension package for **NISAR LSAR data**.  
It integrates the standard DInSAR processing flow from **ISCE3** and selected functionalities from **ISCE2**, enabling end-to-end time-series processing, including coregistration, interferometry, error correction, phase unwrapping, SBAS estimation, and geocoding.

> **Note:**  
> - Only coregistration currently uses ISCE3.  
> - All subsequent processing and output formats follow the **ISCE2 ALOS2Stack** framework.  
> - No filtering module is included in this version due to the relatively high quality of current test data; filtering will be added in a future release.

---

## Features

- Coregistration (ISCE3)
- Interferogram generation
- Error correction
- Phase unwrapping
- SBAS time-series estimation (unweighted least squares)
- Geocoding
- Command-line workflow generation

---

## Requirements

### Software dependencies

- [ISCE2](https://github.com/isce-framework/isce2)
- [ISCE3](https://github.com/isce-framework/isce3)

It is recommended to install ISCE2 and ISCE3 in **two separate Conda environments**.

### Additional Python packages (in the ISCE2 environment)

- `matplotlib`
- `yaml`
- `json`
- `pyaps`
- `pysolid`

---

## Installation

Clone this repository into your working directory and make sure ISCE2 and ISCE3 are properly installed and activated in their respective environments before running NISARAPP.

---

## Input Data Preparation

Before running the package, prepare the following:

- NISAR LSAR SLC data
- A DEM file (`dem.tif`)

---

## Usage

### 1. Configure parameters

Copy the `configs/` folder from the NISARAPP package to your processing directory.  
Edit the following files as needed:

- `config.yaml`: main parameters for NISARAPP processing
- `insar.yaml`: configuration file required by ISCE3

### 2. Generate processing commands

Run the following command to generate shell scripts:
```bash
path_of_PythonToISCE3 path_of_NISARAPP.py \

-config path_to_config.yaml \

-insar path_to_insar.yaml

```

This will create a `cmds/` folder under the output directory defined in `config.yaml`, containing a series of shell scripts (`cmd1_Preprocessing.sh`, `cmd2_Interferometry.sh`, …).

### 3. Execute processing steps

Run the generated shell scripts sequentially:
```bash
cd path/to/cmds
./cmd1_Preprocessing.sh
./cmd2_Interferometry.sh
......

```
