import numpy as np
import isce, isceobj
from isceobj.Alos2Proc.Alos2ProcPublic import create_xml
import os, json
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator

def geocode(
    insar_data,
    lat_rdr,
    lon_rdr,
    hgt_rdr,
    output_path,
    boundary=None,
    dlat=None,
    dlon=None,
    method='linear',
    fill_value=np.nan, 
):
    """
    Geocode InSAR data from radar to regular geographic grid.
    Optimized version: LinearNDInterpolator instead of griddata.
    """

    # ---- 1. Input --------------------------------------------------------
    data = np.asarray(insar_data)
    n_bands, n_lines_rdr, n_samples_rdr = data.shape
    print(f"[geocode] Input InSAR shape: ({n_lines_rdr}, {n_samples_rdr}, {n_bands})")

    mask1 = data[0] == 0.0
    mask2 = data[1] == 0.0
    mask = mask1 & mask2
    data[:, mask] = fill_value
    # ---- 2. Validation ----------------------------------------------------
    for name, arr in [("lat_rdr", lat_rdr), ("lon_rdr", lon_rdr)]:
        if arr.shape != (n_lines_rdr, n_samples_rdr):
            raise ValueError(
                f"{name} shape {arr.shape} != InSAR shape {(n_lines_rdr, n_samples_rdr)}"
            )

    # ---- 3. Source coordinates -------------------------------------------
    lon_flat = lon_rdr.astype(np.float64).ravel()
    lat_flat = lat_rdr.astype(np.float64).ravel()

    valid = np.isfinite(lon_flat) & np.isfinite(lat_flat)
    valid &= (np.abs(lon_flat) <= 180.0) & (np.abs(lat_flat) <= 90.0)

    src_pts = np.column_stack((lon_flat[valid], lat_flat[valid]))
    _, uid = np.unique(src_pts, axis=0, return_index=True)
    uid = np.sort(uid)

    src_lon = src_pts[uid, 0].astype(np.float64)
    src_lat = src_pts[uid, 1].astype(np.float64)

    # ---- 4. Output geographic grid ---------------------------------------
    if dlon is None:
        dlon = (np.nanmax(lon_flat) - np.nanmin(lon_flat)) / (n_samples_rdr - 1)
    if dlat is None:
        dlat = (np.nanmax(lat_flat) - np.nanmin(lat_flat)) / (n_lines_rdr - 1)

    if boundary is not None:
        lon_min, lon_max, lat_min, lat_max = boundary
    else:
        lon_min, lon_max = src_lon.min(), src_lon.max()
        lat_min, lat_max = src_lat.min(), src_lat.max()

    lon_min = np.floor(lon_min / dlon) * dlon
    lon_max = np.ceil(lon_max / dlon) * dlon
    lat_min = np.floor(lat_min / dlat) * dlat
    lat_max = np.ceil(lat_max / dlat) * dlat

    n_cols = int(np.round((lon_max - lon_min) / dlon)) + 1
    n_rows = int(np.round((lat_max - lat_min) / dlat)) + 1

    out_lon = np.linspace(lon_min, lon_max, n_cols)
    out_lat = np.linspace(lat_max, lat_min, n_rows)
    Lon_grid, Lat_grid = np.meshgrid(out_lon, out_lat)

    # ---- 5. Build interpolator ONCE --------------------------------------
    print(f"[geocode] Interpolation method: {method}")

    if method == 'nearest':
        interp = None  
    elif method == 'linear':
        interp = LinearNDInterpolator(
            (src_lon, src_lat),
            np.zeros_like(src_lon),
            fill_value=fill_value,
            rescale=True,
        )
    else:
        raise NotImplementedError(method)

    # ---- 6. Interpolate band-by-band -------------------------------------
    out_data = np.zeros((n_rows, n_bands, n_cols), dtype=np.float32)

    for b in range(n_bands):
        print(f"[geocode] Interpolating band {b+1}/{n_bands}")

        band_flat = data[b].astype(np.float64).ravel()
        band_flat[~valid] = np.nan
        band_src = band_flat[valid][uid]

        if method == 'nearest':
            interp = NearestNDInterpolator(
                (src_lon, src_lat),
                band_src.reshape(-1, 1),
                rescale=True,
            )
            interpolated = interp(Lon_grid, Lat_grid)
        else:
            interp.values = band_src.reshape(-1, 1)
            interpolated = interp(Lon_grid, Lat_grid)

        out_data[:, b, :] = interpolated.astype(np.float32)

    # ---- 7. Write ENVI ---------------------------------------------------
    _write_envi(out_data, output_path, dlat, dlon, lat_max, lon_min)

    try:
        if n_bands == 1:
            create_xml(output_path, n_cols, n_rows, 'float')
        elif n_bands == 2:
            create_xml(output_path, n_cols, n_rows, 'cor')
    except Exception as e:
        print(f"[geocode] Warning: ISCE XML creation failed: {e}")

    return n_cols, n_rows, output_path + '.hdr'

# ---------------------------------------------------------------------------
# ENVI output writers
# ---------------------------------------------------------------------------

def _write_envi(data, output_path, dlat, dlon, lat_north, lon_west,
                description='Data product generated using custom geocode'):
    """
    Write data as ENVI Standard BIL binary file with companion .hdr.

    Parameters
    ----------
    data : numpy.ndarray
        (n_bands, n_lines, n_samples) or (n_lines, n_samples).
    output_path : str
        Full path without extension.
    dlat, dlon : float
        Grid spacing in degrees (positive values).
    lat_north : float
        Latitude of the northern edge (center of top pixel row).
    lon_west : float
        Longitude of the western edge (center of left pixel column).
    description : str
        Free-form description written to the header.
    """

    if data.ndim == 2:
        data = data[np.newaxis, :, :]
        n_bands = 1
        n_lines, n_samples = data.shape[0], data.shape[1]
    else:
        n_lines,n_bands,n_samples = data.shape[0], data.shape[1], data.shape[2]

    # Ensure float32 for ENVI data type 4
    data = data.astype(np.float32)
    data.tofile(output_path)

    # Build header
    dtype_code = 4  # float32
    byte_order = 0  # little-endian

    # ENVI map_info convention:
    # map_info = {Geographic Lat/Lon, x0, y0, x_west, y_north, dx, dy, datum, units=Degrees}
    # x0, y0 are pixel indices (1-based) of the reference point
    map_info = (
        f'{{Geographic Lat/Lon, 1.0, 1.0, {lon_west:.16f}, '
        f'{lat_north:.16f}, {dlon:.16f}, {dlat:.16f}, WGS-84, units=Degrees}}'
    )

    coord_sys = (
        'GEOGCS["GCS_WGS_1984",'
        'DATUM["D_WGS_1984",'
        'SPHEROID["WGS_1984",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],'
        'UNIT["Degree",0.0174532925199433]]'
    )

    hdr_lines = [
        'ENVI',
        f'description = {{{description}}}',
        f'samples = {n_samples}',
        f'lines   = {n_lines}',
        f'bands   = {n_bands}',
        'header offset = 0',
        'file type = ENVI Standard',
        f'data type = {dtype_code}',
        'interleave = bil',
        f'byte order = {byte_order}',
        f'coordinate system string = {{{coord_sys}}}',
        f'map_info = {map_info}',
    ]

    hdr_path = output_path + '.hdr'
    with open(hdr_path, 'w') as f:
        f.write('\n'.join(hdr_lines) + '\n')

    print(f"[geocode] Wrote binary : {output_path}")
    print(f"[geocode] Wrote header : {hdr_path}")
    print(f"[geocode] Shape: ({n_lines}, {n_samples}, {n_bands}), "
          f"grid: {dlon:.8f} x {dlat:.8f} deg")

def cmdLineParse():
    '''
    command line parser.
    '''
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Geocode InSAR data from radar to regular geographic grid.')
    parser.add_argument('-config', dest='config', type=str, required=True,
            help = 'config yaml containing the input paths and parameters')
    parser.add_argument('-input', dest='input', type=str, required=True,
            help = 'input InSAR file to be geocoded')
    parser.add_argument('-lat', dest='lat', type=str, required=False,
            help = 'input latitude file in rdr for geocoding, default is dates/<ref_date>/lat_<InSARLooks>.flt')
    parser.add_argument('-lon', dest='lon', type=str, required=False,
            help = 'input longitude file in rdr for geocoding, default is dates/<ref_date>/lon_<InSARLooks>.flt')
    parser.add_argument('-hgt', dest='hgt', type=str, required=False,
            help = 'input height file in rdr for geocoding, default is dates/<ref_date>/hgt_<InSARLooks>.flt')
    parser.add_argument('-output', dest='output', type=str, required=False, default=None,
            help = 'output geocoded file path, default is input path with .geo suffix')
    parser.add_argument('-dlat', dest='dlat', type=float, required=False, default=3,
            help = 'output grid spacing in latitude (arcsec), default is 3 arcsec')
    parser.add_argument('-dlon', dest='dlon', type=float, required=False, default=3,
            help = 'output grid spacing in longitude (arcsec), default is 3 arcsec')
    parser.add_argument('-method', dest='method', type=str, required=False, default='linear',
            help = 'interpolation method (nearest, linear, cubic), default is linear')
    parser.add_argument('-fill_value', dest='fill_value', type=float, required=False, default=np.nan,
            help = 'fill value for missing data, default is NaN')

    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        return parser.parse_args()


if __name__ == "__main__":
    inps = cmdLineParse()
    config_fp = inps.config
    input_fp = inps.input
    lat_fp = inps.lat
    lon_fp = inps.lon
    hgt_fp = inps.hgt
    geo_output_fp = inps.output
    dlat = inps.dlat
    dlon = inps.dlon
    fill_value = inps.fill_value
    method = inps.method.lower()

    import yaml
    with open(config_fp, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    ref_fp = config['Path of Input']['Super Reference']
    ref_fp_fn = ref_fp.split('/')[-1]
    if 'Track' in ref_fp_fn and 'Frame' in ref_fp_fn:
        ref_date = ref_fp_fn.split('_')[6][:8]
    else:
        ref_date = ref_fp_fn.split('_')[12][:8]
    InSARRangeLooks = config['Multilook Parameters']['InSAR Range Looks']
    InSARAzimuthLooks = config['Multilook Parameters']['InSAR Azimuth Looks']
    output_fp = config['Path of Output']['Output Dict']

    looks = f"{InSARRangeLooks}rlks_{InSARAzimuthLooks}alks"
    if lat_fp is None or lon_fp is None or hgt_fp is None:
        lat_fp = os.path.join(output_fp, 'dates',ref_date,f'lat_{looks}.flt')
        lon_fp = os.path.join(output_fp, 'dates',ref_date,f'lon_{looks}.flt')
        hgt_fp = os.path.join(output_fp, 'dates',ref_date,f'hgt_{looks}.flt')
    img = isceobj.createImage()
    img.load(lat_fp + '.xml')
    lat_rdr = np.fromfile(lat_fp, dtype=np.float32).reshape(img.length, img.width)
    img.load(lon_fp + '.xml')
    lon_rdr = np.fromfile(lon_fp, dtype=np.float32).reshape(img.length, img.width)
    img.load(hgt_fp + '.xml')
    hgt_rdr = np.fromfile(hgt_fp, dtype=np.float32).reshape(img.length, img.width)

    from xml.etree.ElementTree import ElementTree
    xmlx = ElementTree(file=input_fp + '.xml').getroot()
    tmp = xmlx.find("component[@name='coordinate1']/property[@name='size']/value")
    width = int(tmp.text)
    tmp = xmlx.find("component[@name='coordinate2']/property[@name='size']/value")
    length = int(tmp.text)
    tmp = xmlx.find("property[@name='number_bands']/value")
    nbands_unw = int(tmp.text)
    if nbands_unw == 1:
        unw_data = np.fromfile(input_fp, dtype=np.float32).reshape(length, width)
    elif nbands_unw == 2:
        unw_data = np.fromfile(input_fp, dtype=np.float32).reshape(length,nbands_unw,width)
        unw_data = np.transpose(unw_data, (1, 0, 2))  # (n_bands, n_lines, n_samples)

    dlat = dlat / 3600.0
    dlon = dlon / 3600.0

    if geo_output_fp is None:
        geo_output_fp = input_fp+'.geo'

    print(f"[geocode] Input InSAR file: {input_fp}")
    print(f"[geocode] Output geocoded file: {geo_output_fp}")
    print(f"[geocode] Output grid spacing: dlat={dlat:.8f} deg, dlon={dlon:.8f} deg")
    print(f"[geocode] Output grid boundary: lat [{lat_rdr.min():.6f}, {lat_rdr.max():.6f}], "
          f"lon [{lon_rdr.min():.6f}, {lon_rdr.max():.6f}]")
    print(f"[geocode] Interpolation method: {method}")
    print(f"[geocode] Fill value for missing data: {fill_value}")

    try:
        n_cols, n_rows, hdr_fp = geocode(
            unw_data,
            lat_rdr,
            lon_rdr,
            hgt_rdr,
            geo_output_fp,
            dlat=dlat,
            dlon=dlon,
            method=method,
            fill_value=fill_value
        )
    except Exception as e:
        print(f"[geocode] ERROR: {e}")
        raise