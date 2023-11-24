import argparse
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pendulum as pdl
import wrf
import xarray as xr

OUTPUT_DIR = Path(f'C:\\Users\\Viktor\\Desktop\\wrf_test\\')
FILL_VALUE = -9999.0


def write_output(name, data, wrf_source, date):
    assert ~np.ma.is_masked(data)
    print(name)
    for index in range(0, len(data) // 24 + 1):
        date_current = date.add(days=index)

        current_file = f'NEMO_{name}_wrfout_d01_{date_current.year}-{date_current.month:02d}-{date_current.day:02d}.nc'

        arrays = {}
        coords = {}

        encoding = {'_FillValue': FILL_VALUE, 'dtype': 'float64'}
        xtime_current = wrf_source['XTIME'].data[index * 24: (index + 1) * 24]
        if len(wrf_source['XLAT'][:].shape) == 3:
            xlat_current = wrf_source['XLAT'][:][index * 24: (index + 1) * 24]
            xlong_current = wrf_source['XLONG'][:][
                            index * 24: (index + 1) * 24
                            ]
        else:
            xlat_current = np.expand_dims(wrf_source['XLAT'][:], 0)
            xlat_current = np.repeat(xlat_current, 24, 0)
            xlong_current = np.expand_dims(wrf_source['XLONG'][:], 0)
            xlong_current = np.repeat(xlong_current, 24, 0)
        data_current = data.data[index * 24: (index + 1) * 24]

        if name == 'SWDOWN_DAYMEAN':
            name_to_write = 'SWDOWN'
            data_current = np.expand_dims(np.mean(data_current, axis=0), 0)
            xtime_current = xtime_current[:1]
            xlat_current = xlat_current[:1]
            xlong_current = xlong_current[:1]
        else:
            name_to_write = name

        if len(xtime_current) == 0:
            xtime_current = last_xtime_current
            data_current = last_data_current
            xlat_current = last_xlat_current
            xlong_current = last_xlong_current

        print(data_current.shape, xtime_current.shape, xlat_current.shape, xlong_current.shape)
        arrays[name_to_write] = xr.Variable(
            ('Time', 'south_north', 'west_east'), data_current, encoding=encoding
        )
        arrays['XLAT'] = xr.Variable(
            ('Time', 'south_north', 'west_east'), xlat_current, encoding=encoding
        )
        arrays['XLONG'] = xr.Variable(
            ('Time', 'south_north', 'west_east'), xlong_current, encoding=encoding
        )
        arrays['XTIME'] = xr.Variable(
            ('Time',), xtime_current, encoding=encoding

        )

        coords['Time'] = xr.Variable(('Time',), xtime_current)
        coords['west_east'] = xr.Variable(('west_east',), xlat_current[0, 0, :])
        coords['south_north'] = xr.Variable(('south_north',), xlat_current[0, :, 0])
        last_data_current = data_current
        last_xtime_current = xtime_current
        last_xlat_current = xlat_current
        last_xlong_current = xlong_current

        ds = xr.Dataset(data_vars=arrays, coords=coords)
        ds.to_netcdf(OUTPUT_DIR / current_file, unlimited_dims=['Time'])


def main():
    parser = argparse.ArgumentParser(description='process wrfout for nemo forcing')
    parser.add_argument('wrf_source', help='path to the wrfout file',
                        default='C:\\Users\\Viktor\\Desktop\\wrf_test\\wrfout_d01_2019-01-01_000000')
    parser.add_argument('wrf_start_date', help='YYYY-MM-DD starting date of wrfout', default='2019-01-01')
    args = parser.parse_args()

    print(f'Read variables in  {args.wrf_source}...')

    wrf_source = nc.Dataset(args.wrf_source)

    data = {}

    data['Q2'] = wrf.getvar(wrf_source, 'Q2', wrf.ALL_TIMES, meta=False)
    data['SWDOWN'] = wrf.getvar(
        wrf_source, 'SWDOWN', wrf.ALL_TIMES, meta=False
    )
    data['SWDOWN_DAYMEAN'] = wrf.getvar(
        wrf_source, 'SWDOWN', wrf.ALL_TIMES, meta=False
    )
    data['GLW'] = wrf.getvar(wrf_source, 'GLW', wrf.ALL_TIMES, meta=False)

    ATP = (
            wrf.getvar(wrf_source, 'RAINNC', wrf.ALL_TIMES, meta=False)[:]
            + wrf.getvar(wrf_source, 'RAINC', wrf.ALL_TIMES, meta=False)[:]
    )
    TP = ATP * 0
    TP[1:] = ATP[1:] - ATP[:-1]

    TP = np.clip(TP, 0, None)
    data['TP'] = TP[:] / 3600.0
    SNOWNC = wrf.getvar(wrf_source, 'SNOWNC', wrf.ALL_TIMES, meta=False)[:]
    delta_snownc = SNOWNC * 0
    delta_snownc[1:] = SNOWNC[1:] - SNOWNC[:-1]

    data['SNOWNC'] = delta_snownc / 3600.0

    if 'U10E' in wrf_source.variables.keys():
        UMET10 = wrf.getvar(wrf_source, 'U10E', wrf.ALL_TIMES, meta=False)
        VMET10 = wrf.getvar(wrf_source, 'V10E', wrf.ALL_TIMES, meta=False)
    else:
        UVMET10 = wrf.getvar(wrf_source, 'uvmet10', wrf.ALL_TIMES, meta=False)
        UMET10 = UVMET10[0]
        VMET10 = UVMET10[1]

    data['U10'] = UMET10
    data['V10'] = VMET10

    if 'T2K' in wrf_source.variables.keys():
        T2 = wrf.getvar(wrf_source, 'T2K', wrf.ALL_TIMES, meta=False)
    else:
        T2 = wrf.getvar(wrf_source, 'T2', wrf.ALL_TIMES, meta=False)

    data['T2'] = T2

    HGT = wrf.getvar(wrf_source, 'HGT', wrf.ALL_TIMES, meta=False)
    PSFC = wrf.getvar(wrf_source, 'PSFC', wrf.ALL_TIMES, meta=False)

    stemp = T2 + 6.5 * HGT / 1000
    MSLP = PSFC * np.exp(9.81 / (287.0 * stemp) * HGT) * 0.01 + (
            6.7 * HGT / 1000.0
    )

    data['MSLP'] = MSLP * 100
    wrf_source.close()
    wrf_source = xr.open_dataset(args.wrf_source, engine='netcdf4')

    for name, data in data.items():
        write_output(name, data, wrf_source, pdl.parse(args.wrf_start_date))
    wrf_source.close()


if __name__ == '__main__':
    main()
