import math
from pathlib import Path


def split_train_val_test(wrf_folder, era_folder, train_size, validation_size, test_size):
    era_files = sorted(find_files(era_folder, "era*"))
    wrf_files = sorted(find_files(wrf_folder, "wrf*"))
    print(len(era_files), len(wrf_files), 'wrf/era file numbers')
    print(wrf_files[0], wrf_files[-1])
    print(era_files[0], era_files[-1])
    era_files, wrf_files = sync_era_wrf_dates(era_files, wrf_files)
    assert len(wrf_files) == len(era_files)

    train_end = math.ceil(len(wrf_files) * train_size)
    val_end = math.ceil(len(wrf_files) * (train_size + validation_size))
    train = [wrf_files[:train_end], era_files[:train_end]]
    val = [wrf_files[train_end:val_end], era_files[train_end:val_end]]
    test = [wrf_files[val_end:], era_files[val_end:]]
    return train, val, test


def sync_era_wrf_dates(era_dates, wrf_dates):
    min_1, max_1 = Path(era_dates[0]).stem.split('_')[-1], Path(era_dates[-1]).stem.split('_')[-1]
    min_2, max_2 = Path(wrf_dates[0]).stem.split('_')[-2], Path(wrf_dates[-1]).stem.split('_')[-2]
    start_date = sorted([min_1, min_2])[1]
    end_date = sorted([max_1, max_2])[0]
    print(f'Using dates from {start_date} to {end_date}')
    return crop_by_dates(era_dates, start_date, end_date), crop_by_dates(wrf_dates, start_date, end_date)


def crop_by_dates(dates, start, end):
    st, en = None, None
    for date in dates:
        if start in date:
            st = dates.index(date)
        if end in date:
            en = dates.index(date)
    if st is None or en is None:
        raise Exception('List does not contain boundary dates!')
    return dates[st:en + 1]


def find_files(directory, pattern):
    import os, fnmatch
    flist = []
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                filename = filename.replace('\\', '/')
                flist.append(filename)
    return flist


if __name__ == '__main__':
    wrf_folder = '/home/wrf_data'
    era_folder = '/home/era_data'

    train, val, test = split_train_val_test(wrf_folder, era_folder, 0.7, 0.1, 0.2)
    print(train)
    print(val)
    print(test)
