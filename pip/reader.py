from pandas import DataFrame, read_csv
from datetime import datetime
import locale
from numpy import where, append
from pandas import Series
from param import jobs_path, default_cycle_start_days
from google_manager import read_calendar_sheet, read_stats_sheet

locale.setlocale(locale.LC_TIME, "fr_FR")


def read_param_jobs():
    return read_csv(jobs_path, delimiter=";", index_col=0)


def format_calendar_from_ucvet_to_df(values, month, year):
    cal = DataFrame(values).replace('i', 0).replace('s', 1).replace('', 2).fillna(2).set_index(0, drop=True).transpose()
    cal["Day"] = cal.iloc[:, 0].astype(int)
    cal = cal.drop(columns=2)

    cal["Month"] = month
    cal["Year"] = year
    cal['Days'] = ''
    for i in cal.index[:-1]:
        if cal.loc[i, 'Day'] > cal.loc[i + 1, 'Day']:
            cal.loc[i + 1:, 'Month'] += 1
            if cal.loc[i + 1, 'Month'] == 13:
                cal.loc[i + 1:, 'Year'] += 1
                cal.loc[i + 1:, 'Month'] -= 12

    for i, row in cal.iterrows():
        cal.loc[i, 'Days'] = datetime(day=row['Day'], month=row['Month'], year=row['Year'])

    return cal.drop(columns=['Day', 'Month', 'Year']).set_index('Days', drop=True)


def get_calendar_input(sheet_start_month, year):
    *values, abort_consult, abort_echo, cycle_start = read_calendar_sheet()
    index_abort_consult, index_abort_echo = [], []
    holidays_calendar = format_calendar_from_ucvet_to_df(values, sheet_start_month, year)

    if len(abort_consult) > 1:
        index_abort_consult = where(Series(abort_consult[1:]).str.lower() == 'x')[0]
    if len(abort_echo) > 1:
        index_abort_echo = where(Series(abort_echo[1:]).str.lower() == 'x')[0]
    if len(cycle_start) > 1:
        cycle_start = where(Series(cycle_start[1:]).str.lower() == 'x')[0]
    else:
        cycle_start = where(Series(holidays_calendar.index).dt.dayofweek.isin(default_cycle_start_days))[0]
    cycle_len = append((cycle_start[1:] - cycle_start[:-1]), [len(values[0][1:]) - cycle_start[-1]])
    cycles = {a: b for (a, b) in zip(cycle_start, cycle_len)}
    return holidays_calendar, index_abort_consult, index_abort_echo, cycles, len(cycle_start) <= 1


def get_current_stats(people_names):
    stats = read_stats_sheet()
    stats = DataFrame(stats).set_index(0)[:-1].transpose()
    stats.columns = ['Name'] + list(stats.columns[1:])
    stats = stats.set_index('Name')[people_names].transpose()
    return stats.replace('', 0).astype(int)
