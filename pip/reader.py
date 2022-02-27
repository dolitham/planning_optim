from pandas import DataFrame, read_csv
from datetime import datetime
import locale
from numpy import where, append
from pandas import Series
from param import jobs_path, olivier_sheet_id, calendar_sheet_range, stats_sheet_range, default_cycle_start_days
from google_manager import open_google_service
from writer import my_print

locale.setlocale(locale.LC_TIME, "fr_FR")


def read_param_jobs():
    return read_csv(jobs_path, delimiter=";", index_col=0)


def read_sheet(sheet_id, sheet_range):
    service = open_google_service('sheets')
    result = service.spreadsheets().values().get(spreadsheetId=sheet_id, range=sheet_range).execute()
    return result.get('values', [])


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
    *values, abort_consult, abort_echo, cycle_start = read_sheet(olivier_sheet_id, calendar_sheet_range)
    index_abort_consult, index_abort_echo = [], []
    holidays_calendar = format_calendar_from_ucvet_to_df(values, sheet_start_month, year)

    if len(abort_consult) > 1:
        index_abort_consult = where(Series(abort_consult[1:]).str.lower() == 'x')[0]
    if len(abort_echo) > 1:
        index_abort_echo = where(Series(abort_echo[1:]).str.lower() == 'x')[0]
    if len(cycle_start) > 1:
        cycle_type = 'custom'
        cycle_start = where(Series(cycle_start[1:]).str.lower() == 'x')[0]
    else:
        cycle_start = where(Series(holidays_calendar.index).dt.dayofweek.isin(default_cycle_start_days))[0]
        cycle_type = 'auto'
    cycle_len = append((cycle_start[1:] - cycle_start[:-1]), [len(values[0][1:]) - cycle_start[-1]])
    cycles = {a: b for (a, b) in zip(cycle_start, cycle_len)}
    my_print('DEBUT CYCLES HOSPIT & ECHO : ', cycle_type + '\n')
    my_print(list(Series(holidays_calendar.index)[cycles.keys()].dt.strftime('%a %d/%m || ')))
    return holidays_calendar, index_abort_consult, index_abort_echo, cycles


def get_current_stats(people_names):
    result = read_sheet(olivier_sheet_id, stats_sheet_range)
    stats = DataFrame(result).set_index(0)[:-1].transpose()
    stats.columns = ['Name'] + list(stats.columns[1:])
    stats = stats.set_index('Name')[people_names].transpose()
    return stats.replace('', 0).astype(int)
