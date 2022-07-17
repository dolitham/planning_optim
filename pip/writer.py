import warnings

from param import *
from private_param import cwd
from pandas import DataFrame, ExcelWriter, Series
from numpy import where, concatenate, cumsum
from google_manager import send_result
warnings.simplefilter(action='ignore', category=Warning)

global output

output = ''


def my_print(text, end_char='\n'):
    global output
    if type(text) == list:
        text = ' '.join(str(u) for u in text)
    output += text + end_char
    print(text, end=end_char)


def stats_sum(this_month, so_far):
    return DataFrame({col: (this_month[col] + so_far[col] * ('Current' not in col))
                     .astype(float if col == 'Forfait écho' else int) for col in so_far.columns})


def calculate_stats(result):
    people_names = result.columns.to_list()
    job_names = set(concatenate(result.replace('!', '').values)) - {'', 'off'}
    weekend_days_map = result.index.day_of_week >= 5

    stats = DataFrame(columns=people_names, index=job_names).fillna(0)
    if len(job_names) == 0:
        return stats

    result_no_off = result.replace('off', '')
    working = result_no_off != ''
    working_we = working[working.index.day_of_week == 5].reset_index(drop=True) + \
        working[working.index.day_of_week == 6].reset_index(drop=True)

    for name in people_names:
        nb_shifts = result_no_off[name].value_counts()
        for job in set(result_no_off[name].value_counts().index) & set(job_names):
            stats.loc[job, name] = nb_shifts[job]
        stats.loc['Jours WE', name] = (result_no_off[name][weekend_days_map] != '').sum()
        stats.loc['Total avec imagerie', name] = (result_no_off[name] != '').sum()
        stats.loc['Current shift', name] = where(working[::-1][name] == False)[0][0]
        stats.loc['Current WE streak', name] = where(working_we[::-1][name] == False)[0][0]

    stats.loc['Total sans imagerie', :] = stats.loc['Hospit', :] + stats.loc['Consult', :] + stats.loc['Hospit+E', :]
    stats.loc['Forfait écho', :] = (pay['Echo'] * stats.loc['Echo', :]
                                    + pay['Hospit+E'] * stats.loc['Hospit+E', :])
    stats.loc['Hospit', :] = stats.loc['Hospit+E', :] + stats.loc['Hospit', :]
    stats.loc['Imagerie', :] = stats.loc['Echo', :] + stats.loc['Endo', :]
    stats.drop(labels='Hospit+E', axis="rows", inplace=True)
    stats.drop(labels=['Echo', 'Endo'], axis="rows", inplace=True)
    stats.loc['Forfait écho', 'Marine'] = 0
    stats.loc['Single day WE', :] = (working[result.index.day_of_week == 5].reset_index(drop=True)
                                     > working[result.index.day_of_week == 6].reset_index(drop=True)).sum()

    stats = stats.transpose()
    for col in stats.columns:
        if col == 'Forfait écho':
            stats[col] = stats[col].map(lambda x: round(x, 2))
        else:
            stats[col] = stats[col].astype(int)
    col_order = ['Jours WE', 'Hospit', 'Consult', 'Total sans imagerie', 'Imagerie', 'Total avec imagerie',
                 'Forfait écho', 'Single day WE', 'Current shift', 'Current WE streak']
    return stats[col_order]


def format_planning_to_ucvet(planning):
    dated_planning = planning[planning.columns]
    dated_planning['Nuit'] = ''
    nb_lines = dated_planning.shape[1] + 1

    column = list(Series(dated_planning.index).dt.dayofweek)
    line = [0] + list(cumsum([column[i + 1] < column[i] for i in range(len(column) - 1)]))
    weeks = dict({u: [index for index, val in enumerate(line) if val == u] for u in set(line)})
    dated_planning.index = Series(dated_planning.index).dt.strftime('%d-%b')

    ucvet = DataFrame(index=range(6 * len(set(line))), columns=range(8), dtype=str).fillna('')
    for week_index in weeks.keys():
        this_week_columns = [0] + [column[i] + 1 for i in weeks[week_index]]
        this_week = dated_planning.iloc[weeks[week_index]].reset_index().transpose().reset_index()
        ucvet.iloc[week_index * nb_lines:(week_index + 1) * nb_lines, this_week_columns] = this_week
    ucvet = ucvet.transpose().set_index(0).transpose().set_index('Days')
    ucvet.index.names = ['']
    ucvet.index = Series(ucvet.index).str.replace('Days', '')
    return ucvet


def write_excel(planning, result, stats_this_month, stats_cumul, ucvet, run_id):
    global output
    colors = dict({"Esther": "#65B7FF",
                   "Gael": "#FFC665",
                   "Olivier": "#8DD797",
                   "Marine": "#FF6565",
                   "Estelle": "#FFEF65",
                   "off": "#A0DE98",
                   hospit_solo_name: "#C8E4F4",
                   hospit_echo_name: '#B9CBED',
                   consult_name: "#FDD5E8",
                   echo_name: "#C2C3EF",
                   endo_name: "#F7E0CA",
                   "": "#FFFFFF"})

    def highlight(df):
        weekend_color = 'background-color: {}'.format('#c4c4c4')
        df_attributes = DataFrame('', index=df.index, columns=df.columns)

        if 'Days' in df.columns:
            for prefix in {'Sam', 'Dim'}:
                df_attributes['Days'] = df_attributes['Days'] + \
                                        DataFrame(where(df['Days'].str.startswith(prefix), weekend_color, ''))
        for name, color in colors.items():
            attr = 'background-color: {}'.format(color)
            df_attributes = df_attributes + DataFrame(where(df == name, attr, ""), index=df.index, columns=df.columns)
        return df_attributes

    def export_as_excel(df, writer, sheet_name, with_dates=True):
        if with_dates:
            df.index = df.index.strftime('%a %d %b')
        else:
            df = df.reset_index()
        df = df.style.apply(highlight, axis=None)
        df.to_excel(writer, engine="openpyxl", sheet_name=sheet_name, index=with_dates)

    if len(set(planning.values.flatten())) > 1:
        output += '\n' + 'SUCCESS - writing excel doc'
        with ExcelWriter(cwd + files_directory + run_id + excel_files_suffix) as my_writer:
            export_as_excel(ucvet, my_writer, 'UCVet_' + run_id, with_dates=False)
            export_as_excel(stats_this_month, my_writer, 'Stats_' + run_id, with_dates=False)
            export_as_excel(stats_cumul, my_writer, 'Cumul_Stats_' + run_id, with_dates=False)
            export_as_excel(result, my_writer, 'People_' + run_id, with_dates=True)
            export_as_excel(planning, my_writer, 'Clinic_' + run_id, with_dates=True)


def save_planning(planning, result, stats_so_far, timestamp):
    if len(set(planning['Hospit'])) > 1:
        stats_this_month = calculate_stats(result)
        stats_cumul = stats_sum(this_month=stats_this_month, so_far=stats_so_far)
        ucvet = format_planning_to_ucvet(planning)
        write_excel(planning, result, stats_this_month, stats_cumul, ucvet, timestamp)
        send_result(timestamp, output, attach_files=True)
    else:
        send_result(timestamp, output)


# TODO color first row
# TODO bold totaux + jours we
