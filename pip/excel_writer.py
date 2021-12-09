from param import *
from pandas import DataFrame, ExcelWriter
from numpy import where, concatenate
from datetime import datetime

suffix_time = datetime.now().strftime('%H%M%S')


def make_shifts_info(result):
    people_names = result.columns.to_list()
    job_names = set(concatenate(result.replace('!','').values)) - {'', 'off'}
    weekend_days_map = result.index.str.startswith('Sam') + result.index.str.startswith('Dim')

    shifts_info = DataFrame(columns=people_names, index=job_names).fillna(0)
    if len(job_names) == 0:
        return shifts_info

    for name in people_names:
        nb_shifts = result[name].value_counts()
        for job in set(result[name].value_counts().index) & set(job_names):
            shifts_info.loc[job, name] = nb_shifts[job]
        shifts_info.loc['Jours weekend travaillés', name] = (result[name][weekend_days_map] != '').sum()
        shifts_info.loc['Jours off', name] = (result[name] == '').sum()
        shifts_info.loc['Jours on', name] = (result[name] != '').sum()
    shifts_info.loc['Jours forfait', :] = shifts_info.loc['Hospit', :] + shifts_info.loc['Consult', :] \
                                          + shifts_info.loc['Hospit+E', :]
    shifts_info.loc['equiv Jours echo', :] = (pay['Echo'] * shifts_info.loc['Echo', :]
                                              + pay['Hospit+E'] * shifts_info.loc['Hospit+E', :])
    shifts_info.loc['Hospit', :] = shifts_info.loc['Hospit+E', :] + shifts_info.loc['Hospit', :]
    shifts_info.drop(labels='Hospit+E', axis="rows", inplace=True)
    for index, row in shifts_info.iterrows():
        if index == 'equiv Jours echo':
            shifts_info.loc[index, :] = shifts_info.loc[index, :].map('€{:,.2f}'.format)
        else:
            shifts_info.loc[index, :] = shifts_info.loc[index, :].astype(int).astype(str)
    return shifts_info

def write_excel(planning, result):
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
        df = df.style.apply(highlight, axis=None)
        df.to_excel(writer, engine="openpyxl", sheet_name=sheet_name)


    with ExcelWriter(cwd + suffix_time + '_UCVet.xlsx') as my_writer:
        export_as_excel(planning, my_writer, 'Clinic', with_dates=True)
        export_as_excel(result, my_writer, 'People', with_dates=True)
        export_as_excel(make_shifts_info(result), my_writer, 'Info', with_dates=False)
