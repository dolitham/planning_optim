from pandas import DataFrame, read_csv
from datetime import datetime
import locale
import os.path
from numpy import where
from pandas import Series
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from param import cwd, olivier_sheet_id, calendar_sheet_range, stats_sheet_range, jobs_path

locale.setlocale(locale.LC_TIME, "fr_FR")


def read_param_jobs():
    return read_csv(jobs_path, delimiter=";", index_col=0)


def open_sheet_service():
    scopes = ['https://www.googleapis.com/auth/spreadsheets']
    cred = None
    if os.path.exists(cwd + 'token.json'):
        cred = Credentials.from_authorized_user_file(cwd + 'token.json', scopes)
    if not cred or not cred.valid:
        if cred and cred.expired and cred.refresh_token:
            cred.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(cwd + 'credentials.json', scopes)
            cred = flow.run_local_server(port=0)
        with open(cwd + 'token.json', 'w') as token:
            token.write(cred.to_json())
    return build('sheets', 'v4', credentials=cred)


def read_sheet(sheet_id, sheet_range):
    service = open_sheet_service()
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


def get_holidays_input(sheet_start_month, year):
    *values, abort_consult, abort_echo= read_sheet(olivier_sheet_id, calendar_sheet_range)
    index_abort_consult, index_abort_echo, index_cycle_start = [], [], []

    if len(abort_consult) > 1:
        index_abort_consult = where(Series(abort_consult[1:]).str.lower() == 'x')[0]
    if len(abort_echo) > 1:
        index_abort_echo = where(Series(abort_echo[1:]).str.lower() == 'x')[0]

    return format_calendar_from_ucvet_to_df(values, sheet_start_month, year), index_abort_consult, index_abort_echo


def get_current_stats(people_names):
    result = read_sheet(olivier_sheet_id, stats_sheet_range)
    stats = DataFrame(result).set_index(0)[:-1].transpose()
    stats.columns = ['Name'] + list(stats.columns[1:])
    stats = stats.set_index('Name')[people_names].transpose()
    return stats.replace('', 0).astype(int)
