import calendar
import pandas as pd
import datetime as datetime
import unidecode
import locale
locale.setlocale(locale.LC_TIME, "fr_FR")
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

sheet_id = '1H0QFSi9Tqi5JQV0rLhe6k30C_PBYUxO3AKjLJR31kKA'

def read_sheet(sheet_id, month):
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
    sheet_range = unidecode.unidecode(calendar.month_name[month]).title() + '!A1:AZ7'

    creds = None
    if os.path.exists('pip/token.json'):
        creds = Credentials.from_authorized_user_file('pip/token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('cwd/credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('pip/token.json', 'w') as token:
            token.write(creds.to_json())

    service = build('sheets', 'v4', credentials=creds)

    result = service.spreadsheets().values().get(spreadsheetId=sheet_id, range=sheet_range).execute()
    return result.get('values', [])


def format_calendar_from_ucvet_to_df(values, month, year):
    cal = pd.DataFrame(values).replace('i', 0).replace('s', 0).replace('', 1).fillna(1).set_index(0, drop=True).transpose()
    cal["Day"] = cal.iloc[:, 0].astype(int)
    cal = cal.drop(columns=1)

    next_month_index = list(cal.loc[20:, "Day"][cal.loc[20:, "Day"].astype(int) <= 10].index)
    cal["Month"] = month
    cal["Year"] = year
    cal['Days'] = ''
    cal.loc[next_month_index, "Month"] = (month + 1) % 12
    if month == 12:
        cal.loc[next_month_index, "Year"] = year + 1

    for i, row in cal.iterrows():
        cal.loc[i, 'Days'] = datetime.datetime(day=row['Day'], month=row['Month'], year=row['Year'])

    return cal.drop(columns=['Day', 'Month', 'Year']).set_index('Days', drop=True)


month = 12
year = 2021


def get_holiday_requests(month, year):
    values = read_sheet(sheet_id, month)
    return format_calendar_from_ucvet_to_df(values, month, year)
