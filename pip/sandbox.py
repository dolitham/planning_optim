import pandas as pd
from param import olivier_sheet_id
from google_manager import open_google_service
pd.set_option("max_columns", 20)
pd.set_option("display.width", 200)


def add_sheets(service, g_sheet_id, sheet_name):
    try:
        response = service.spreadsheets().batchUpdate(
            spreadsheetId=g_sheet_id,
            body={'requests': [{'addSheet': {'properties': {'title': sheet_name}}}]}
        ).execute()
        return response
    except Exception as e:
        print("FAILED CREATING SHEET", sheet_name, e)


add_sheets(open_google_service('sheets'), olivier_sheet_id, 'Sandbox')
