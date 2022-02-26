import pandas as pd
from param import olivier_sheet_id
from reader import open_sheet_service
pd.set_option("max_columns", 10)

service = open_sheet_service()


def add_sheets(g_sheet_id, sheet_name):
    try:
        response = service.spreadsheets().batchUpdate(
            spreadsheetId=g_sheet_id,
            body={'requests': [{'addSheet': {'properties': {'title': sheet_name}}}]}
        ).execute()
        return response
    except Exception as e:
        print("FAILED CREATING SHEET", sheet_name, e)


add_sheets(olivier_sheet_id, 'Sandbox')
