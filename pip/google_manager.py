import base64
import os.path
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from googleapiclient.errors import HttpError
from param import cwd, files_directory, excel_files_suffix, olivier_sheet_id, calendar_sheet_range, \
    stats_sheet_range

sender = os.environ.get('sender')
recipient = os.environ.get('recipient')

def open_google_service(mode):
    # mode = 'sheets' or 'gmail'
    scopes = ['https://www.googleapis.com/auth/gmail.send',
              'https://www.googleapis.com/auth/spreadsheets']
    cred = None
    if os.path.exists(cwd + 'token.json'):
        cred = Credentials.from_authorized_user_file(cwd + 'token.json', scopes)
    if not cred or not cred.valid:
        flow = InstalledAppFlow.from_client_secrets_file(cwd + 'credentials.json', scopes)
        cred = flow.run_local_server(port=0)
        with open(cwd + 'token.json', 'w') as token:
            token.write(cred.to_json())
    return build(mode, 'v4' if mode == 'sheets' else 'v1', credentials=cred)


def create_message_with_attachment(email_from, email_to, subject, message_text, filenames=None):
    message = MIMEMultipart()
    message['to'] = email_to
    message['from'] = email_from
    message['subject'] = subject

    msg = MIMEText(message_text)
    message.attach(msg)

    for file in filenames:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(open(file, 'rb').read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment', filename=file.split('/')[-1])
        message.attach(part)

    return {'raw': base64.urlsafe_b64encode(message.as_string().encode()).decode()}


def send_message(service, user_id, message):
    try:
        service.users().messages().send(userId=user_id, body=message).execute()
    except HttpError:
        print('HTTP ERROR SENDING MESSAGE')
        return False
    return True


def send_result(file_id, body, attach_files=False):
    msg = create_message_with_attachment(sender, recipient, 'Planning - ' + file_id.replace('_', ' '), body,
                                         filenames=([files_directory + file_id + excel_files_suffix] if attach_files else []))
    gmail_service = open_google_service('gmail')
    print('Send message result', file_id, send_message(gmail_service, 'me', msg))


def read_sheet(sheet_id, sheet_range):
    service = open_google_service('sheets')
    result = service.spreadsheets().values().get(spreadsheetId=sheet_id, range=sheet_range).execute()
    return result.get('values', [])


def read_calendar_sheet():
    return read_sheet(olivier_sheet_id, calendar_sheet_range)


def read_stats_sheet():
    return read_sheet(olivier_sheet_id, stats_sheet_range)
