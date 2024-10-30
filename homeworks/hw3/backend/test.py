import os
import base64
from email.mime.text import MIMEText
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

def create_message(to, subject, body):
    """Create a MIMEText email message."""
    message = MIMEText(body, "plain")  # Change "plain" to "html" for HTML content
    message['to'] = to
    message['from'] = 'tmr880826@gmail.com'  # Replace with your email
    message['subject'] = subject
    return base64.urlsafe_b64encode(message.as_bytes()).decode()

def send_email(service, to, subject, body):
    """Send an email message using the Gmail API."""
    raw_message = create_message(to, subject, body)
    try:
        send_message = service.users().messages().send(userId='me', body={'raw': raw_message}).execute()
        print('Email sent successfully! Message ID: %s' % send_message['id'])
    except Exception as e:
        print('Failed to send email: %s' % e)

def main():
    """Sends an email using OAuth 2.0."""
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('private/email_credentials.json', SCOPES)
            creds = flow.run_local_server(port=8081)
        
        raise Exception("Please provide your own email_credentials.json file")
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    # Build the Gmail service
    service = build('gmail', 'v1', credentials=creds)

    # Get email details from user input
    to = input("Enter recipient email: ")
    subject = input("Enter email subject: ")
    body = input("Enter email body: ")

    # Send the email
    send_email(service, to, subject, body)

if __name__ == '__main__':
    main()
