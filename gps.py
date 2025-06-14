import requests
import smtplib
from email.mime.text import MIMEText

def get_location():
    try:
        res = requests.get('https://ipinfo.io/json')
        data = res.json()
        loc = data.get('loc')  # latitude,longitude
        city = data.get('city')
        region = data.get('region')
        country = data.get('country')

        return f"📍 Emergency Detected!\nLocation: {city}, {region}, {country}\nCoordinates: {loc}"
    except:
        return "Location not available."

def send_email(subject, body, to_email):
    from_email = "neha.pandit.it.2021@tint.edu.in"
    app_password = "ivui mqzf hhoe liia"  # 16-char Gmail app password

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    try:
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(from_email, app_password)
        server.send_message(msg)
        server.quit()
        print("✅ Email sent successfully!")
    except Exception as e:
        print("❌ Failed to send email:", e)

# -------- MAIN --------
location_info = get_location()
send_email("🚨 Emergency Location Alert", location_info, "akashmondal1812@gmail.com")  # Replace with actual email
