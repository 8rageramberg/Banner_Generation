import requests

# Samll bot script to use for notifications. 

def send_telegram_message(message, bot_token, chat_id):
    url = f""
    payload = {"chat_id": chat_id, "text": message}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Telegram notification sent!")
        else:
            print("Failed to send Telegram notification:", response.text)
    except Exception as e:
        print("Error sending Telegram notification:", e)

# Replace with your actual bot token and chat id
#bot_token = ""
#chat_id = ""
#send_telegram_message("Your script has finished!", bot_token, chat_id)