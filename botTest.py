import requests

def send_telegram_message(message, bot_token, chat_id):
    url = f"https://api.telegram.org/bot7601899143:AAHiJK-ppo0c9yX1zaa-a0PtcPd315QReeM/sendMessage"
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
# bot_token = "7601899143:AAHiJK-ppo0c9yX1zaa-a0PtcPd315QReeM"
# chat_id = "8030867302"
# send_telegram_message("Your script has finished!", bot_token, chat_id)