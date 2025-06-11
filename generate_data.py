import csv
import random
from datetime import datetime, timedelta

def generate_messaging_data(num_records=1000):
    """
    Generates synthetic data for a real-time messaging system.

    Features:
    - Active users: number
    - New users: number
    - message rate: number (messages per second)
    - media sharing: percentage
    - spam ratio: percentage
    - user sentiment: positive, neutral, negative
    - server load: percentage
    - Time of day: Morning, afternoon, evening, night
    - Latency: Time(in ms)
    - Bandwidth usage: mbps
    - Traffic state: High, medium, low (This will be our target variable initially)
    """

    data = []
    start_time = datetime(2023, 1, 1, 0, 0, 0) # Arbitrary start time

    for i in range(num_records):
        record_time = start_time + timedelta(minutes=i*10) # Simulate data every 10 minutes

        # Determine time of day
        hour = record_time.hour
        time_of_day = ""
        if 6 <= hour < 12:
            time_of_day = "Morning"
        elif 12 <= hour < 18:
            time_of_day = "Afternoon"
        elif 18 <= hour < 22:
            time_of_day = "Evening"
        else:
            time_of_day = "Night"

        # Simulate fluctuating values based on time of day
        active_users = random.randint(500, 5000)
        new_users = random.randint(10, 200)
        message_rate = random.randint(100, 1000) # Messages per second
        media_sharing = round(random.uniform(0.05, 0.30), 2) # 5-30%
        spam_ratio = round(random.uniform(0.01, 0.10), 2) # 1-10%
        user_sentiment = random.choice(["positive", "neutral", "negative"])
        server_load = random.randint(20, 90) # 20-90%
        latency = random.randint(10, 200) # 10-200 ms
        bandwidth_usage = random.randint(50, 500) # 50-500 mbps

        # Adjust values for peak times (e.g., afternoon/evening)
        if time_of_day in ["Afternoon", "Evening"]:
            active_users = random.randint(3000, 15000)
            new_users = random.randint(50, 500)
            message_rate = random.randint(500, 5000)
            server_load = random.randint(60, 98)
            latency = random.randint(50, 500)
            bandwidth_usage = random.randint(200, 1000)

        # Determine traffic state based on a simplified heuristic (this is for initial modeling)
        # In a real scenario, 'traffic state' might be an output of your framework,
        # or a label derived from historical expert decisions.
        traffic_state = "Low"
        if message_rate > 1000 or server_load > 70 or latency > 150:
            traffic_state = "Medium"
        if message_rate > 3000 or server_load > 85 or latency > 300:
            traffic_state = "High"

        data.append([
            active_users,
            new_users,
            message_rate,
            media_sharing,
            spam_ratio,
            user_sentiment,
            server_load,
            time_of_day,
            latency,
            bandwidth_usage,
            traffic_state
        ])

    return data

def save_to_csv(data, filename="messaging_traffic_data.csv"):
    """
    Saves the generated data to a CSV file.
    """
    headers = [
        "Active users", "New users", "Message rate", "Media sharing",
        "Spam ratio", "User sentiment", "Server load", "Time of day",
        "Latency", "Bandwidth usage", "Traffic state"
    ]
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(data)
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    generated_data = generate_messaging_data(num_records=5000) # Generate 5000 records
    save_to_csv(generated_data)