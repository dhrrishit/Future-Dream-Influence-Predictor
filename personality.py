def get_personality_data():
    print("Please answer the following questions on a scale from 1 (low) to 10 (high):")
    intuition = input("How much do you rely on intuition for decision-making? ")
    stress = input("How high is your typical stress level? ")
    personality = {
        "intuition": int(intuition),
        "stress": int(stress)
    }
    return personality