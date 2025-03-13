def get_personality_data():
    print("Please answer the following questions on a scale from 1 (low) to 10 (high):")
    intuition = input("How much do you rely on intuition for decision-making? ")
    stress = input("How high is your typical stress level? ")
    creativity = input("How would you rate your creative thinking abilities? ")
    analytical = input("How analytical are you in your approach to problems? ")
    emotional = input("How emotionally sensitive are you to your surroundings? ")
    routine = input("How much do you prefer routine and structure in your life? ")
    
    personality = {
        "intuition": int(intuition),
        "stress": int(stress),
        "creativity": int(creativity),
        "analytical": int(analytical),
        "emotional": int(emotional),
        "routine": int(routine)
    }
    
    return personality

def get_personality_profile(personality):
    profile = "Personality Profile:\n\n"
    
    if personality.get('intuition', 0) > personality.get('analytical', 0) + 2:
        profile += "You tend to rely more on intuition than analytical thinking. "
        profile += "Your dreams may contain more symbolic elements that provide intuitive guidance.\n\n"
    elif personality.get('analytical', 0) > personality.get('intuition', 0) + 2:
        profile += "You have a more analytical approach to life than intuitive. "
        profile += "Your dreams may reflect problem-solving processes and logical connections.\n\n"
    else:
        profile += "You have a balanced approach between intuition and analysis. "
        profile += "Your dreams likely contain both symbolic guidance and logical problem-solving elements.\n\n"
    
    if personality.get('stress', 0) > 7:
        profile += "Your high stress levels may manifest in your dreams as anxiety scenarios or recurring stressful themes. "
        profile += "Pay attention to how your dreams process daily stressors.\n\n"
    elif personality.get('stress', 0) < 4:
        profile += "Your relatively low stress levels may allow for more creative or exploratory dreams. "
        profile += "Your dreams might focus less on processing stress and more on creative possibilities.\n\n"
    
    if personality.get('creativity', 0) > 7:
        profile += "With high creativity, your dreams are likely to be vivid and imaginative. "
        profile += "You may benefit from artistic expression of dream content.\n\n"
    
    if personality.get('emotional', 0) > 7:
        profile += "Your high emotional sensitivity means your dreams may strongly reflect your emotional state. "
        profile += "Dream emotions may be particularly significant for you to analyze.\n\n"
    
    if personality.get('routine', 0) > 7:
        profile += "Your preference for routine may make disruptions in dream patterns more meaningful. "
        profile += "Pay attention to dreams that break from your usual patterns."
    elif personality.get('routine', 0) < 4:
        profile += "Your comfort with variety may be reflected in diverse dream scenarios. "
        profile += "Look for underlying themes connecting your varied dream experiences."
    
    return profile

def get_dream_processing_style(personality):
    
    symbolic_score = (personality.get('intuition', 5) + personality.get('creativity', 5)) / 2
    literal_score = (personality.get('analytical', 5) + personality.get('routine', 5)) / 2
    emotional_score = personality.get('emotional', 5)
    
    scores = {
        'symbolic': symbolic_score,
        'literal': literal_score,
        'emotional': emotional_score
    }
    
    sorted_styles = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary_style = sorted_styles[0][0]
    secondary_style = sorted_styles[1][0]
    
    style_descriptions = {
        'symbolic': "You tend to process dreams symbolically, looking for deeper meanings and patterns beyond the literal content.",
        'literal': "You tend to process dreams literally, focusing on concrete elements and connections to real-life events.",
        'emotional': "You tend to process dreams emotionally, paying particular attention to the feelings they evoke."
    }
    
    result = f"Primary dream processing style: {primary_style.capitalize()}\n"
    result += f"{style_descriptions[primary_style]}\n\n"
    result += f"Secondary style: {secondary_style.capitalize()}\n"
    
    if primary_style == 'symbolic':
        result += "\nRecommendation: Keep a symbol dictionary or journal to track recurring symbols in your dreams."
    elif primary_style == 'literal':
        result += "\nRecommendation: Track connections between daily events and dream content to identify patterns."
    else:  
        result += "\nRecommendation: Note the emotions in your dreams and how they relate to your waking emotional state."
    
    return result
