

turn_level_utts = {
    "interesting": {
        "positive": ["Wow that is really interesting.", "That's really interesting!",
                     "Cool! That sounds super interesting."],
        "negative": ["That's not very interesting.", "That's really boring.", "That was a really boring response."]
    },
    "engaging": {
        "positive": ["Wow! That's really cool!", "Tell me more!",
                     "I'm really interested in learning more about this."],
        "negative": ["Let's change the topic.", "I don't really care. That's pretty boring.",
                     "I want to talk about something else."]
    },
    "specific": {
        "positive": ["That's good to know. Cool!", "I see, that's interesting.", "That's a good point."],
        "negative": ["That's a very generic response.", "Not really relevant here.",
                     "That's not really relevant here."]
    },
    "relevant": {
        "positive": [],
        "negative": ["That's not even related to what I said.", "Don't change the topic!",
                     "Why are you changing the topic?"]
    },
    "correct": {
        "positive": [],
        "negative": ["You're not understanding me!", "I am so confused right now!",
                     "I don't understand what you're saying."]
    },
    "semantically appropriate": {
        "positive": ["That makes sense!", "You have a good point."],
        "negative": ["That makes no sense!"]
    },
    "understandable": {
        "positive": ["That makes sense!", "You have a good point."],
        "negative": ["I don't understand at all!", "I'm so confused!", "That makes no sense!",
                     "What does that even mean?"]
    },
    "fluent": {
        "positive": ["That makes sense!", "You have a good point."],
        "negative": ["Is that real English?", "I'm so confused right now!", "That makes no sense!"]
    },
}

dialog_level_utts = {
    "coherent": {
        "positive": [],
        "negative": ["You're making no sense at all.", "You're changing the topic so much!",
                     "You are so confusing."]
    },
    "error recovery": {
        "positive": [],
        "negative": ["I am so confused right now.", "You're really confusing.",
                     "I don't understand what you're saying."]
    },
    "consistent": {
        "positive": [],
        "negative": ["That's not what you said earlier!", "Stop contradicting yourself!"],
    },
    "diverse": {
        "positive": [],
        "negative": ["Stop saying the same thing repeatedly.", "Why are you repeating yourself?",
                     "Stop repeating yourself!"]
    },
    "depth": {
        "positive": [],
        "negative": ["Stop changing the topic so much.", "Don't change the topic!"],
    },
    "likeable": {
        "positive": ["I like you!", "You're super polite and fun to talk to", "Great talking to you."],
        "negative": ["You're not very nice.", "You're not very fun to talk to.", "I don't like you."]
    },
    "understand": {
        "positive": [],
        "negative": ["You're not understanding me!", "What are you trying to say?",
                     "I don't understand what you're saying."]
    },
    "flexible": {
        "positive": ["You're very easy to talk to!", "Wow you can talk about a lot of things!"],
        "negative": ["I don't want to talk about that!", "Do you know how to talk about something else?"],
    },
    "informative": {
        "positive": ["Thanks for all the information!", "Wow that's a lot of information.",
                     "You know a lot of facts!"],
        "negative": ["You're really boring.", "You don't really know much."],
    },
    "inquisitive": {
        "positive": ["You ask a lot of questions!", "That's a lot of questions!"],
        "negative": ["You don't ask many questions.", "You don't seem interested."],
    },
}

all_positive, all_negative = set(), set()
# for feature, texts in dialog_level_utts.items():
#     all_positive.update(texts['positive'])
#     all_negative.update(texts['negative'])

for feature, texts in turn_level_utts.items():
    all_positive.update(texts['positive'])
    all_negative.update(texts['negative'])

# all_positive = [(x, 1) for x in all_positive]
# all_negative = [(x, -1) for x in all_negative]