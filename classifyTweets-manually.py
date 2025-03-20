import json
import sys

def classify_tweets_numeric_incremental_skip_existing_other_dedupe(input_file, output_file):
    """
    Classifies tweets from a JSON file using numeric input, saves incrementally, skips existing entries, includes "Other" classification, and removes duplicate tweets from the input.

    Args:
        input_file (str): Path to the input JSON file.
        output_file (str): Path to the output JSON file.
    """

    categories = {
        0: "Other",
        1: "Human Rights and Equality",
        2: "Homosexuality - Equal rights",
        3: "Tougher on illegal immigration",
        4: "Asylum System - More strict",
        5: "Do more to help refugees including children",
        6: "Reduce Spending on Welfare Benefits",
        7: "Higher Benefits for Ill and Disabled",
        8: "Higher Pay for Public Sector Workers",
        9: "More Emergency Service Workers",
        10: "More funds for social care",
        11: "Openness and Transparency - In Favour",
        12: "Higher taxes on banks",
        13: "Reduce central funding for local government",        
        14: "Imported Goods Must Equal UK Standards",
        15: "Energy Prices - More Affordable",
        16: "Stop climate change",
        17: "Public Ownership of Railways",
        18: "State control of bus services",
        19: "Require voters to show photo ID before voting",
        20: "Right to strike",
        21: "Voting age - Reduce to 16",
        22: "Support current and former armed service members",
        23: "Use of UK Military Forces Overseas",
        24: "Openness and Transparency - In Favour"
    }

    try:
        with open(input_file, 'r') as f:
            tweets_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Input file '{input_file}' is not valid JSON.")
        return

    # Deduplicate tweets
    unique_tweets = []
    seen_tweets = set()
    for tweet_data in tweets_data:
        tweet = tweet_data.get('Tweet')
        if tweet and tweet.get('text') and tweet.get('text') not in seen_tweets:
            unique_tweets.append(tweet_data)
            seen_tweets.add(tweet.get('text'))

    classified_tweets = []
    try:
        with open(output_file, 'r') as f:
            classified_tweets = json.load(f)
    except FileNotFoundError:
        pass
    except json.JSONDecodeError:
        print(f"Warning: Output file '{output_file}' exists but is not valid JSON. Overwriting.")

    existing_tweet_texts = set()
    if classified_tweets:
        existing_tweet_texts = {item['TweetMsg'] for item in classified_tweets}

    for tweet_data in unique_tweets:
        tweet = tweet_data.get('Tweet')
        if tweet and tweet.get('text'):
            tweet_text = tweet.get('text')
            tweet_id = tweet.get('id')
            if tweet_text in existing_tweet_texts:
                #print(f"Tweet already classified: {tweet_text}. Skipping.")
                continue

            
            for num, category in categories.items():
                print(f"{num}: {category}")

            while True:
                print(f"Classify the following tweet: {tweet_text}")
                try:
                    response = int(input("Enter the category number: "))
                    if response in categories:
                        break
                    else:
                        print("Invalid category number. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number.")

            classified_tweets.append({'TweetID': tweet_id, 'TweetMsg': tweet_text, 'Category': categories[response]})

            with open(output_file, 'w') as f:
                json.dump(classified_tweets, f, indent=4)
            print(f"Classification saved to '{output_file}'.")
        else:
            print("Warning: Found a tweet without a 'Tweet.text'. Skipped.")

    print("Classification complete.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <input_file> <output_file>")
        sys.exit(1)

    input_filename = sys.argv[1]
    output_filename = sys.argv[2]

    classify_tweets_numeric_incremental_skip_existing_other_dedupe(input_filename, output_filename)