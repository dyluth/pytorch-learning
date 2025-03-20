# need to convert /Users/cam/go/src/github.com/dyluth/votes/classifier/approvedResponses.json to csv format: tweet, category
import pandas as pd


# Rishi Sunak has historically voted strongly against the policy: Higher taxes on banks
def modifyColumn(c):
    x = c.split(": ") 
    return x[-1]

with open('/Users/cam/go/src/github.com/dyluth/votes/classifier/approvedResponses-short.json', encoding='utf-8') as inputfile:
    ##df = pd.read_json(inputfile, lines=True, orient="records")

    df = pd.DataFrame.from_dict(inputfile)
    # modify the ApprovedResponse column to the new shape
    response_col = df["ApprovedResponse"]
    new_policy_column = response_col.apply(modifyColumn)
    df['category'] = new_policy_column



df.to_csv('csvfile.csv', encoding='utf-8', index=False)