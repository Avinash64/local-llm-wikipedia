from openai import OpenAI
import json
import wikipedia
import datetime

question = "Who won the 2025 Belgian Grand Prix and how?"

# Point to the local server
client = OpenAI(base_url="http://192.168.1.100:1234/v1", api_key="lm-studio")


determine_known = client.chat.completions.create(
  model="model-identifier",
  messages=[
    {"role": "system", "content": 'Identify if you have the knowlege in your training to answer this question. Format the response as raw JSON like {"user": user message here, "known": True or False.} Dont use ```json to format it/'},
    {"role": "user", "content": question}
  ],
  temperature=0.7,
)
print(determine_known.choices[0].message.content)
result = json.loads(determine_known.choices[0].message.content)
print(result)

if not result['known']:
    # Get the current date
    current_date = datetime.date.today()
    # Format the date as "MM/DD/YYYY"
    formatted_date = current_date.strftime("%m/%d/%Y")
    search_term = client.chat.completions.create(
    model="model-identifier",
    messages=[
        {"role": "system", "content": f"Identify a wikipedia search term that can be used to find articles to answer the specific user question in the json input finding knowledge you dont already have. The current date is {formatted_date}. Return only the search term."},
        {"role": "user", "content": result['user']}
    ],
    temperature=0.7,
    )
    print(search_term.choices[0].message.content)


    summary = wikipedia.summary(search_term.choices[0].message.content)
    print(summary)

    response = client.chat.completions.create(
    model="model-identifier",
    messages=[
        {"role": "system", "content": f"Using the following information answer the users question. {summary}"},
        {"role": "user", "content": question}
    ],
    temperature=0.7,
    )
else: 
    response = client.chat.completions.create(
  model="model-identifier",
  messages=[
    {"role": "system", "content": "Answer the users questions"},
    {"role": "user", "content": question}
  ],
  temperature=0.7,
)
    
print(response.choices[0].message.content)

