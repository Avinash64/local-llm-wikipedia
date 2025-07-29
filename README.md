Running a large language model locally has some advantages, specifically privacy and cost. One significant drawback is that it's limited to the information it was trained on. My goal in this example is to show how we can use python and the Wikipedia library to use Wikipedia as a knowledge base for getting current information.

# Step 1: Set up you're locally running LLM

For this example I'm using gemma-3-12 https://huggingface.co/google/gemma-3-12b-it. The simplest way to run it with an openai compatible api is to use lm studio. https://lmstudio.ai/docs/app. Follow to guide to run the model of your choice locally. https://lmstudio.ai/docs/app/basics. 

Once it's running make sure the API server is also running.
![[lmstudo_screenshot.png]]

Note: To connect to the api from a different device on the same network change the ip to your computers local ip address.

# Step 2: Connect using python and the openai library
```python
from openai import OpenAI
import json
import wikipedia
import datetime

question = "Who won the 2025 Belgian Grand Prix and how?"

# Change the IP if you are connecting to a server running on another device
client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
  
completion = client.chat.completions.create(
  model="model-identifier",
  messages=[
    {"role": "system", "content": "Answer the users questions"},
    {"role": "user", "content": question}
  ],
  temperature=0.7,
)

print(completion.choices[0].message.content)
```

Below is the output of asking the model a question it wont know (Who won the 2025 Belgian Grand Prix?) because it requires information from after it was trained.

```
As of today the 2025 Belgian Grand Prix hasn't happened yet! Therefore, there is no winner. It will take place in late June/early July of 2025.


Do you want to know about past winners?
```

# Step 3: Having the LLM determine if it needs additional information to answer the question

We can do this by prompting the model to respond in json format if it knows the answer based on it's training data.

System prompt: IIdentify if you have the knowlege in your training to answer this question. Format the response as JSON like {"user": user message here, "known": True or False.}

```python
determine_known = client.chat.completions.create(

  model="model-identifier",

  messages=[

    {"role": "system", "content": 'Identify if you have the knowlege in your training to answer this question. Format the response as JSON like {"user": user message here, "known": True or False.}'},

    {"role": "user", "content": question}

  ],

  temperature=0.7,

)

result = json.loads(determine_known.choices[0].message.content)
print(result)
```
 
 Output:
`{'user': 'Who won the 2025 Belgian Grand Prix?', 'known': False}`

# Step 4: Prompt the model to return a Wikipedia article it should search for

The Wikipedia library can pull data for an article based on a search term. Using the below system we can get it to output a simple relevant search term to the question.

System Prompt - `Identify a wikipedia search term that can be used to find articles to answer the specific user question in the json input finding knowledge you dont already have. The current date {date}. Return only the search term.`

```python
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
```

Output: `2025 Belgian Grand Prix`

# Step 5: Provide the information from the Wikipedia article in the system prompt and ask the original question.

```python
summary = wikipedia.summary(search_term.choices[0].message.content)
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
```

Output:

```Oscar Piastri of McLaren won the 2025 Belgian Grand Prix. He initially started second in the sprint race, losing to Max Verstappen, but then overtook his teammate Lando Norris during the main race (which was delayed by rain) and held the lead to win.```


# Conclusion

The Wikipedia library for python can be a free and easy way to give locally running LLMs access to updated information. 
