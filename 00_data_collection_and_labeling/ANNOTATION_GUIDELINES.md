# Annotation Guidelines

## Objective

The objective of the task of annotation is to label each datapoint with its corresponding stance. By accurately annotating the stances of the datapoints, we aim to build a comprehensive dataset that can be used for various analyses and natural language processing tasks.

---

## Scope

The dataset provided is a random sample that contains social media (Reddit) posts and comments from 5 subreddits:

- r/AskMiddleEast
- r/IsraelPalestine
- r/Israel
- r/IsrealPaletineWar_23
- r/Palestine

Each datapoint represents a unique perspective or opinion expressed by an individual or group regarding the Israel-Gaza war. Some of the datapoints might seem trivial or incomplete in this regard, since they might be a continuation of some conversation.

---

## Guidelines

In order to annotate correctly, the task is to review each datapoint and assign one of the predefined stance labels that best represents the expressed or implied stance towards the subject matter.

Our work revolves around the Israel-Gaza war.

For our purpose, **“Stance”** refers to the expressed or implied position, viewpoint, or sentiment conveyed by the datapoint towards a particular subject, topic, or entity.

### Example

> “Israel doesnt occupy Gaza, for the past 18 years. They pulled settlers and military out in 2005.”

This statement reflects clear support for Israel as the author of the text seems to provide justifications that prove Israel as not being any sort of culprit, so the stance is **“Supports Israel”**.

---

# Stance Labels

| Label | Meaning                      | Description                                                                                                                                                                                                                                                                                                                                                                                    | Examples                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ----- | ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **P** | **Supports Palestine**       | This label signifies a position that advocates for the rights, interests, or perspectives of the Palestinian people, their statehood, sovereignty, and their aspirations for self-determination, independence, and equality within the region.                                                                                                                                                 | - “Hamas wanted to negotiate the week of the 7th. USA/Israel said no.”<br><br>- “I think most of us know that not all Palestinians support Hamas. However, being able to express it is a different story in Gaza and usually leads to death.”<br><br>- “But it is more than just Israel. It is the imperialist countries, too, including the United States.”                                                                                                                                                                                                                                                                  |
| **I** | **Supports Israel**          | This designation reflects a stance supportive of Israel's interests, security, and rights, including its sovereignty, territorial integrity, and the protection of its citizens. It may also encompass backing for Israel's right to defend itself and ensure its survival in the region.                                                                                                      | - “It’s the ultimate gaslighting to blame Israel for people deciding to be terrorists.”<br><br>- “Israel doesnt occupy Gaza, for the past 18 years. They pulled settlers and military out in 2005.”<br><br>- “Thank you israel. Taking the garbage out now so we dont have to in 5 years.”                                                                                                                                                                                                                                                                                                                                    |
| **N** | **Neutral / Unclear Stance** | This classification denotes an impartial or ambiguous viewpoint that refrains from taking a definitive position in favour of either the Palestinian or Israeli side. It may indicate a lack of sufficient information, understanding, or conviction to align with one perspective over the other, or it could reflect an intentional avoidance of partisanship in complex geopolitical issues. | - “There's a pretty large difference between engineers making nuclear weapons for a country that swears to destroy you in a rain of hellfire vs. random Israeli civilians.”<br><br>- “Not a progressive and IDF stalled negotiations and brought back the fighting.”<br><br>- “All I ask is for my legitimate questions to be legitimately answered, how am I so despicable in this scenario? Words matter, if convincing me is your goal here, it's not working. If it is simply to channel aggressions towards someone who doesn't agree with you, by all means, keep at it. I am open to conversation the moment you are.” |
| **R** | **Irrelevant**               | This label is assigned to datapoints that are not related to the Israel-Gaza war, the Israeli-Palestinian conflict, or discussions about stances on these topics. This includes spam, off-topic comments, personal anecdotes unrelated to the conflict, or any content that does not pertain to the subject matter of our analysis.                                                            | - "What's your favorite pizza topping?"<br><br>- "Check out this unrelated meme lol"<br><br>- "I'm going to the store, anyone need anything?"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |

---

# Points to Consider

- Evaluate each datapoint based on its context, tone, and content to determine the appropriate stance label.
- Take into consideration the nuances and subtleties in the language used to express the stance.
- Avoid making assumptions or introducing bias when assigning stance labels.
- Prioritize filtering out **Irrelevant (R)** content first before assigning stance labels.
- When distinguishing between **Neutral (N)** and other stance labels: Neutral indicates no clear position on either side, while P/I/R are definitive classifications.
