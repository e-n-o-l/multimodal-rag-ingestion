import re, json
import torch
from torch import Tensor

system_prompt = """
You are a precise data parser for automotive classified ads.
Your task is to analyze the text of an Otomoto listing and extract only the specified fields.
Parsing rules:
Respond ONLY with a valid JSON object. Do not add any introductory text, summaries, or markdown tags (do not use ```json).
Clean all numbers by removing spaces, commas, and units (e.g., "km", "PLN", "cm3", "HP"), and convert them into a pure numeric type (int or float).
Translate all extracted values (such as color, fuel_type, gearbox, body_type, etc.) from Polish into English (e.g., "Benzyna" -> "Gasoline", "Manualna" -> "Manual", "Kompakt" -> "Compact", "Czarny" -> "Black").
If a specific field is not present in the text, assign it a null value.
Expected JSON structure:
{
"price": int or null,           // price only as a number, e.g., 45000
"currency": "string" or null,    // e.g., "PLN"
"brand": "string" or null,       // Vehicle brand from the Basic info section
"model": "string" or null,       // Vehicle model from the Basic info section
"color": "string" or null,       // Color translated to English, e.g., "Black"
"production_year": int or null, // Year of production as a number, e.g., 2016
"mileage": int or null,         // Mileage in km as a number, e.g., 70900
"fuel_type": "string" or null,   // Fuel type translated to English, e.g., "Gasoline"
"gearbox": "string" or null,     // Transmission translated to English, e.g., "Manual"
"body_type": "string" or null,   // Body style translated to English, e.g., "Compact"
"engine_capacity": int or null, // Engine capacity in cm3 as a number, e.g., 1368
"engine_power": int or null      // Engine power in HP as a number, e.g., 120
}
"""

def build_prompt(user_content: str) -> list:
    global system_prompt

    return [
        {"role" : "system", "content": system_prompt},
        {"role" : "user", "content": user_content}
    ]

def collate_fn(batch) -> tuple[Tensor, list, list, list] | tuple[None, None, None, None]:
    if batch is None:
        return None, None, None, None

    image = torch.cat([item[0] for item in batch])
    groups = [item[1] for item in batch]
    htmls = [item[2] for item in batch]
    urls = [item[3] for item in batch]

    return image, groups, htmls, urls


def extract_labels(output) -> list | None:
    raw_responses = [out[0]["generated_text"][-1]["content"] for out in output]

    cleaned_response = [re.sub(r"^```(?:json)?\s*", "", response) for response in raw_responses]
    cleaned_response = [re.sub(r"\s*```$", "", response).strip() for response in cleaned_response]

    try:
        return [json.loads(response) for response in cleaned_response]

    except json.JSONDecodeError:
        return None