import requests
from typing import Optional

def get_company_name(ticker: str) -> Optional[str]:
    """
    Retrieves the company name for a given ticker symbol using the SEC's company_tickers.json file.

    Args:
        ticker: The stock ticker symbol.

    Returns:
        The company name if found, otherwise None.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {
        "User-Agent": "joe shmo joesmo@gmail.com"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        for entry in data.values():
            if entry.get('ticker', '').upper() == ticker.upper():
                return entry.get('title', 'Not Found')

    except requests.exceptions.RequestException as e:
        print(f"Error fetching or parsing data: {e}")
        return "Not Found"

    return "Not Found"
