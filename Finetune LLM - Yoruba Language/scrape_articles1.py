import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# Function to scrape individual article details
def scrape_article_details(url):
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract article title
        title_tag = soup.find('h1', class_='bbc-csfh25 e1p3vdyi0')
        title = title_tag.text.strip() if title_tag else 'No title found'

        # Extract date and time
        datetime_tag = soup.find('time', class_='bbc-j3wi5n e1mklfmt0')
        datetime = datetime_tag['datetime'] if datetime_tag else 'No datetime found'

        # Extract article content
        paragraphs = soup.find_all('p')
        content = ' '.join([p.text for p in paragraphs]) if paragraphs else 'No content found'

        return {
            'title': title,
            'datetime': datetime,
            'content': content,
            'link': url
        }
    else:
        return None

# Base URL for the topics pages
base_url = 'https://www.bbc.com/yoruba/topics/c340q0y3p5kt'

# Initialize a list to store article details
all_articles = []

# Loop through pages 1 to 15
for page in range(1, 16):
    # Construct the URL for the current page
    page_url = f'{base_url}?page={page}'
    response = requests.get(page_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all article links
        links = soup.find_all('a', class_='focusIndicatorDisplayBlock bbc-uk8dsi e1d658bg0')

        # Loop through each link and scrape article details
        for link in links:
            href = link.get('href')
            if href:
                # Ensure the link is absolute
                full_link = href if href.startswith('http') else 'https://www.bbc.com' + href
                article_details = scrape_article_details(full_link)
                if article_details:
                    all_articles.append(article_details)

        # Wait for 10 seconds before making the next request
        time.sleep(10)
    else:
        print(f'Failed to retrieve page {page}')

# Convert the list of articles to a DataFrame
df = pd.DataFrame(all_articles)

# Display the DataFrame
print(df)

# Save the DataFrame to a CSV file
df.to_csv('bbc_yoruba_articles.csv', index=False)