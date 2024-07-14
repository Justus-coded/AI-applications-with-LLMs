import requests
from bs4 import BeautifulSoup

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
            'content': content
        }
    else:
        return None

# URL of the main page
main_url = 'https://www.bbc.com/yoruba'

# Send a GET request to the main page
response = requests.get(main_url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the HTML content of the main page
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all article links
    links = soup.find_all('a', class_='focusIndicatorDisplayBlock bbc-uk8dsi e1d658bg0')

    # Initialize a list to store article details
    articles = []

    # Loop through each link and scrape article details
    for link in links:
        href = link.get('href')
        if href:
            # Ensure the link is absolute
            full_link = href if href.startswith('http') else 'https://www.bbc.com' + href
            article_details = scrape_article_details(full_link)
            if article_details:
                articles.append(article_details)

    # Print the scraped article details
    for article in articles:
        print(f"Title: {article['title']}")
        print(f"Date and Time: {article['datetime']}")
        print(f"Content: {article['content']}\n")
else:
    print('Failed to retrieve the main page')