"""AO3 Scraping

Instructions:
1. Select a fandom and copy exactly how its tag on AO3 is written. Put it between quotes.
(eg. Harry Potter -> 'Harry Potter - JK Rowling')

2. Run this file and input crawl() into the console, putting what you collected in step 1 in the ().
(eg. crawl('Harry Potter - JK Rowling'))

3. Two variables are not scraped by the above function: publish date and the body text of the
fanfiction's first chapter. If you would like it, print crawl_extras() into the console. This will
take significantly longer to complete as it will access the page of each individual fanfiction.

ctrl + c will interrupt any function running in the console. If you must stop, it is recommended you
do this while the console has printed 'Waiting for 5 seconds', since it is not performing any
operation during that time.

Extra details are specified in the function descriptions regarding:
- Changing the name of the output csv
- Selecting specific pages to start/finish scraping from
- Restarting an interrupted operation.
"""

from bs4 import BeautifulSoup
import requests
import pandas as pd
import datetime
import time


def get_url(fandom: str, page: int) -> str:
    """Get the URL of site, given the page number. Sorts the search alphabetically."""
    return 'https://archiveofourown.org/tags/' + format_for_url(fandom) + '/works?commit=\
Sort+and+Filter&page=' + str(page) + '&utf8=%E2%9C%93&work_search%5Bcomplete%5D=&\
work_search%5Bcrossover%5D=&work_search%5Bdate_from%5D=&work_search%5Bdate_to%5D=&\
work_search%5Bexcluded_tag_names%5D=&work_search%5Blanguage_id%5D=&\
work_search%5Bother_tag_names%5D=&work_search%5Bquery%5D=&\
work_search%5Bsort_column%5D=title_to_sort_on&work_search%5Bwords_from%5D=&\
work_search%5Bwords_to%5D='


def format_for_url(text: str) -> str:
    """Make a title fit a URL format.

    >>> format_for_url('Harry Potter - J. K. Rowling')
    'Harry%20Potter%20-%20J*d*%20K*d*%20Rowling'
    """
    return text.replace(' ', '%20').replace('.', '*d*').replace('|', '%7C')


def scrape(url: str) -> pd.DataFrame:
    """Scrape all the necessary data from one page and return a dataframe of it."""
    # Get website data, turn into HTML
    html_text = requests.get(url).text
    soup = BeautifulSoup(html_text, 'lxml')

    # Create list of HTMLs of each work
    works = soup.findAll('li', class_='work')
    result = []

    # Iterate through the HTML of every work
    for work in works:
        # Find the relevant HTMl tags and attributes
        title = format_symbol(work.find('a').text)
        link = 'https://archiveofourown.org' + work.a['href']
        work_id = work.a['href'][7:]
        language = work.find('dd', class_='language').text
        fandom = work.find('h5', class_='fandoms').text[10:-13]

        words = work.find('dd', class_='words').text.replace(',', '')
        words = 0 if words == '' else int(words)
        chapters = work.find('dd', class_='chapters').text

        # Find the h4 header containing the gifts
        gift_section = str(work.find('h4'))
        # If there are three <a> tags (the third one being the reciever of the gift)
        if gift_section.count('<a') != 3:
            gift = None
        else:
            # Get rid of the first three <a> tags
            for _ in range(3):
                gift_section = gift_section[gift_section.find('<a') + 2:]
            # Find the area between '> ... </a>'
            gift = gift_section[gift_section.find('>') + 1: gift_section.find('</a>')]

        # If chapter is not of format 123/456, assume it is a one-shot (1/1). Otherwise, find
        # what is before/after the '/' and split the variable into two
        if chapters.find('/') == -1:
            chapter_total = 1
            chapter_current = 1
        else:
            chapter_total = chapters[chapters.find('/') + 1:]
            if chapter_total == '?':
                chapter_total = None
            else:
                chapter_total = int(chapter_total)
            chapter_current = int(chapters[0:chapters.find('/')])

        # fill_na_0() turns None values into 0 and converts into integers
        comments = fill_na_0(work.find('dd', class_='comments'))
        kudos = fill_na_0(work.find('dd', class_='kudos'))
        bookmarks = fill_na_0(work.find('dd', class_='bookmarks'))
        hits = fill_na_0(work.find('dd', class_='hits'))

        # convert_date() turns the string format date into a datetime.datetime format
        last_update = convert_date(work.find('p', class_='datetime').text)

        rating = work.find('span', class_='rating').text
        category = work.find('span', class_='category').text
        complete = work.find('span', class_='iswip').text
        warnings = work.find('span', class_='warnings').text

        # shorten() iterates through every tag, gets rid of surrounding HTML, and turns it into a
        # string like 'tag1, tag2, tag3'
        author = shorten(work.findAll(rel="author"))
        if author == '':
            author = 'Anonymous'

        characters = shorten(work.findAll(class_='characters'))
        relationships = shorten(work.findAll(class_='relationships'))
        freeforms = shorten(work.findAll(class_='freeforms'))
        # format_period() adds a space after periods since paragraphs are deleted and words might
        # be stuck together.
        summary = work.find('blockquote')
        if summary is not None:
            summary = format_period(format_symbol(summary.text[1:-1]))

        series = work.find(class_='series')
        if series is not None:
            series = series.text.replace('\n', '')
            while '  ' in series:
                series = series.replace('  ', ' ')
            series_part = int(series[series.find('Part') + 5:series.find(' of')])
            series = series[series.find('of ') + 3:]
        else:
            series_part = None

        temp_dict = {'title': title,
                     'author': author,
                     'gift': gift,
                     'fandom': fandom,
                     'series': series,
                     'series_part': series_part,
                     'hits': hits,
                     'kudos': kudos,
                     'bookmarks': bookmarks,
                     'comments': comments,
                     'chapter_current': chapter_current,
                     'chapter_total': chapter_total,
                     'words': words,
                     'last_update': last_update,
                     'published': None,
                     'language': language,
                     'summary': summary,
                     'warnings': warnings,
                     'category': category,
                     'rating': rating,
                     'complete': complete,
                     'relationships': relationships,
                     'characters': characters,
                     'freeforms': freeforms,
                     'text': None,
                     'work_id': work_id,
                     'link': link,
                     }
        temp_df = pd.DataFrame(temp_dict, index=[0])
        result.append(temp_df)

    return pd.concat(result)


def format_period(text: str) -> str:
    """Propertly format periods and add spaces when appropriate.

    >>> format_period('Hello I am cool.Woah... wowza')
    'Hello I am cool. Woah... wowza'
    """
    str_so_far = ''
    for i in range(len(text) - 1):
        str_so_far += text[i]
        if text[i] == '.' and text[i + 1] not in {'.', ' ', ']', '}', ')', '"', "'", '>'}:
            str_so_far += ' '
        elif text[i] == ' ' and text[i + 1] == '.':
            str_so_far = str_so_far[:len(str_so_far) - 1]
    str_so_far += text[-1]
    return str_so_far


def format_symbol(text: str) -> str:
    """Properly format extra symbols."""
    text = text.replace('—', '--')
    text = text.replace('\xa0', '')
    text = text.replace('  ', ' ')
    return text


def fill_na_0(tag) -> int:
    """Replace None with 0.

    Preconditions:
    - The input is a bs4 HTML tag
    """
    if tag is None:
        return 0
    else:
        return int(tag.text)


def shorten(iterable: iter) -> str:
    """Shave the HTML off of each AO3 tag, add commas to separate them, and return them as one
    string that will fit in one cell.

    Preconditions:
    - Each item of the iterable is a bs4 HTML tag
    """
    return ', '.join([x.text for x in iterable])


def convert_date(date: str) -> str:
    """Convert a date of format DD MON YYYY to a YY/MM/DD format.

    Preconditions:
    - 10000 >= int(date[7:11]) >= 1950
    - date[3:6] in {'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov',\
'Dec'}
    - 31 >= int(date[0:2]) >= 1

    >>> convert_date('02 Jan 2018')
    '2018/01/02'
    """
    month_converter = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
                       'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
    return date[7:11] + '/' + month_converter[date[3:6]] + '/' + date[0:2]


def get_maxpage(url: str) -> int:
    """Get the largest page number to iterate to."""
    html_text = requests.get(url).text
    soup = BeautifulSoup(html_text, 'lxml')

    # Find the text on the buttons, where the last number in the string is the highest number
    pages = soup.find(title='pagination').text
    start = 0
    end = 0

    # Iterate through the string pages
    for i in range(1, len(pages) + 1):
        # Find end and start indices to splice
        if pages[-i] == ' ' and pages[-i - 1].isnumeric() and pages[-i + 1] == 'N':
            end = -i
        if pages[-i] == ' ' and pages[-i + 1].isnumeric():
            start = -i + 1
        # When both end and start indices have been selected, return the spliced version
        if start != 0 and end != 0:
            return int(pages[start:end])
    return 0


def crawl(fandom: str, filename: str = 'data.csv', start_page: int = 1, end_page: int = None,
          delay: int = 5) -> None:
    """Iterate through every possible page and scrape its content, producing a csv file. The main
    function to use in this scraping file.

    fandom: the exact name of the fandom listed in AO3. (eg. 'Harry Potter - J. K. Rowling')
    filename: the name of the created csv file. Is 'data.csv' by default.
    start_page: the first page to scrape from. Is 1 by default.
    end_page: the last page to scrape from. Is automatically-calculated by default.
    delay: the seconds to wait between accessing a new page. Is 5 by default.

    If this function is interrupted for any reason, it can be restarted by setting start_page as
    the page after the last-scraped page. It is up to you to remember what was the last-scraped page
    (you can check the console).

    It is recommended the delay is not decreased due to ethical scraping guidelines and because the
    AO3 site will block requests made too quickly in succession.
    """
    start_time = datetime.datetime.now()
    print('Scraping fanfictions from ' + fandom + ', starting ' + str(start_time))

    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        df = pd.DataFrame()

    # Find URL of first page
    url = get_url(fandom, 1)
    # Use the URL of the first page to find the end page
    if end_page is None:
        maxpage = get_maxpage(url)
    else:
        maxpage = end_page

    # Iterate from the start page (by default, 1) to the end page
    for i in range(start_page, maxpage + 1):
        # Update URL for each new page
        new_url = get_url(fandom, i)

        # Scrape the page
        print('Scraping page ' + str(i) + ' of ' + str(maxpage))
        df = df.append(scrape(new_url))

        # Save the data to the csv
        df.to_csv(filename, index=False, encoding='utf-8')

        # Wait for five seconds
        if i != maxpage:
            print('Complete. Waiting ' + str(delay) + ' seconds')
            time.sleep(delay)

    end_time = datetime.datetime.now()
    elapsed = end_time - start_time
    print('Complete. Finished scraping at ' + str(end_time))

    # Create a text file with details of the number of fanfictions scraped, the fandom, as well as
    # the start and end times.
    file = open('data_info.txt', 'w+')
    file.write(str(len(df.index)) + ' fanfictions were scraped from the fandom of ' + fandom +
               ' on AO3, starting ' + str(start_time) + ' and finishing ' + str(end_time) +
               ' for a total elapsed time of ' + str(elapsed) + '.')
    file.close()


def crawl_extras(filename: str = 'data.csv', save: str = 'data_extra.csv', delay: int = 5) -> None:
    """Crawl through every single fanfiction to find the publish date and first chapter body.

    Will continue scraping from a csv where this function has not completely finished its work. Note
    that if this is the case, filename must be set to the name of the unfinished csv. For instance,
    if the original csv is called 'data.csv' and crawl_extras has created an unfinished
    'data_extra.csv', you must call crawl_extras('data_extra.csv'), or otherwise crawl_extras will
    restart all progress.

    filename: the name of the csv to retrieve work ids from. Is 'data.csv' by default.
    save: the name of the csv to save results to. Is 'data_extra.csv' by default.
    delay: the seconds to wait between accessing a new page. Is 5 by default.

    It is recommended the delay is not decreased due to ethical scraping guidelines and because the
    AO3 site will block requests made too quickly in succession.
    """
    df = pd.read_csv(filename)

    start_time = datetime.datetime.now()
    print('Scraping publish dates and text bodies, starting ' + str(start_time))

    # For each row in the dataframe, which corresponds to a work
    for i in range(len(df.index)):
        # Find the id of the work
        work_id = df['work_id'][i]
        if pd.isnull(df['text'][i]):
            # Find the corresponding URL of the work and request its HTML
            url = 'https://archiveofourown.org/works/' + str(work_id)
            print('Scraping fanfiction of id ' + str(work_id) + ', ' + df['title'][i] + ' by ' +
                  df['author'][i])
            html_text = requests.get(url).text
            soup = BeautifulSoup(html_text, 'lxml')

            # Retrieve the publish date and body text of the work
            published = convert_date_extra(soup.find('dd', class_='published').text)
            text = soup.find('div', role='article')
            if text is not None:
                text = format_symbol(format_period(text.text[1:-1]))
                for _ in range(2):
                    text = text[text.find(' ') + 1:]
                while '  ' in text:
                    text = text.replace('  ', ' ')

            df.loc[i, 'published'] = published
            df.loc[i, 'text'] = text

            df.to_csv(save, index=False, encoding='utf-8')

            if i != len(df.index) - 1:
                print('Complete. Waiting ' + str(delay) + ' seconds')
                time.sleep(delay)

    end_time = datetime.datetime.now()
    elapsed = end_time - start_time
    print('Complete. Finished scraping extras at ' + str(end_time))

    # Create a text file with details of the number of fanfictions scraped, the fandom, as well as
    # the start and end times.
    file = open('data_extra_info.txt', 'w+')
    file.write(str(len(df.index)) + ' fanfictions were scraped for extras on AO3, \
starting ' + str(start_time) + ' and finishing ' + str(end_time) + ' for a total elapsed time of ' +
               str(elapsed) + '.')
    file.close()


def convert_date_extra(date: str) -> str:
    """Convert a string date of format MM-DD-YYYY to a YYYY/MM/DD string format.

    >>> convert_date_extra('2022-01-05')
    2022/01/05
    """
    return date[0:4] + '/' + date[5:7] + '/' + date[8:10]


def remove_text(filename: str, newfile: bool = True, newname: str = None) -> None:
    """Remove the body text column inside a csv file so that it may be more usable in Excel."""
    df = pd.read_csv(filename)
    df = df.drop('text', 1)

    if newfile:
        if newname is None:
            newname = filename[:-4] + '_remove_text.csv'
    else:
        newname = filename
    df.to_csv(newname, index=False, encoding='utf-8')
