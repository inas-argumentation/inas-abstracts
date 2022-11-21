import json
import os
import time
import random
from tqdm import tqdm
import editdistance
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.keys import Keys
from auxiliary.load_data import load_DOIs_links_titles_labels

def replace_characters(str):
    return str.replace("’", "'").replace("\u2010", "-").replace("\ufb01", "fi").\
        replace("\n", " ").replace("‘", "'").replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").\
        replace("ú", "u").replace("ñ", "n").replace("\u2423", "").replace("\u2032", "'").replace("ě", "e").replace("č", "c").\
        replace("\u223c", "~").replace("\u2212", "-").replace("na€ ıve", "naive").replace("\u03b2", "beta").replace("\u2002", " ").\
        replace("–", "-").replace("”", "\"").replace("À", "A").replace("±", "+-").replace("•", "-").replace("©", "").replace("ô", "o").\
        replace("°", "").replace("\u03c7", "").replace("\u2265", ">=").replace("\u03b4", "delta").replace("\u2033", "\"").replace("×", " ").\
        replace("Á", "A").replace("–", "-").replace("—", "-").replace("“", "\"").replace("”", "\"").replace("è", "e")

data = load_DOIs_links_titles_labels()
bar = tqdm(total=len(data))

files = os.listdir("data/abstracts")
for key in list(data.keys()):
    if f"{int(data[key]['index']):03d}.txt" in files:
    #if (web := "webofscience" in (link := data[key]["link"])) or "researchgate" in link:
        del data[key]
        bar.update(1)

def download_abstracts_WebOfScience_and_ResearchGate():
    options = Options()
    options.add_argument('--headless')
    browser = webdriver.Firefox()

    for key in list(data.keys()):
        bar.update(1)
        if (web := "webofscience" in (link := data[key]["link"])) or "researchgate" in link:
            browser.get(link)
            time.sleep(4)

            if web:
                abstract = browser.find_element_by_xpath('//div[@class="abstract--instance"]').text
            else:
                abstract = browser.find_element_by_xpath('//div[@class="nova-legacy-e-text nova-legacy-e-text--size-m nova-legacy-e-text--family-sans-serif nova-legacy-e-text--spacing-none nova-legacy-e-text--color-grey-800 research-detail-middle-section__abstract"]').get_attribute("innerText")

            abstract = abstract.replace("\n", " ")
            abstract = replace_characters(abstract)
            if len(abstract) > 0:
                with open(f"data/abstracts/{data[key]['index']:03d}.txt", "w+") as f:
                    f.write(f"{data[key]['title']}\n{abstract}\n{','.join(data[key]['sub-labels'])}")
            del data[key]
            try:
                pass
            except:
                try:
                    os.remove(f"data/abstracts/{data[key]['index']:03d}.txt")
                except:
                    pass

    bar.close()

def add_remaining_abstracts_by_hand():
    total = len(data)
    for idx, key in enumerate(list(data.keys())):
        title, link = data[key]["title"], data[key]["link"]
        print(f"\nTitle ({idx+1}/{total}): " + title + "\nLink: " + link)
        abstract = input("Abstract: ")
        abstract = abstract.replace("\n", " ")
        abstract = replace_characters(abstract)
        if len(abstract) == 0:
            print("No abstract entered.")
        else:
            with open(f"data/abstracts/{data[key]['index']:03d}.txt", "w+") as f:
                f.write(f"{data[key]['title']}\n{abstract}\n{','.join(data[key]['sub-labels'])}")

if __name__ == '__main__':
    # Sometimes, the page needs to long to load and thus an abstract can not be downloaded.
    # Therefore, it is often useful to run the scaper multiple times to minimize the number of abstracts that need to be added manually.
    download_abstracts_WebOfScience_and_ResearchGate()

    # Make sure that there are no line breaks in the abstracts you enter manually.
    add_remaining_abstracts_by_hand()
