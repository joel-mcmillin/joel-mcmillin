#!/usr/bin/env python
# coding: utf-8

# ### API work with Project Gutenberg, OMDB and Twitter

# In[2]:


# Importing libraries

import urllib.request, urllib.parse, urllib.error
import requests
from bs4 import BeautifulSoup
import ssl
import re


# In[3]:


#Checking the SSL certificate:

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


# In[4]:


# Read the HTML from the URL:

top100url = 'https://www.gutenberg.org/browse/scores/top'
response = requests.get(top100url)


# In[5]:


# A function to check the status of the web request:

def status_check(r):
    if r.status_code==200:
        print("Success!")
        return 1
    else:
        print("Request Failed")
        return -1


# In[6]:


status_check(response)


# In[7]:


# Decode the response and pass onto BeautifulSoup for HTML parsing:

contents = response.content.decode(response.encoding)


# In[8]:


soup = BeautifulSoup(contents, 'html.parser')


# In[9]:


# Finding all the HTML href tags and store them in the list of links

link_list = []


# In[10]:


for link in soup.find_all('a'):
    link_list.append(link.get('href'))


# In[11]:


# Printing the first 40 elements - 

link_list[:40]


# In[12]:


# Next, I use a regular expression to find the numeric digits in the links
# These are the file numbers for the top 100 eBooks
# Also initializing the empty list to hold the file numbers over an appropriate range and
# Then, using regex to find numeric digits in the link href string using findall method - 

book_num = [] #Creating our empty list


# In[13]:


for i in range(19, 119):
    link = link_list[i]
    link = link.strip()
    n = re.findall('[0-9]+', link)
    if len(n) == 1:
        book_num.append(int(n[0]))


# In[14]:


print ("\nThe following are file numbers for the top 100 ebooks on Gutenberg:\n"+"-"*70)
print(book_num)


# In[15]:


# What does soup object's text look like?
# Using .text method we can print and show the first 2,000 characters:

print(soup.text[:2000])


# In[16]:


# I create an empty list for our top 100 books
# This can be searched in the extracted text (using regular expressions) from the soup object to find the 
# names of the top 100 eBooks - 

title_list = []


# In[17]:


# Next creating a starting index. It should point to the text Top 100 Ebooks yesterday. 
# This uses the splitlines method of soup.text - 

start_idx = soup.text.splitlines().index('Top 100 EBooks yesterday')


# In[18]:


# Looping 1-100 to add the strings of the next 100 lines to this list, also using splitlines method - 

for i in range(100):
    title_list.append(soup.text.splitlines()[start_idx + 2 + i])


# In[19]:


# Next I use a regular expression to extract only text from the name strings and append it to an empty 
# list, using match and span:

titles_list = []
for i in range(100):
    id1,id2=re.match('^[a-zA-Z ]*', title_list[i]).span()
    titles_list.append(title_list[i][id1:id2])


# In[20]:


# Printing out the titles

for t in titles_list:
    print(t)


# ### OMDB API Work - 

# In[22]:


# Import libraries

import urllib.request, urllib.parse, urllib.error
import json
import requests


# In[24]:


# First, creating API Key on the OMDB site which I saved to Jupyter notebook
# Then, opening the JSON file

with open('APIkeys.json') as f:
    keys = json.load(f)
    omdbapi = keys['OMDBapi'] 
    
# (Created a JSON file in Jupyter that had a dictionary which saved OMDBapi with my key)


# In[26]:


# Next, assigning OMDB to variable
# Then, creating the apikey variable w/ last portion of the URL and the key

serviceurl = 'http://www.omdbapi.com/?'
apikey = '&apikey=' + omdbapi            

# (The apikey variable is a combo of the end of the URL and the secret 
# key stored in the omdbapi variable)


# In[27]:


# Writing a utility fuction called print_json to print the movie data from the portal

def print_json(json_data):
    list_keys = ['Title', 'Year', 'Rated', 'Released', 'Runtime', 'Genre', 'Director', 'Writer', 
               'Actors', 'Plot', 'Language', 'Country', 'Awards', 'Ratings', 
               'Metascore', 'imdbRating', 'imdbVotes', 'imdbID']   
    # here are all of the different categories 
    print("-" * 100)     
    # formatting w/ asterisks
    for k in list_keys:
        if k in list(json_data.keys()):
            print(f"{k}: {json_data[k]}")
    print("-" * 100)


# In[28]:


# Writing a utility function to download a poster of the movie 
# based on the info from the JSON dataset and save to local folder, using OS module
# This will save the image as an image file!

def save_poster(json_data):
    import os
    title = json_data['Title']
    poster_url = json_data['Poster']
    poster_file_extension=poster_url.split('.')[-1]
    poster_data = urllib.request.urlopen(poster_url).read()
        
    savelocation=os.getcwd()+'\\'+'Posters'+'\\'
    if not os.path.isdir(savelocation):
        os.mkdir(savelocation)
    
    filename=savelocation+str(title)+'.'+poster_file_extension
    f=open(filename,'wb')
    f.write(poster_data)
    f.close()


# In[29]:


# Writing a utility function called search_movie to search for a 
# movie by name and printing JSON file and saving the poster in local folder
# using Try-Except loop and the serviceurl and apikey variables
# Passing on dictionary w key, t, and the movie name as the corresponding value to the urllib.parse.urlencode() 
# function and adding serviceurl and apikey to the output of the function to construct a full URL. 
# Also setting it so it will check for errors 


def search_movie(title):
    try:
        url = serviceurl + urllib.parse.urlencode({'t': str(title)}) + apikey
        print(f'Retrieving "{title}" - ')
        print(url)
        uh = urllib.request.urlopen(url)
        data = uh.read()
        json_data=json.loads(data)
        
        if json_data['Response'] == 'True':
            print_json(json_data)
            if json_data['Poster'] != 'N/A':
                save_poster(json_data)
        else:
            print("Error: ", json_data['Error'])
    
    except urllib.error.URLError as e:
        print(f"Error: {e.reason}")


# In[30]:


# Testing our search_movie function by entering "Titanic" 

search_movie("Titanic")


# (Successfully pulls up movie data!)


# In[31]:


# Finally, testing search_movie by entering "Random_error":

search_movie("Random_error")

# (Successfully shows movie not found!)


# ### Twitter API Work - Completing a data pull 

# In[34]:


get_ipython().system('pip install python-twitter')


# In[35]:


# Importing the Twitter library and setting api

import twitter
api = twitter.Api(consumer_key = 'XXXXXX',
                  consumer_secret = 'XXXXX',
                  access_token_key = 'XXXXX',
                  access_token_secret = 'XXXXX')

# (I ran w/ the actual tokens but 'XXXXX'd them out for printing...)


# In[36]:


print(api.VerifyCredentials())

# (And we are in!)


# In[37]:


# Confirming no tweets to date
statuses = api.GetUserTimeline(screen_name = 'Giova0303')
print([s.text for s in statuses])


# In[38]:


# Here is my search for Ukraine (50 most recent posts since May 16, 2022)
api.GetSearch(term = 'Ukraine', since = 2022-5-16, count = 10)


# In[ ]:




