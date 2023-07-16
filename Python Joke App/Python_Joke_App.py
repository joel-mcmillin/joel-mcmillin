#!/usr/bin/env python
# coding: utf-8

# In[3]:


import json
import requests


def get_joke():

    # If user immediately exits, the below message is displayed
    prompt = input("For a joke, enter '1', or to quit, enter '2': ")
    if prompt == '2':
        print("********************************************************************************************")
        print("It's OK - They're not that funny anyway... Thanks for using the Chuck Norris Joke Retriever!")
        
    # This message displays with the first joke retrieved
    while prompt == '1':
        response = requests.get("https://api.chucknorris.io/jokes/random")        
        chuck = response.json()                                                   
        print("Here's your joke: ", chuck['value'])                               
        prompt = input("For another joke, enter '1', or to quit, enter '2': ")    

        if prompt == '2':
            print("*************************************************")
            print("Thanks for using the Chuck Norris Joke Retriever!")
            print("*************************************************")
            return response
    if prompt not in ('1', '2'):    # In case of invalid responses 
        prompt = input("Invalid response - Press Enter to continue")
        get_a_response()



def get_a_response():        
    
    # If user enters a 2 to exit after having pulled 1+ jokes, then this message displays
    prompt = input("For a joke, enter '1', or to quit, enter '2': ")    
    if prompt == '2':                                                   
        print("*************************************************")      
        print("Thanks for using the Chuck Norris Joke Retriever!")      
        print("*************************************************")

    while prompt == '1':
        response = requests.get("https://api.chucknorris.io/jokes/random")
        chuck = response.json()
        print("Here's your joke: ", chuck['value'])
        prompt = input("For another joke, enter '1', or to quit, enter '2': ")

        if prompt == '2':
            print("*************************************************")
            print("Thanks for using the Chuck Norris Joke Retriever!")
            print("*************************************************")

    if prompt not in ('1', '2'):
        prompt = input("Invalid response - Press Enter to continue")
        get_a_response()


def main():
    
    # The welcome message for the joke retrieval app
    print("<<<<< Welcome to the Chuck Norris Joke Retriever! >>>>>")
    get_joke()
    print('')
    print("...Joking like it's 2004: a Chuck Norris Joke Retrieval System...")    
                                                                                  

if __name__ == "__main__":
    main()





# In[ ]:




