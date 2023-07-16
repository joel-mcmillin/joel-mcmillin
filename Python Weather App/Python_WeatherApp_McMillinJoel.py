#!/usr/bin/env python
# coding: utf-8

# ### Weather App Using Python

# In[1]:


import requests

# Defining a function to check weather by zipcode or city
# using data from API call to Openweathermap.org
def get_weather(zipcode=None, city=None):   

    base_url = "http://api.openweathermap.org/data/2.5/weather"
    apiid = "384a695be127cb44cd66042512c4cdf5"  
    # To allow a user to enter their own API key, an input prompt could be used in 
    # and then set up if/else statements to catch errors 
    
    if zipcode != None:
        base_url += "?zip=" + str(zipcode) + ",us&units=imperial"   
        # This add-on for data in Imperial format (F)

    elif city != None:
        base_url += "?q=" + str(city) + ",us&units=imperial"

    base_url += "&appid=" + apiid       
    # This gets our final web address that then returns weather forecast
    
    response = requests.get(base_url)
    return response

def show_weather(response):

    if response.status_code == 200:             
        # If 200 status code, then will provide weather output printout
        
        weather = response.json()
        print("Successfully Connected!")
        print(f"""{weather['name']} Current Conditions: 
        Current Temp: {weather['main']['temp']}F. 
        Low Temp today will be {weather['main']['temp_min']}F.
        High Temp today will be {weather['main']['temp_max']}F.
        Humidity: {weather['main']['humidity']}%. 
        Air Pressure: {weather['main']['pressure']}hPa.
        Cloud Cover: {weather['weather'][0]['description']} with winds of {weather['wind']['speed']} MPH.
        Visibility: {weather['visibility']} feet.       
        """)

    else:                      
        # Otherwise, prints the error code 
        print("Error - Try Again Error Code: ", response.status_code, "- Please check input!")



def main():
    print("Welcome to the Weather Getter App! You can find current local weather conditions by zipcode or city name!")

    while True:
    # This will allow the user to select whether they search by zipcode or city, or to exit the app
        prompt = int(input("To search for weather by zipcode, press '1'; by city, press '2'; or '0' to quit: "))

        if prompt == 1:
            try:
                # Zipcode prompt
                prompt = int(input("Enter a zipcode for local weather: "))
                response = get_weather(prompt, None)
                show_weather(response)
            except Exception as e:
                print("Error: ", e)

        elif prompt == 2:
            try:
                # City prompt
                prompt = input("Enter a city name for local weather: ")
                response = get_weather(None, prompt)
                show_weather(response)
            except Exception as e:
                print("Error: ", e)

        elif prompt == 0:
            # Exit prompt
            print("Thank you for using the Weather Getter App!")
            break

        else:
            # Invalid entry - Re-do entry
            print("Invalid entry - Please enter a valid option!")
            
if __name__ == "__main__":
    main()

