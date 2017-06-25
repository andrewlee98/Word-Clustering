import urllib
import urllib.request
import re
from bs4 import BeautifulSoup
import pandas as pd

test = 'http://www.tripadvisor.com/Attraction_Review-g150800-d153711-Reviews-National_Museum_of_Anthropology_Museo_Nacional_de_Antropologia-Mexico_City_Central.html#REVIEWS'
page = urllib.request.urlopen(test)

# you can quickly parse html by using the beautiful soup library
soup=BeautifulSoup(page.read(), 'html5lib')
#print soup.prettify()

pTags = soup.findAll(re.compile('p'))
print(len(pTags))

print([tag.name for tag in pTags])

# check all the tags conditionally:
#for tag in pTags:
#    print tag.name

# if you examine these individually you'll see that input, option and script tags are useless
#for tag in pTags:
#    if tag.name == 'input': #change me to 'option' or 'script'
#        print tag

# this is a smarter way to get the reviews:
myreviews = []

for tag in pTags:
    if tag.name == 'p':
        print(tag)
        myreviews.append(tag)

print("let's check the reviews:")
print(myreviews)
print(len(myreviews))


# use the review positions to calculate the position of the other information
mypositions = []
for position, tag in enumerate(pTags):
    if tag.name == 'p':
        print("this is the position:")
        print(position)
        mypositions.append(position)

print("lets check positions:")
print(mypositions)
print(len(mypositions))

# lets see if this works:
mydates = []
print("this gives the date of the review:")
i = 0
for position in mypositions:
    print(pTags[position-2]) # change me to find new info
    mydates.append(pTags[position-2])

print("this gives the star rating of the review:")
i = 0
mystars = []
for position in mypositions:
    print (pTags[position-3]) # change me to find new info
    mystars.append(pTags[position-3])

print("this gives the title of the review:")
i = 0
mytitles = []
for position in mypositions:
    print(pTags[position-4]) # change me to find new info
    mytitles.append(pTags[position-4])

#print "you can check everything:"
#print myreviews
#print mydates
#print mystars
#print mytitles
