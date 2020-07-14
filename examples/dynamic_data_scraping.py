# author - Aayushi Kulshrestha
# contact - aayushi.kulshrestha@mail.mcgill.ca
from selenium import webdriver
import time

browser = webdriver.Chrome()
browser.get('https://stacksity.com/$best')

index = 0
max = 1
posts = []
height = 0
tags = []
while len(posts) < 1000:
	print 'index : ' + str(index)
	browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
	time.sleep(1)
	posts = browser.find_elements_by_xpath("//div[contains(@class, 'item') and contains(@class, 'post')]")
	print len(posts)
	while index < len(posts):
		print '-------------------------new post-------------------------'
		inside = posts[index].find_element_by_class_name("textcon")
		a_tags = inside.find_elements_by_xpath(".//div[1]/p/a")
		print len(a_tags)
		tags.append(a_tags[-1].text)
		index = index + 1
	print "Length of tags-----------"
	print len(tags)
print tags

