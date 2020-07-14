# author - Aayushi Kulshrestha
# contact - aayushi.kulshrestha@mail.mcgill.ca

from selenium import webdriver
import time
import json

browser = webdriver.Chrome()
browser.get('http://www.whosdatedwho.com/')
total_map = {}

def add_to_map(name, list):
    total_map[name] = list

# Traverse through all alphabets
index_alpha = -1
while True:
    index_alpha = index_alpha + 1
    print index_alpha
    try:
        alphabets = browser.find_element_by_xpath("//ul[contains(@class, 'ff-az')]")
        alphabets.find_elements_by_tag_name("li")[index_alpha].click()
        index = 0
        while True:
            try:
                stars = browser.find_elements_by_xpath("//li[contains(@class, 'ff-grid-box') and contains(@class, 'ff-list')]//a[contains(@href,'http://www.whosdatedwho.com/dating')]")
                print len(stars)
                print index
                celeb_list = []
                stars[index].click()
                try:
                    try:
                        browser.find_element_by_xpath("//a[text()='Agree I am at least 18']").click()
                        print 'Alert found!!!!!!!!'
                    except Exception, e:
                        print str(e)
                        print "Alert not found"

                    meta_tag_name = browser.find_element_by_xpath("/html/head/meta[9]")
                    celeb_name = meta_tag_name.get_attribute("content")
                    relation_boxes = browser.find_element_by_xpath(".//*[@id='ff-dating-history-grid']").find_elements_by_class_name('ff-grid-box')
                    for box in relation_boxes:
                        name = box.find_element_by_class_name("ff-title")
                        year = box.find_element_by_xpath(".//h3")
                        if name.text:
                            celeb_item = {}
                            celeb_item['name'] = name.text
                            celeb_item['year'] = year.text
                            celeb_list.append(celeb_item)
                    print celeb_list
                except Exception, e:
                    print str(e)
                    print 'no dating history'
                finally:
                    add_to_map(celeb_name, celeb_list)
                    browser.back()
                    index = index + 1
            except Exception, e:
                print str(e)
                print "Done for all within this alphabet"
                browser.back()
                break
    except Exception, e:
        print str(e)
        break
    time.sleep(1)


with open('whodatedwho.txt', 'w') as file:
     file.write(json.dumps(total_map))
