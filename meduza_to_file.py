# -*- coding: utf-8 -*-

from bs4.element import NavigableString
from bs4 import BeautifulSoup
import requests
c = 0
def parse_url(url):
	headers = {'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
	'accept-encoding': 'gzip, deflate, br',
	'accept-language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7'}
	html = requests.get(url, headers=headers).text
	soup = BeautifulSoup(html, 'html.parser')
	text = ''
	art = soup.find(class_="GeneralMaterial-article")
	for j in art.find_all('p'):
		text += j.get_text().strip() + '\n'
	with open('meduza_data/1.txt', 'a', encoding='utf-8') as f:
		global c
		c += 1
		f.write(text + '\n')
	return

headers = {'accept': 'application/json',
	'accept-encoding': 'gzip, deflate, br',
	'accept-language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7'}
html = requests.get("https://meduza.io/api/w5/search?chrono=news&page=0&per_page=500&locale=ru", headers=headers)
json = html.json()
for i in json['collection']:
	if i[:4] == 'news':
		print(i)
		parse_url("https://meduza.io/"+i)

print(c)
