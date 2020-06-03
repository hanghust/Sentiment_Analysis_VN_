#!/usr/bin/env python
# -*- coding: utf-8 -*-
from scrapy import Spider
from scrapy.selector import Selector
from tutorial.items import TutorialItem

class CrawlerSpider(Spider):
    name = "crawler"
    allowed_domains = ["thegioididong.com"]
    start_urls = [
        "https://www.thegioididong.com/dtdd/samsung-galaxy-a50/danh-gia",]

    def parse(self, response):
        questions = Selector(response).xpath('//ul[@class="ratingLst"]/li')

        for question in questions:
            item = TutorialItem()

            item['User'] = question.xpath(
                'div[@class="rh"]/span/text()').extract_first()
            item['Comment'] = question.xpath(
                'div[@class="rc"]/p/i/text()').extract_first()
            item['Time'] = question.xpath(
                'div[@class="ra"]/a[@class="cmtd"]/text()').extract_first()

            yield item
