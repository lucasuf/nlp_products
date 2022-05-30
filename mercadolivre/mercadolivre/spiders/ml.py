import scrapy


class MlSpider(scrapy.Spider):
    name = "ml"
    start_urls = [
        f"https://www.mercadolivre.com.br/ofertas?page={page}"
        for page in range(1, 210)
    ]

    def parse(self, response, **kwargs):
        p_class = "promotion-item"
        for promotion_item in response.xpath(
            f'//li[@class="{p_class}" or @class="{p_class} default" or @class="{p_class} avg" or @class="{p_class} min"]'
        ):
            title = promotion_item.xpath(
                './/p[@class="promotion-item__title"]/text()'
            ).get()
            link = promotion_item.xpath("./a/@href").get()
            yield {"title": title, "link": link}
