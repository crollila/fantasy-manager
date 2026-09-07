"""Cache public NFL headlines as evidence; never auto-convert headlines to rankings."""
from xml.etree import ElementTree
from app.data import Cache
from app.storage import DATA

URL='https://www.espn.com/espn/rss/nfl/news'
def refresh_news():
    try:
        path=Cache().fetch('nfl-news.xml',URL,ttl=1800)
        root=ElementTree.fromstring(path.read_bytes())
        return {'source':URL,'items':[{'title':item.findtext('title'),'url':item.findtext('link'),'published':item.findtext('pubDate'),'description':item.findtext('description')} for item in root.findall('.//item')[:50]],'status':'ok','note':'Evidence only. Confirm canonical player ID, timing and quantitative impact before creating a structured event.'}
    except Exception as exc:
        return {'source':URL,'status':'unavailable','error':str(exc),'items':[]}
