import re

import pandas as pd

from nlpia.web import requests_get


def simplify_address(address, remove_zip=True, remove_apt=True):
    address = address.lower()
    zipcode = re.compile('[0-9]{4,5}[-]?[0-9]{0,4}$')
    address = zipcode.sub('', address or '')
#     aptnum =  re.compile('(\b#[ ]?|apt|unit|appartment)\s?([A-Z]?[-]?[0-9]{0,6})')
#     address = aptnum.sub('', address or '')
    return address


def geocode_osm(address, polygon=0):
    polygon = int(polygon)
    address = address.replace(' ', '+').replace('\r\n', ',').replace('\r', ',').replace('\n', ',')
    osm_url = 'http://nominatim.openstreetmap.org/search'
    osm_url += '?q={address}&format=json&polygon={polygon}&addressdetails={addressdetails}'.format(
        address=address, polygon=polygon, addressdetails=0)

    print(osm_url)
    resp = requests_get(osm_url, timeout=5)
    print(resp)
    d = resp.json()
    print(d)

    return {
        'lat': d[0].get('lat', pd.np.nan),
        'lon': d[0].get('lon', pd.np.nan),
        }


def encode_get_args(s):
    return s.replace(' ', '+').replace('\r\n', ',').replace('\r', ',').replace('\n', ',')


def geocode_google(address, apikey=None):
    apikey = apikey or 'AIzaSyC--s1-y1xkIxzO7wfIUOeHm8W-ID9fbfM'  # this is a Total Good API key, GET YOUR OWN!
    google_url = 'https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={apikey}'.format(
        address=address, apikey=apikey)
    resp = requests_get(google_url, allow_redirects=True, timeout=5)
    results = resp.json()
    results = results.get('results', {})
    results = [{}] if not len(results) else results
    latlon = results[0].get('geometry', {}).get('location', {})
    return {
        'lat': latlon.get('lat', pd.np.nan),
        'lon': latlon.get('lng', pd.np.nan),
        }
