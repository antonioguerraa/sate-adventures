#!/usr/bin/env python3
import json
import os
import requests
import time
from typing import Dict, Optional
from urllib.parse import urlparse, parse_qs

# Load airport data
with open('/workspaces/sate-adventures/code/flight-search-engine/data/airport-images.json', 'r') as f:
    airport_data = json.load(f)

with open('/workspaces/sate-adventures/code/flight-search-engine/data/ryanair-airports.json', 'r') as f:
    ryanair_data = json.load(f)

# Create a mapping of airport codes to city/country
airport_info = {}
for airport in ryanair_data['airports']:
    airport_info[airport['code']] = {
        'city': airport['city'],
        'country': airport['country']
    }

# Directory for images
IMAGE_DIR = '/workspaces/sate-adventures/code/flight-search-engine/public/airport-images'

# Curated high-quality images for major airports
CURATED_IMAGES = {
    'MAD': 'https://images.unsplash.com/photo-1539037116277-4db20889f2d4?w=800&h=600&fit=crop',  # Madrid - Plaza Mayor
    'BCN': 'https://images.unsplash.com/photo-1583422409516-2895a77efded?w=800&h=600&fit=crop',  # Barcelona - Sagrada Familia
    'SVQ': 'https://images.unsplash.com/photo-1559386081-325882507af7?w=800&h=600&fit=crop',  # Seville - Plaza de España
    'VLC': 'https://images.unsplash.com/photo-1596402184320-417e7178b2cd?w=800&h=600&fit=crop',  # Valencia - City of Arts
    'AGP': 'https://images.unsplash.com/photo-1578495396609-2c84c1863f2e?w=800&h=600&fit=crop',  # Malaga - Beach
    'PMI': 'https://images.unsplash.com/photo-1558642084-fd07fae5282e?w=800&h=600&fit=crop',  # Palma - Cathedral
    'IBZ': 'https://images.unsplash.com/photo-1476297820623-03984cf5cdbb?w=800&h=600&fit=crop',  # Ibiza - Beach
    'DUB': 'https://images.unsplash.com/photo-1565788018453-f107fee1dc3f?w=800&h=600&fit=crop',  # Dublin - Temple Bar
    'STN': 'https://images.unsplash.com/photo-1513635269975-59663e0ac1ad?w=800&h=600&fit=crop',  # London - Tower Bridge
    'LGW': 'https://images.unsplash.com/photo-1513635269975-59663e0ac1ad?w=800&h=600&fit=crop',  # London - Tower Bridge
    'LTN': 'https://images.unsplash.com/photo-1513635269975-59663e0ac1ad?w=800&h=600&fit=crop',  # London - Tower Bridge
    'MAN': 'https://images.unsplash.com/photo-1515586838455-8f8f940d6853?w=800&h=600&fit=crop',  # Manchester
    'EDI': 'https://images.unsplash.com/photo-1518709779341-56cf4535e94b?w=800&h=600&fit=crop',  # Edinburgh - Castle
    'BER': 'https://images.unsplash.com/photo-1560969184-10fe8719e047?w=800&h=600&fit=crop',  # Berlin - Brandenburg Gate
    'MUC': 'https://images.unsplash.com/photo-1595867818082-083862f3d630?w=800&h=600&fit=crop',  # Munich - Marienplatz
    'FRA': 'https://images.unsplash.com/photo-1577693373683-61376b2792bc?w=800&h=600&fit=crop',  # Frankfurt - Skyline
    'AMS': 'https://images.unsplash.com/photo-1534351590666-13e3e96b5017?w=800&h=600&fit=crop',  # Amsterdam - Canals
    'BRU': 'https://images.unsplash.com/photo-1491557345352-5929e343eb89?w=800&h=600&fit=crop',  # Brussels - Grand Place
    'CDG': 'https://images.unsplash.com/photo-1502602898657-3e91760cbb34?w=800&h=600&fit=crop',  # Paris - Eiffel Tower
    'BVA': 'https://images.unsplash.com/photo-1502602898657-3e91760cbb34?w=800&h=600&fit=crop',  # Paris - Eiffel Tower
    'FCO': 'https://images.unsplash.com/photo-1552832230-c0197dd311b5?w=800&h=600&fit=crop',  # Rome - Colosseum
    'CIA': 'https://images.unsplash.com/photo-1552832230-c0197dd311b5?w=800&h=600&fit=crop',  # Rome - Colosseum
    'MXP': 'https://images.unsplash.com/photo-1520440229-6469a96ac815?w=800&h=600&fit=crop',  # Milan - Duomo
    'BGY': 'https://images.unsplash.com/photo-1520440229-6469a96ac815?w=800&h=600&fit=crop',  # Milan - Duomo
    'VCE': 'https://images.unsplash.com/photo-1523906834658-6e24ef2386f9?w=800&h=600&fit=crop',  # Venice - Canals
    'NAP': 'https://images.unsplash.com/photo-1534308983496-4fabb1a015ee?w=800&h=600&fit=crop',  # Naples - Vesuvius
    'ATH': 'https://images.unsplash.com/photo-1555993539-1732b0258235?w=800&h=600&fit=crop',  # Athens - Acropolis
    'LIS': 'https://images.unsplash.com/photo-1585208798174-6cedd86e019a?w=800&h=600&fit=crop',  # Lisbon - Tram
    'OPO': 'https://images.unsplash.com/photo-1555881400-74d7acaacd8b?w=800&h=600&fit=crop',  # Porto - Riverside
    'VIE': 'https://images.unsplash.com/photo-1516550893923-42d28e5677af?w=800&h=600&fit=crop',  # Vienna - Schönbrunn
    'PRG': 'https://images.unsplash.com/photo-1541849546-216549ae216d?w=800&h=600&fit=crop',  # Prague - Charles Bridge
    'BUD': 'https://images.unsplash.com/photo-1549877452-9c387954fbc2?w=800&h=600&fit=crop',  # Budapest - Parliament
    'WAW': 'https://images.unsplash.com/photo-1607427293702-036933bbf746?w=800&h=600&fit=crop',  # Warsaw - Old Town
    'KRK': 'https://images.unsplash.com/photo-1606992894253-6e18de2a0e47?w=800&h=600&fit=crop',  # Krakow - Main Square
    'CPH': 'https://images.unsplash.com/photo-1513622470522-26c3c8a854bc?w=800&h=600&fit=crop',  # Copenhagen - Nyhavn
    'ARN': 'https://images.unsplash.com/photo-1509356843151-3e7d96241e11?w=800&h=600&fit=crop',  # Stockholm - Gamla Stan
    'OSL': 'https://images.unsplash.com/photo-1533929736458-ca588d08c8be?w=800&h=600&fit=crop',  # Oslo - Opera House
    'HEL': 'https://images.unsplash.com/photo-1538332576228-eb5b4c4de6f5?w=800&h=600&fit=crop',  # Helsinki - Cathedral
    'ZRH': 'https://images.unsplash.com/photo-1515488764276-beab7607c1e6?w=800&h=600&fit=crop',  # Zurich - Lake
    'GVA': 'https://images.unsplash.com/photo-1535378620166-273708d44e4c?w=800&h=600&fit=crop',  # Geneva - Jet d'Eau
    'NCE': 'https://images.unsplash.com/photo-1491166617655-0723a0999cfc?w=800&h=600&fit=crop',  # Nice - Beach
    'MRS': 'https://images.unsplash.com/photo-1566072845259-d6d601cb8836?w=800&h=600&fit=crop',  # Marseille - Port
    'LPA': 'https://images.unsplash.com/photo-1521716691619-c13b61cff3f0?w=800&h=600&fit=crop',  # Gran Canaria
    'TFS': 'https://images.unsplash.com/photo-1521804906057-1df8fdb718b7?w=800&h=600&fit=crop',  # Tenerife
    'RAK': 'https://images.unsplash.com/photo-1539020140153-e479b8c22e70?w=800&h=600&fit=crop',  # Marrakech
    'TLV': 'https://images.unsplash.com/photo-1552423314-cf29ab68ad73?w=800&h=600&fit=crop',  # Tel Aviv
    'DLM': 'https://images.unsplash.com/photo-1528045274126-4f8e4318dd5e?w=800&h=600&fit=crop',  # Dalaman - Beach
    'AYT': 'https://images.unsplash.com/photo-1616423841125-8307665a0469?w=800&h=600&fit=crop',  # Antalya
    'DBV': 'https://images.unsplash.com/photo-1555990538-1e6e5bc3bebb?w=800&h=600&fit=crop',  # Dubrovnik
    'SPU': 'https://images.unsplash.com/photo-1559583109-3e7968136c99?w=800&h=600&fit=crop',  # Split
    'ZAG': 'https://images.unsplash.com/photo-1605206031982-c75e59878c94?w=800&h=600&fit=crop',  # Zagreb
}

def download_image(url: str, filepath: str) -> bool:
    """Download an image from URL to filepath"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def get_generic_city_image(city: str, country: str) -> str:
    """Get a generic city image from Unsplash"""
    query = f"{city} {country} city landmark"
    return f"https://source.unsplash.com/800x600/?{query}"

def main():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    
    # Track results
    successful = []
    failed = []
    
    # Process each airport
    for airport_code in airport_info.keys():
        print(f"Processing {airport_code}...")
        
        # Check if image already exists
        image_path = os.path.join(IMAGE_DIR, f"{airport_code}.jpg")
        if os.path.exists(image_path):
            print(f"  Image already exists for {airport_code}")
            successful.append(airport_code)
            continue
        
        # Try to get image URL
        image_url = None
        
        # First try curated images
        if airport_code in CURATED_IMAGES:
            image_url = CURATED_IMAGES[airport_code]
            print(f"  Using curated image for {airport_code}")
        # Then try existing mapping
        elif airport_code in airport_data.get('images', {}):
            image_url = airport_data['images'][airport_code]
            print(f"  Using existing mapping for {airport_code}")
        # Finally use generic city image
        else:
            info = airport_info[airport_code]
            image_url = get_generic_city_image(info['city'], info['country'])
            print(f"  Using generic image for {airport_code} ({info['city']}, {info['country']})")
        
        # Download the image
        if download_image(image_url, image_path):
            successful.append(airport_code)
            print(f"  ✓ Downloaded {airport_code}")
        else:
            failed.append(airport_code)
            print(f"  ✗ Failed to download {airport_code}")
        
        # Rate limiting
        time.sleep(0.5)
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Download Summary:")
    print(f"  Successful: {len(successful)} airports")
    print(f"  Failed: {len(failed)} airports")
    
    if failed:
        print(f"\nFailed airports: {', '.join(failed)}")
    
    # Create a mapping file for the application
    mapping = {}
    for airport_code in successful:
        mapping[airport_code] = f"/airport-images/{airport_code}.jpg"
    
    mapping_file = os.path.join(os.path.dirname(IMAGE_DIR), 'airport-images-mapping.json')
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"\nCreated mapping file: {mapping_file}")

if __name__ == "__main__":
    main()