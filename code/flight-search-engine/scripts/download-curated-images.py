#!/usr/bin/env python3
import json
import os
import requests
import time

# Directory for images
IMAGE_DIR = '/workspaces/sate-adventures/code/flight-search-engine/public/airport-images'

# High-quality curated images for Ryanair airports
AIRPORT_IMAGES = {
    # Spain
    'MAD': 'https://images.unsplash.com/photo-1539037116277-4db20889f2d4?w=800&h=600&fit=crop&q=80',
    'BCN': 'https://images.unsplash.com/photo-1583422409516-2895a77efded?w=800&h=600&fit=crop&q=80',
    'SVQ': 'https://images.unsplash.com/photo-1559386081-325882507af7?w=800&h=600&fit=crop&q=80',
    'VLC': 'https://images.unsplash.com/photo-1596402184320-417e7178b2cd?w=800&h=600&fit=crop&q=80',
    'AGP': 'https://images.unsplash.com/photo-1578495396609-2c84c1863f2e?w=800&h=600&fit=crop&q=80',
    'ALC': 'https://images.unsplash.com/photo-1562883642-d1fcbc57284a?w=800&h=600&fit=crop&q=80',
    'PMI': 'https://images.unsplash.com/photo-1558642084-fd07fae5282e?w=800&h=600&fit=crop&q=80',
    'IBZ': 'https://images.unsplash.com/photo-1476297820623-03984cf5cdbb?w=800&h=600&fit=crop&q=80',
    'BIO': 'https://images.unsplash.com/photo-1570698473651-b2de99bae12f?w=800&h=600&fit=crop&q=80',
    'SDR': 'https://images.unsplash.com/photo-1565071559227-20ab25b7685e?w=800&h=600&fit=crop&q=80',
    'MAH': 'https://images.unsplash.com/photo-1570077188670-e3a8d69ac5ff?w=800&h=600&fit=crop&q=80',
    
    # UK & Ireland
    'STN': 'https://images.unsplash.com/photo-1513635269975-59663e0ac1ad?w=800&h=600&fit=crop&q=80',
    'LGW': 'https://images.unsplash.com/photo-1513635269975-59663e0ac1ad?w=800&h=600&fit=crop&q=80',
    'LTN': 'https://images.unsplash.com/photo-1513635269975-59663e0ac1ad?w=800&h=600&fit=crop&q=80',
    'MAN': 'https://images.unsplash.com/photo-1515586838455-8f8f940d6853?w=800&h=600&fit=crop&q=80',
    'EDI': 'https://images.unsplash.com/photo-1518709779341-56cf4535e94b?w=800&h=600&fit=crop&q=80',
    'DUB': 'https://images.unsplash.com/photo-1565788018453-f107fee1dc3f?w=800&h=600&fit=crop&q=80',
    'ORK': 'https://images.unsplash.com/photo-1590846083693-f23fdede3a7e?w=800&h=600&fit=crop&q=80',
    'BRS': 'https://images.unsplash.com/photo-1569764356260-f3f30ddbefb4?w=800&h=600&fit=crop&q=80',
    'LPL': 'https://images.unsplash.com/photo-1560032779-0a61868c6c6e?w=800&h=600&fit=crop&q=80',
    'BHX': 'https://images.unsplash.com/photo-1568844285816-39c21394b9a7?w=800&h=600&fit=crop&q=80',
    
    # Germany
    'BER': 'https://images.unsplash.com/photo-1560969184-10fe8719e047?w=800&h=600&fit=crop&q=80',
    'MUC': 'https://images.unsplash.com/photo-1595867818082-083862f3d630?w=800&h=600&fit=crop&q=80',
    'FRA': 'https://images.unsplash.com/photo-1577693373683-61376b2792bc?w=800&h=600&fit=crop&q=80',
    'CGN': 'https://images.unsplash.com/photo-1604580864964-0462f5d5b1a8?w=800&h=600&fit=crop&q=80',
    'HAM': 'https://images.unsplash.com/photo-1552751117-385aaf5d6c71?w=800&h=600&fit=crop&q=80',
    
    # Netherlands & Belgium
    'AMS': 'https://images.unsplash.com/photo-1534351590666-13e3e96b5017?w=800&h=600&fit=crop&q=80',
    'BRU': 'https://images.unsplash.com/photo-1491557345352-5929e343eb89?w=800&h=600&fit=crop&q=80',
    'CRL': 'https://images.unsplash.com/photo-1491557345352-5929e343eb89?w=800&h=600&fit=crop&q=80',
    'EIN': 'https://images.unsplash.com/photo-1568793237737-438eacf0cfc5?w=800&h=600&fit=crop&q=80',
    
    # France
    'CDG': 'https://images.unsplash.com/photo-1502602898657-3e91760cbb34?w=800&h=600&fit=crop&q=80',
    'BVA': 'https://images.unsplash.com/photo-1502602898657-3e91760cbb34?w=800&h=600&fit=crop&q=80',
    'NCE': 'https://images.unsplash.com/photo-1491166617655-0723a0999cfc?w=800&h=600&fit=crop&q=80',
    'MRS': 'https://images.unsplash.com/photo-1566072845259-d6d601cb8836?w=800&h=600&fit=crop&q=80',
    'BOD': 'https://images.unsplash.com/photo-1569165003085-e8a1b6bd5e1c?w=800&h=600&fit=crop&q=80',
    'TLS': 'https://images.unsplash.com/photo-1627896155938-4cd6a5b1f689?w=800&h=600&fit=crop&q=80',
    'NTE': 'https://images.unsplash.com/photo-1569165003027-8df0f24b9771?w=800&h=600&fit=crop&q=80',
    
    # Italy
    'FCO': 'https://images.unsplash.com/photo-1552832230-c0197dd311b5?w=800&h=600&fit=crop&q=80',
    'CIA': 'https://images.unsplash.com/photo-1552832230-c0197dd311b5?w=800&h=600&fit=crop&q=80',
    'MXP': 'https://images.unsplash.com/photo-1569264651238-c1b8542e6667?w=800&h=600&fit=crop&q=80',
    'BGY': 'https://images.unsplash.com/photo-1569264651238-c1b8542e6667?w=800&h=600&fit=crop&q=80',
    'VCE': 'https://images.unsplash.com/photo-1523906834658-6e24ef2386f9?w=800&h=600&fit=crop&q=80',
    'NAP': 'https://images.unsplash.com/photo-1534308983496-4fabb1a015ee?w=800&h=600&fit=crop&q=80',
    'BLQ': 'https://images.unsplash.com/photo-1557747756-8d3bf1cce501?w=800&h=600&fit=crop&q=80',
    'PSA': 'https://images.unsplash.com/photo-1604580977654-d11f10c36b69?w=800&h=600&fit=crop&q=80',
    'PMO': 'https://images.unsplash.com/photo-1591098595258-3c910c9f4847?w=800&h=600&fit=crop&q=80',
    'CTA': 'https://images.unsplash.com/photo-1596894756027-3e6c368ec9ff?w=800&h=600&fit=crop&q=80',
    'CAG': 'https://images.unsplash.com/photo-1543785734-4b6e564642f8?w=800&h=600&fit=crop&q=80',
    'BRI': 'https://images.unsplash.com/photo-1599316290302-17ba7baa0c98?w=800&h=600&fit=crop&q=80',
    'TRN': 'https://images.unsplash.com/photo-1567525031308-ad3d14e6a447?w=800&h=600&fit=crop&q=80',
    'VRN': 'https://images.unsplash.com/photo-1562178406-4a36a9e77633?w=800&h=600&fit=crop&q=80',
    
    # Greece
    'ATH': 'https://images.unsplash.com/photo-1555993539-1732b0258235?w=800&h=600&fit=crop&q=80',
    'SKG': 'https://images.unsplash.com/photo-1578662996442-48f60103fc4e?w=800&h=600&fit=crop&q=80',
    'CHQ': 'https://images.unsplash.com/photo-1503152394-c571994fd383?w=800&h=600&fit=crop&q=80',
    'CFU': 'https://images.unsplash.com/photo-1533104816931-20fa691ff6ca?w=800&h=600&fit=crop&q=80',
    'RHO': 'https://images.unsplash.com/photo-1601142634808-d1ff3a9e316a?w=800&h=600&fit=crop&q=80',
    
    # Portugal
    'LIS': 'https://images.unsplash.com/photo-1585208798174-6cedd86e019a?w=800&h=600&fit=crop&q=80',
    'OPO': 'https://images.unsplash.com/photo-1555881400-74d7acaacd8b?w=800&h=600&fit=crop&q=80',
    'FAO': 'https://images.unsplash.com/photo-1552733407-5d5c46c3bb3b?w=800&h=600&fit=crop&q=80',
    'FNC': 'https://images.unsplash.com/photo-1567527259232-3a7fcd490c55?w=800&h=600&fit=crop&q=80',
    
    # Eastern Europe
    'VIE': 'https://images.unsplash.com/photo-1516550893923-42d28e5677af?w=800&h=600&fit=crop&q=80',
    'PRG': 'https://images.unsplash.com/photo-1541849546-216549ae216d?w=800&h=600&fit=crop&q=80',
    'BUD': 'https://images.unsplash.com/photo-1549877452-9c387954fbc2?w=800&h=600&fit=crop&q=80',
    'WAW': 'https://images.unsplash.com/photo-1607427293702-036933bbf746?w=800&h=600&fit=crop&q=80',
    'KRK': 'https://images.unsplash.com/photo-1606992894253-6e18de2a0e47?w=800&h=600&fit=crop&q=80',
    'BTS': 'https://images.unsplash.com/photo-1609856878074-cf31e21ccb6b?w=800&h=600&fit=crop&q=80',
    
    # Scandinavia
    'CPH': 'https://images.unsplash.com/photo-1513622470522-26c3c8a854bc?w=800&h=600&fit=crop&q=80',
    'ARN': 'https://images.unsplash.com/photo-1509356843151-3e7d96241e11?w=800&h=600&fit=crop&q=80',
    'OSL': 'https://images.unsplash.com/photo-1533929736458-ca588d08c8be?w=800&h=600&fit=crop&q=80',
    'HEL': 'https://images.unsplash.com/photo-1538332576228-eb5b4c4de6f5?w=800&h=600&fit=crop&q=80',
    'GOT': 'https://images.unsplash.com/photo-1560453753-5ef350a93eb9?w=800&h=600&fit=crop&q=80',
    
    # Croatia
    'DBV': 'https://images.unsplash.com/photo-1555990538-1e6e5bc3bebb?w=800&h=600&fit=crop&q=80',
    'SPU': 'https://images.unsplash.com/photo-1559583109-3e7968136c99?w=800&h=600&fit=crop&q=80',
    'ZAG': 'https://images.unsplash.com/photo-1605206031982-c75e59878c94?w=800&h=600&fit=crop&q=80',
    'ZAD': 'https://images.unsplash.com/photo-1562632141-61e72b16a858?w=800&h=600&fit=crop&q=80',
    
    # Switzerland
    'ZRH': 'https://images.unsplash.com/photo-1515488764276-beab7607c1e6?w=800&h=600&fit=crop&q=80',
    'GVA': 'https://images.unsplash.com/photo-1535378620166-273708d44e4c?w=800&h=600&fit=crop&q=80',
    'BSL': 'https://images.unsplash.com/photo-1527004637293-6416dc62c36b?w=800&h=600&fit=crop&q=80',
    
    # Canary Islands
    'LPA': 'https://images.unsplash.com/photo-1521716691619-c13b61cff3f0?w=800&h=600&fit=crop&q=80',
    'TFS': 'https://images.unsplash.com/photo-1521804906057-1df8fdb718b7?w=800&h=600&fit=crop&q=80',
    'TFN': 'https://images.unsplash.com/photo-1521804906057-1df8fdb718b7?w=800&h=600&fit=crop&q=80',
    'ACE': 'https://images.unsplash.com/photo-1560625172-066d9c4f2046?w=800&h=600&fit=crop&q=80',
    'FUE': 'https://images.unsplash.com/photo-1584280077641-322e2230ade1?w=800&h=600&fit=crop&q=80',
    
    # Morocco & Israel
    'RAK': 'https://images.unsplash.com/photo-1539020140153-e479b8c22e70?w=800&h=600&fit=crop&q=80',
    'AGA': 'https://images.unsplash.com/photo-1568322674942-e5bfbc646774?w=800&h=600&fit=crop&q=80',
    'TLV': 'https://images.unsplash.com/photo-1552423314-cf29ab68ad73?w=800&h=600&fit=crop&q=80',
    
    # Turkey
    'DLM': 'https://images.unsplash.com/photo-1528045274126-4f8e4318dd5e?w=800&h=600&fit=crop&q=80',
    'AYT': 'https://images.unsplash.com/photo-1616423841125-8307665a0469?w=800&h=600&fit=crop&q=80',
    
    # Baltic
    'TLL': 'https://images.unsplash.com/photo-1594736797933-d0501ba2fe65?w=800&h=600&fit=crop&q=80',
    'RIX': 'https://images.unsplash.com/photo-1592113630251-67082dbdb7af?w=800&h=600&fit=crop&q=80',
    'VNO': 'https://images.unsplash.com/photo-1591098506952-f0fe7d11edd2?w=800&h=600&fit=crop&q=80',
    
    # Romania & Bulgaria
    'OTP': 'https://images.unsplash.com/photo-1587974928442-77dc3e0dba72?w=800&h=600&fit=crop&q=80',
    'SOF': 'https://images.unsplash.com/photo-1609103032237-ce3052bd0fb2?w=800&h=600&fit=crop&q=80',
    
    # Luxembourg
    'LUX': 'https://images.unsplash.com/photo-1559113513-e5bb93df5d94?w=800&h=600&fit=crop&q=80',
    
    # Additional airports - generic but good quality
    'BRE': 'https://images.unsplash.com/photo-1599641187333-e62f1f04e81b?w=800&h=600&fit=crop&q=80',
    'NUE': 'https://images.unsplash.com/photo-1573900675254-ea7c891f4bf5?w=800&h=600&fit=crop&q=80',
    'FKB': 'https://images.unsplash.com/photo-1573900675254-ea7c891f4bf5?w=800&h=600&fit=crop&q=80',
    'NQY': 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800&h=600&fit=crop&q=80',
    'EMA': 'https://images.unsplash.com/photo-1569764356260-f3f30ddbefb4?w=800&h=600&fit=crop&q=80',
    'NCL': 'https://images.unsplash.com/photo-1590767187868-b8e9ece0ebb2?w=800&h=600&fit=crop&q=80',
    'GLA': 'https://images.unsplash.com/photo-1548450000-bd893b598478?w=800&h=600&fit=crop&q=80',
    'LBA': 'https://images.unsplash.com/photo-1571136356578-52a2caf1ef73?w=800&h=600&fit=crop&q=80',
    'CWL': 'https://images.unsplash.com/photo-1584634428004-b31a14e84d94?w=800&h=600&fit=crop&q=80',
}

# Add fallback generic images for remaining airports
GENERIC_FALLBACK = 'https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=800&h=600&fit=crop&q=80'

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
        print(f"Error downloading: {e}")
        return False

def main():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    
    # Load all Ryanair airports
    with open('/workspaces/sate-adventures/code/flight-search-engine/data/ryanair-airports.json', 'r') as f:
        ryanair_data = json.load(f)
    
    all_airports = [airport['code'] for airport in ryanair_data['airports']]
    
    successful = []
    failed = []
    
    print(f"Downloading images for {len(all_airports)} airports...")
    
    for airport_code in all_airports:
        image_path = os.path.join(IMAGE_DIR, f"{airport_code}.jpg")
        
        # Skip if already exists
        if os.path.exists(image_path):
            print(f"✓ {airport_code} already exists")
            successful.append(airport_code)
            continue
        
        # Get image URL
        image_url = AIRPORT_IMAGES.get(airport_code, GENERIC_FALLBACK)
        
        print(f"Downloading {airport_code}...", end=' ')
        if download_image(image_url, image_path):
            print("✓")
            successful.append(airport_code)
        else:
            # Try fallback
            print("trying fallback...", end=' ')
            if download_image(GENERIC_FALLBACK, image_path):
                print("✓")
                successful.append(airport_code)
            else:
                print("✗")
                failed.append(airport_code)
        
        # Rate limiting
        time.sleep(0.3)
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Download Summary:")
    print(f"  Successful: {len(successful)}/{len(all_airports)} airports")
    print(f"  Failed: {len(failed)} airports")
    
    if failed:
        print(f"\nFailed airports: {', '.join(failed)}")
    
    # Create a mapping file
    mapping = {}
    for airport_code in successful:
        mapping[airport_code] = f"/airport-images/{airport_code}.jpg"
    
    mapping_file = '/workspaces/sate-adventures/code/flight-search-engine/public/airport-images-mapping.json'
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"\nCreated mapping file: {mapping_file}")

if __name__ == "__main__":
    main()