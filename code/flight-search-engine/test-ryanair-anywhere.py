#!/usr/bin/env python3
"""
Test script for Ryanair API 'Search Anywhere' functionality
Testing flights from Alicante (ALC) to ANY destination
"""

from datetime import datetime, timedelta
from ryanair import Ryanair
import json

def test_search_anywhere():
    """Test Ryanair's 'search anywhere' feature"""
    
    # Initialize the API with EUR currency
    api = Ryanair(currency="EUR")
    
    # Set search parameters matching the URL
    origin = "ALC"  # Alicante Airport
    outbound_date = datetime(2025, 8, 7).date()
    return_date = datetime(2025, 8, 14).date()
    
    print("=" * 70)
    print("RYANAIR 'SEARCH ANYWHERE' TEST")
    print(f"From: {origin} (Alicante)")
    print("To: ANYWHERE")
    print(f"Outbound: {outbound_date.strftime('%B %d, %Y')}")
    print(f"Return: {return_date.strftime('%B %d, %Y')}")
    print("=" * 70)
    print()
    
    try:
        # 1. Search for one-way flights to anywhere
        print("🔍 SEARCHING ONE-WAY FLIGHTS TO ANYWHERE...")
        print("-" * 70)
        
        # Get all cheapest flights from ALC on the outbound date
        outbound_next_day = outbound_date + timedelta(days=1)
        one_way_flights = api.get_cheapest_flights(origin, outbound_date, outbound_next_day)
        
        if one_way_flights:
            # Group by destination for better organization
            destinations_dict = {}
            for flight in one_way_flights:
                dest = flight.destination
                if dest not in destinations_dict or flight.price < destinations_dict[dest]['price']:
                    destinations_dict[dest] = {
                        'flight': flight,
                        'destination_full': flight.destinationFull,
                        'price': flight.price
                    }
            
            # Sort by price
            sorted_destinations = sorted(destinations_dict.items(), key=lambda x: x[1]['price'])
            
            print(f"\n✈️ Found flights to {len(sorted_destinations)} destinations from {origin}:\n")
            
            # Show top 15 cheapest destinations
            for i, (dest_code, info) in enumerate(sorted_destinations[:15], 1):
                flight = info['flight']
                print(f"{i:2}. {info['destination_full']} ({dest_code})")
                print(f"    💰 €{flight.price:.2f}")
                print(f"    🕐 {flight.departureTime.strftime('%H:%M')} - Flight {flight.flightNumber}")
                print()
            
            if len(sorted_destinations) > 15:
                print(f"... and {len(sorted_destinations) - 15} more destinations\n")
            
            # Price statistics
            prices = [info['price'] for _, info in sorted_destinations]
            print("📊 PRICE STATISTICS:")
            print(f"  • Cheapest: €{min(prices):.2f}")
            print(f"  • Most expensive: €{max(prices):.2f}")
            print(f"  • Average: €{sum(prices)/len(prices):.2f}")
            print(f"  • Total destinations: {len(sorted_destinations)}")
        else:
            print("❌ No flights found from ALC on this date")
        
        # 2. Search for return trips to anywhere
        print("\n" + "=" * 70)
        print("🔄 SEARCHING RETURN TRIPS TO ANYWHERE...")
        print("-" * 70)
        
        return_next_day = return_date + timedelta(days=1)
        
        # Get return trips
        print(f"\nSearching return trips ({outbound_date} to {return_date})...\n")
        
        return_trips = api.get_cheapest_return_flights(
            origin,
            outbound_date,
            outbound_next_day,
            return_date,
            return_next_day
        )
        
        if return_trips:
            # Group by destination
            trip_destinations = {}
            for trip in return_trips:
                dest = trip.outbound.destination
                if dest not in trip_destinations or trip.totalPrice < trip_destinations[dest]['price']:
                    trip_destinations[dest] = {
                        'trip': trip,
                        'price': trip.totalPrice,
                        'destination_full': trip.outbound.destinationFull
                    }
            
            # Sort by total price
            sorted_trips = sorted(trip_destinations.items(), key=lambda x: x[1]['price'])
            
            print(f"✈️ Found return trips to {len(sorted_trips)} destinations:\n")
            
            # Show top 10 cheapest return trips
            for i, (dest_code, info) in enumerate(sorted_trips[:10], 1):
                trip = info['trip']
                print(f"{i:2}. {info['destination_full']} ({dest_code})")
                print(f"    💰 Total: €{trip.totalPrice:.2f}")
                print(f"    ✈️  Outbound: {trip.outbound.flightNumber} at {trip.outbound.departureTime.strftime('%H:%M')} - €{trip.outbound.price:.2f}")
                print(f"    ✈️  Return: {trip.inbound.flightNumber} at {trip.inbound.departureTime.strftime('%H:%M')} - €{trip.inbound.price:.2f}")
                print()
            
            if len(sorted_trips) > 10:
                print(f"... and {len(sorted_trips) - 10} more destinations\n")
        else:
            print("❌ No return trips found")
        
        # 3. Popular destinations with best prices
        print("\n" + "=" * 70)
        print("🌟 POPULAR DESTINATIONS FROM ALICANTE")
        print("-" * 70)
        
        popular_destinations = {
            "LON": "London (All)",
            "MAD": "Madrid",
            "BCN": "Barcelona", 
            "MIL": "Milan (All)",
            "ROM": "Rome (All)",
            "PAR": "Paris (All)",
            "BER": "Berlin",
            "DUB": "Dublin",
            "AMS": "Amsterdam",
            "BRU": "Brussels"
        }
        
        print("\nChecking prices to popular destinations:\n")
        
        for dest_code, dest_name in popular_destinations.items():
            # Find flights to this destination
            matching_flights = [f for f in one_way_flights if f.destination.startswith(dest_code[:3])]
            if matching_flights:
                cheapest = min(matching_flights, key=lambda x: x.price)
                print(f"  • {dest_name}: €{cheapest.price:.2f} ({cheapest.destination})")
        
        # 4. Best value analysis
        print("\n" + "=" * 70)
        print("💡 RECOMMENDATIONS")
        print("-" * 70)
        
        if sorted_destinations:
            print("\n🏆 TOP 5 BEST VALUE DESTINATIONS:")
            for i, (dest_code, info) in enumerate(sorted_destinations[:5], 1):
                flight = info['flight']
                print(f"  {i}. {info['destination_full']} - €{flight.price:.2f}")
            
            # Find destinations under €30
            budget_destinations = [(code, info) for code, info in sorted_destinations if info['price'] < 30]
            if budget_destinations:
                print(f"\n💰 BUDGET OPTIONS (Under €30): {len(budget_destinations)} destinations")
                
            # Find destinations under €50
            mid_range = [(code, info) for code, info in sorted_destinations if 30 <= info['price'] < 50]
            if mid_range:
                print(f"💵 MID-RANGE (€30-50): {len(mid_range)} destinations")
            
            # Weekend trip recommendations (if searching for weekdays)
            if outbound_date.weekday() < 5:  # If weekday
                print("\n📅 Consider weekend travel for potentially better deals!")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        print("\nPossible reasons:")
        print("  • API rate limiting")
        print("  • Network connection issues")
        print("  • Invalid airport code")
        import traceback
        traceback.print_exc()

def test_flexible_search():
    """Test searching with flexible dates"""
    
    print("\n" + "=" * 70)
    print("📅 FLEXIBLE DATE SEARCH (±3 days)")
    print("-" * 70)
    
    api = Ryanair(currency="EUR")
    origin = "ALC"
    base_date = datetime(2025, 8, 7).date()
    
    print(f"\nSearching for best prices around {base_date} (±3 days):\n")
    
    best_prices = []
    
    for day_offset in range(-3, 4):
        check_date = base_date + timedelta(days=day_offset)
        next_day = check_date + timedelta(days=1)
        
        try:
            flights = api.get_cheapest_flights(origin, check_date, next_day)
            if flights:
                # Get top 5 cheapest for this day
                cheapest = sorted(flights, key=lambda x: x.price)[:5]
                for flight in cheapest:
                    best_prices.append({
                        'date': check_date,
                        'destination': flight.destination,
                        'destination_full': flight.destinationFull,
                        'price': flight.price,
                        'flight': flight.flightNumber,
                        'time': flight.departureTime.strftime('%H:%M')
                    })
        except:
            pass
    
    if best_prices:
        # Sort by price
        best_prices.sort(key=lambda x: x['price'])
        
        print("🏆 BEST DEALS ACROSS ALL DATES:\n")
        for i, deal in enumerate(best_prices[:10], 1):
            date_str = deal['date'].strftime('%b %d (%a)')
            print(f"{i:2}. {date_str} → {deal['destination_full']}")
            print(f"    €{deal['price']:.2f} | Flight {deal['flight']} at {deal['time']}")
            print()

if __name__ == "__main__":
    print("=" * 70)
    print("RYANAIR 'SEARCH ANYWHERE' FEATURE TEST")
    print("Testing the ability to search for flights to ANY destination")
    print("=" * 70)
    print()
    
    # Run the main test
    test_search_anywhere()
    
    # Run flexible date search
    test_flexible_search()
    
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE")
    print("=" * 70)