#!/usr/bin/env python3
"""
Search for exact one-way flights (no date buffer) to anywhere from a given origin
"""
import json
import sys
from datetime import datetime
from ryanair import Ryanair

def search_exact_one_way(origin, departure_date_str):
    """Search for exact date one-way trips to anywhere from origin"""
    try:
        api = Ryanair(currency="EUR")
        
        # Parse date - NO BUFFER, exact date only
        departure_date = datetime.strptime(departure_date_str, "%Y-%m-%d").date()
        
        # Get one-way flights for EXACT date
        flights = api.get_cheapest_flights(
            origin,
            departure_date,
            departure_date,  # Same date - no buffer
            destination_country=None
        )
        
        # Group by destination and find cheapest
        destinations = {}
        for flight in flights:
            dest = flight.destination
            if dest not in destinations or flight.price < destinations[dest]['price']:
                destinations[dest] = {
                    'destination': dest,
                    'destinationName': flight.destinationFull,
                    'price': flight.price,
                    'flightNumber': flight.flightNumber,
                    'departureTime': flight.departureTime.isoformat(),
                    'arrivalTime': flight.arrivalTime.isoformat() if hasattr(flight, 'arrivalTime') and flight.arrivalTime else None,
                    # Include the actual flight date for verification
                    'actualDepartureDate': flight.departureTime.date().isoformat()
                }
        
        # Sort by price and return top results
        sorted_dests = sorted(destinations.values(), key=lambda x: x['price'])
        return sorted_dests[:100]  # Return top 100 destinations
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return []

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(json.dumps([]))
        sys.exit(0)
    
    origin = sys.argv[1]
    departure_date = sys.argv[2]
    
    results = search_exact_one_way(origin, departure_date)
    print(json.dumps(results))