#!/usr/bin/env python3
"""
Search for one-way flights to anywhere from a given origin
"""
import json
import sys
from datetime import datetime, timedelta
from ryanair import Ryanair

def search_one_way_anywhere(origin, departure_date_str):
    """Search for one-way trips to anywhere from origin"""
    try:
        api = Ryanair(currency="EUR")
        
        # Parse date
        departure_date = datetime.strptime(departure_date_str, "%Y-%m-%d").date()
        
        # Search for exact date only (no buffer)
        # Using same date for start and end means exact date search
        
        # Get one-way flights to anywhere
        flights = api.get_cheapest_flights(
            origin,
            departure_date,
            departure_date,  # Same date for exact search
            destination_country=None
        )
        
        # Group by destination and find cheapest
        destinations = {}
        flight_counts = {}
        
        for flight in flights:
            dest = flight.destination
            
            # Count flights per destination
            if dest not in flight_counts:
                flight_counts[dest] = 0
            flight_counts[dest] += 1
            
            # Keep only the cheapest flight
            if dest not in destinations or flight.price < destinations[dest]['price']:
                destinations[dest] = {
                    'destination': dest,
                    'destinationName': flight.destinationFull,
                    'price': flight.price,
                    'flightNumber': flight.flightNumber,
                    'departureTime': flight.departureTime.isoformat(),
                    'arrivalTime': flight.arrivalTime.isoformat() if hasattr(flight, 'arrivalTime') and flight.arrivalTime else None
                }
        
        # Add flight count information
        for dest in destinations:
            destinations[dest]['hasMultipleFlights'] = flight_counts[dest] > 1
            destinations[dest]['flightCount'] = flight_counts[dest]
        
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
    
    results = search_one_way_anywhere(origin, departure_date)
    print(json.dumps(results))