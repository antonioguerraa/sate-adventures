#!/usr/bin/env python3
"""
Search for round-trip flights to anywhere from a given origin
"""
import json
import sys
from datetime import datetime, timedelta
from ryanair import Ryanair

def search_return_anywhere(origin, outbound_date_str, return_date_str):
    """Search for return trips to anywhere from origin"""
    try:
        api = Ryanair(currency="EUR")
        
        # Parse dates
        outbound_date = datetime.strptime(outbound_date_str, "%Y-%m-%d").date()
        return_date = datetime.strptime(return_date_str, "%Y-%m-%d").date()
        
        # Search for exact dates only (no buffer)
        # Using same date for start and end means exact date search
        
        # Get return trips to anywhere
        trips = api.get_cheapest_return_flights(
            origin,
            outbound_date,
            outbound_date,  # Same date for exact search
            return_date,
            return_date     # Same date for exact search
        )
        
        # Group by destination and find cheapest
        destinations = {}
        flight_counts = {}
        
        for trip in trips:
            dest = trip.outbound.destination
            
            # Count trips per destination
            if dest not in flight_counts:
                flight_counts[dest] = 0
            flight_counts[dest] += 1
            
            # Keep only the cheapest trip
            if dest not in destinations or trip.totalPrice < destinations[dest]['totalPrice']:
                destinations[dest] = {
                    'destination': dest,
                    'destinationName': trip.outbound.destinationFull,
                    'outboundPrice': trip.outbound.price,
                    'returnPrice': trip.inbound.price,
                    'totalPrice': trip.totalPrice,
                    'outboundFlight': {
                        'flightNumber': trip.outbound.flightNumber,
                        'departureTime': trip.outbound.departureTime.isoformat(),
                    },
                    'returnFlight': {
                        'flightNumber': trip.inbound.flightNumber,
                        'departureTime': trip.inbound.departureTime.isoformat(),
                    }
                }
        
        # Add flight count information
        for dest in destinations:
            destinations[dest]['hasMultipleFlights'] = flight_counts[dest] > 1
            destinations[dest]['flightCount'] = flight_counts[dest]
        
        # Sort by total price and return top results
        sorted_dests = sorted(destinations.values(), key=lambda x: x['totalPrice'])
        return sorted_dests[:100]  # Return top 100 destinations
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return []

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(json.dumps([]))
        sys.exit(0)
    
    origin = sys.argv[1]
    outbound_date = sys.argv[2]
    return_date = sys.argv[3]
    
    results = search_return_anywhere(origin, outbound_date, return_date)
    print(json.dumps(results))