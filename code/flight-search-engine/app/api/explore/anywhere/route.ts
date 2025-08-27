import { NextRequest, NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';
import { format } from 'date-fns';
import RyanairAirports from '@/data/ryanair-airports.json';
import AirportImagesMapping from '@/public/airport-images-mapping.json';

const execAsync = promisify(exec);

interface AnywhereRequest {
  origin: string;
  outboundDate: string;
  returnDate: string;
}

interface FlightInfo {
  flightNumber: string;
  departureTime: string;
  arrivalTime: string;
  price: number;
}

interface DestinationResult {
  destination: string;
  destinationName: string;
  city: string;
  country: string;
  outboundPrice: number;
  returnPrice: number;
  totalPrice: number;
  outboundFlight: FlightInfo;
  returnFlight: FlightInfo;
  bookingUrl: string;
  imageUrl?: string;
}


function generateRoundTripBookingUrl(origin: string, destination: string, outboundDate: string, returnDate: string): string {
  // Generate Ryanair booking URL for round trip
  const baseUrl = 'https://www.ryanair.com/es/es/trip/flights/select';
  const params = new URLSearchParams({
    adults: '1',
    children: '0',
    infants: '0',
    teens: '0',
    tpAdults: '1',
    tpChildren: '0',
    tpInfants: '0',
    tpTeens: '0',
    dateOut: outboundDate,
    tpStartDate: outboundDate,
    dateIn: returnDate,
    tpEndDate: returnDate,
    isReturn: 'true',
    discount: '0',
    tpDiscount: '0',
    originIata: origin,
    destinationIata: destination,
    tpOriginIata: origin,
    tpDestinationIata: destination,
  });
  
  return `${baseUrl}?${params.toString()}`;
}

export async function POST(request: NextRequest) {
  try {
    const body: AnywhereRequest = await request.json();
    
    // Validate required fields
    if (!body.origin || !body.outboundDate || !body.returnDate) {
      return NextResponse.json(
        { error: 'Missing required fields: origin, outboundDate, and returnDate' },
        { status: 400 }
      );
    }

    // Find origin airport details
    const originAirport = RyanairAirports.airports.find(a => a.code === body.origin);
    if (!originAirport) {
      return NextResponse.json(
        { error: 'Invalid origin airport code' },
        { status: 400 }
      );
    }

    // Execute Python script to get return trip data
    const scriptPath = '/workspaces/sate-adventures/code/flight-search-engine/test-return-anywhere.py';
    
    const { stdout, stderr } = await execAsync(
      `python3 ${scriptPath} "${body.origin}" "${body.outboundDate}" "${body.returnDate}"`
    );
    
    if (stderr && !stderr.includes('WARNING')) {
      console.error('Python script error:', stderr);
    }

    const rawResults = JSON.parse(stdout);
    
    // Transform and enrich the results
    const destinations: DestinationResult[] = rawResults.map((trip: any) => {
      const destAirport = RyanairAirports.airports.find(a => a.code === trip.destination);
      
      return {
        destination: trip.destination,
        destinationName: trip.destinationName,
        city: destAirport?.city || trip.destinationName.split(',')[0],
        country: destAirport?.country || trip.destinationName.split(',')[1]?.trim() || '',
        outboundPrice: Math.round(trip.outboundPrice),
        returnPrice: Math.round(trip.returnPrice),
        totalPrice: Math.round(trip.totalPrice),
        outboundFlight: {
          flightNumber: trip.outboundFlight.flightNumber,
          departureTime: trip.outboundFlight.departureTime,
          arrivalTime: trip.outboundFlight.arrivalTime || 
            new Date(new Date(trip.outboundFlight.departureTime).getTime() + 2 * 60 * 60 * 1000).toISOString(),
          price: Math.round(trip.outboundPrice)
        },
        returnFlight: {
          flightNumber: trip.returnFlight.flightNumber,
          departureTime: trip.returnFlight.departureTime,
          arrivalTime: trip.returnFlight.arrivalTime || 
            new Date(new Date(trip.returnFlight.departureTime).getTime() + 2 * 60 * 60 * 1000).toISOString(),
          price: Math.round(trip.returnPrice)
        },
        bookingUrl: generateRoundTripBookingUrl(body.origin, trip.destination, body.outboundDate, body.returnDate),
        imageUrl: AirportImagesMapping[trip.destination as keyof typeof AirportImagesMapping] || '/airport-images/generic.jpg'
      };
    });

    // Sort by total price
    destinations.sort((a, b) => a.totalPrice - b.totalPrice);

    return NextResponse.json({
      origin: body.origin,
      originName: `${originAirport.city}, ${originAirport.country}`,
      destinations: destinations.slice(0, 50), // Limit to top 50
      searchDates: {
        outbound: body.outboundDate,
        return: body.returnDate
      },
      totalFound: destinations.length
    });
    
  } catch (error) {
    console.error('Anywhere search error:', error);
    return NextResponse.json(
      { error: 'Internal server error', details: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
}