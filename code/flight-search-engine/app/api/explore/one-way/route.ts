import { NextRequest, NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';
import RyanairAirports from '@/data/ryanair-airports.json';
import AirportImagesMapping from '@/public/airport-images-mapping.json';

const execAsync = promisify(exec);

interface OneWayRequest {
  origin: string;
  departureDate: string;
}

interface FlightResult {
  destination: string;
  destinationName: string;
  city: string;
  country: string;
  price: number;
  flightNumber: string;
  departureTime: string;
  arrivalTime: string;
  bookingUrl: string;
  imageUrl?: string;
}


function generateBookingUrl(origin: string, destination: string, departureDate: string): string {
  // Generate Ryanair booking URL for one-way flight
  const baseUrl = 'https://www.ryanair.com/es/es/trip/flights/select';
  const params = new URLSearchParams({
    adults: '1',
    children: '0',
    infants: '0',
    teens: '0',
    dateOut: departureDate,
    isReturn: 'false',
    discount: '0',
    originIata: origin,
    destinationIata: destination,
  });
  
  return `${baseUrl}?${params.toString()}`;
}

export async function POST(request: NextRequest) {
  try {
    const body: OneWayRequest = await request.json();
    
    // Validate required fields
    if (!body.origin || !body.departureDate) {
      return NextResponse.json(
        { error: 'Missing required fields: origin and departureDate' },
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

    // Execute Python script to get one-way flight data
    const scriptPath = '/workspaces/sate-adventures/code/flight-search-engine/search-one-way-anywhere.py';
    
    const { stdout, stderr } = await execAsync(
      `python3 ${scriptPath} "${body.origin}" "${body.departureDate}"`
    );
    
    if (stderr && !stderr.includes('WARNING')) {
      console.error('Python script error:', stderr);
    }

    const rawResults = JSON.parse(stdout);
    
    // Transform and enrich the results
    const destinations: FlightResult[] = rawResults.map((flight: any) => {
      const destAirport = RyanairAirports.airports.find(a => a.code === flight.destination);
      
      return {
        destination: flight.destination,
        destinationName: flight.destinationName,
        city: destAirport?.city || flight.destinationName.split(',')[0],
        country: destAirport?.country || flight.destinationName.split(',')[1]?.trim() || '',
        price: Math.round(flight.price),
        flightNumber: flight.flightNumber,
        departureTime: flight.departureTime,
        arrivalTime: flight.arrivalTime || 
          new Date(new Date(flight.departureTime).getTime() + 2 * 60 * 60 * 1000).toISOString(),
        bookingUrl: generateBookingUrl(body.origin, flight.destination, body.departureDate),
        imageUrl: AirportImagesMapping[flight.destination as keyof typeof AirportImagesMapping] || '/airport-images/generic.jpg'
      };
    });

    // Sort by price
    destinations.sort((a, b) => a.price - b.price);

    return NextResponse.json({
      origin: body.origin,
      originName: `${originAirport.city}, ${originAirport.country}`,
      destinations: destinations.slice(0, 50), // Limit to top 50
      searchDate: body.departureDate,
      totalFound: destinations.length
    });
    
  } catch (error) {
    console.error('One-way search error:', error);
    return NextResponse.json(
      { error: 'Internal server error', details: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
}