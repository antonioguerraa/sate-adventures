'use client';

import { useSearchParams, useRouter } from 'next/navigation';
import { useState, useEffect, Suspense, useRef, useCallback } from 'react';
import { format } from 'date-fns';
import { MapPin, Calendar, Plane, Euro, ExternalLink, ArrowLeft } from 'lucide-react';
import Link from 'next/link';
import dynamic from 'next/dynamic';
import AirportImagesMapping from '@/public/airport-images-mapping.json';
import RyanairAirports from '@/data/ryanair-airports.json';
import { translations, Language } from '@/lib/translations';
import LanguageSelector from '@/components/LanguageSelector';
import QuickSearch from '@/components/QuickSearch';

// Dynamically import the map component
const SimpleDestinationMap = dynamic(() => import('@/components/SimpleDestinationMap'), {
  ssr: false,
  loading: () => <div className="animate-pulse bg-gray-100 h-full rounded-xl" />
});

interface Flight {
  destination: string;
  destinationName: string;
  city: string;
  country: string;
  price?: number;
  totalPrice?: number;
  outboundPrice?: number;
  returnPrice?: number;
  flightNumber?: string;
  departureTime?: string;
  arrivalTime?: string;
  outboundFlight?: {
    flightNumber: string;
    departureTime: string;
    arrivalTime: string;
  };
  returnFlight?: {
    flightNumber: string;
    departureTime: string;
    arrivalTime: string;
  };
  bookingUrl: string;
  imageUrl?: string;
  hasMultipleFlights?: boolean;
  flightCount?: number;
}

function ResultsContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const [results, setResults] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [hoveredDestination, setHoveredDestination] = useState<string | null>(null);
  const [language, setLanguage] = useState<Language>('en');
  const listRef = useRef<HTMLDivElement>(null);
  const itemRefs = useRef<{ [key: string]: HTMLDivElement | null }>({});
  
  const t = translations[language];

  const [tripType, setTripType] = useState<'one-way' | 'round-trip'>(
    (searchParams.get('tripType') as 'one-way' | 'round-trip') || 'round-trip'
  );
  const [origin, setOrigin] = useState(searchParams.get('origin') || '');
  const [departureDate, setDepartureDate] = useState(searchParams.get('departureDate') || '');
  const [returnDate, setReturnDate] = useState(searchParams.get('returnDate') || '');

  const fetchResults = useCallback(async (
    searchOrigin?: string,
    searchDepartureDate?: string,
    searchReturnDate?: string,
    searchTripType?: 'one-way' | 'round-trip'
  ) => {
    const finalOrigin = searchOrigin || origin;
    const finalDepartureDate = searchDepartureDate || departureDate;
    const finalReturnDate = searchReturnDate || returnDate;
    const finalTripType = searchTripType || tripType;
    
    if (!finalOrigin || !finalDepartureDate) return;

    setLoading(true);
    
    try {
      const endpoint = finalTripType === 'one-way' 
        ? '/api/explore/one-way'
        : '/api/explore/anywhere';
      
      const body = finalTripType === 'one-way'
        ? { origin: finalOrigin, departureDate: finalDepartureDate }
        : { origin: finalOrigin, outboundDate: finalDepartureDate, returnDate: finalReturnDate };

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      // Ensure all destinations have images
      if (data.destinations) {
        data.destinations = data.destinations.map((dest: Flight) => ({
          ...dest,
          imageUrl: dest.imageUrl || AirportImagesMapping[dest.destination as keyof typeof AirportImagesMapping] || 
                    '/airport-images/generic.jpg'
        }));
      }
      
      setResults(data);
    } catch (error) {
      console.error('Search failed:', error);
    } finally {
      setLoading(false);
    }
  }, [origin, departureDate, returnDate, tripType]);

  useEffect(() => {
    fetchResults();
  }, [fetchResults]);

  const handleSearch = useCallback((params: {
    origin: string;
    departureDate: string;
    returnDate: string;
    tripType: 'one-way' | 'round-trip';
  }) => {
    // Update state
    setOrigin(params.origin);
    setDepartureDate(params.departureDate);
    setReturnDate(params.returnDate);
    setTripType(params.tripType);
    
    // Update URL without causing re-render
    const searchParams = new URLSearchParams({
      tripType: params.tripType,
      origin: params.origin,
      departureDate: params.departureDate,
      ...(params.tripType === 'round-trip' && { returnDate: params.returnDate })
    });
    window.history.replaceState(null, '', `/results?${searchParams.toString()}`);
    
    // Fetch new results
    fetchResults(params.origin, params.departureDate, params.returnDate, params.tripType);
  }, [fetchResults]);

  const handleDestinationClick = (destination: string) => {
    // Scroll to the item in the list
    if (itemRefs.current[destination]) {
      itemRefs.current[destination]?.scrollIntoView({ 
        behavior: 'smooth', 
        block: 'center' 
      });
      // Highlight the item briefly
      setHoveredDestination(destination);
      setTimeout(() => setHoveredDestination(null), 2000);
    }
  };

  const handleDestinationHover = (destination: string | null) => {
    setHoveredDestination(destination);
    // Also scroll to the item when hovering on map
    if (destination && itemRefs.current[destination]) {
      itemRefs.current[destination]?.scrollIntoView({ 
        behavior: 'smooth', 
        block: 'center' 
      });
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50 flex items-center justify-center">
        <div className="bg-white rounded-2xl p-8 shadow-xl max-w-sm w-full mx-4">
          <div className="flex flex-col items-center">
            <div className="relative w-24 h-24 mb-4">
              <div className="absolute inset-0 border-4 border-blue-200 rounded-full"></div>
              <div className="absolute inset-0 border-4 border-t-blue-600 rounded-full animate-spin"></div>
              <Plane className="absolute inset-0 m-auto w-10 h-10 text-blue-600 animate-pulse" />
            </div>
            <p className="text-lg font-semibold text-gray-800 mb-2">{t.loading}</p>
            <p className="text-sm text-gray-600 text-center">{t.searchingFrom} {(() => {
              const airport = RyanairAirports.airports.find(a => a.code === origin);
              return airport ? `${airport.city} (${origin})` : origin;
            })()}...</p>
          </div>
        </div>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50 p-8">
        <div className="text-center">
          <p className="text-gray-600 mb-4">{t.noResults}</p>
          <Link href="/" className="inline-block bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors">
            {t.backToSearch}
          </Link>
        </div>
      </div>
    );
  }

  const destinations = results.destinations || [];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-indigo-600 shadow-lg">
        <div className="max-w-7xl mx-auto px-4 py-3">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-3">
              <img 
                src="/logo.svg" 
                alt="vuelochill" 
                className="w-10 h-10 drop-shadow-md"
              />
              <div>
                <h1 className="text-xl font-bold text-white">vuelochill</h1>
                <Link href="/" className="flex items-center gap-1 text-blue-100 hover:text-white transition-colors text-xs">
                  <ArrowLeft className="w-3 h-3" />
                  <span>{t.newSearch}</span>
                </Link>
              </div>
            </div>
            <LanguageSelector 
              currentLanguage={language} 
              onLanguageChange={setLanguage}
              isWhite={true}
            />
          </div>
        </div>
      </div>
      
      {/* Quick Search Bar */}
      <QuickSearch
        origin={origin}
        departureDate={departureDate}
        returnDate={returnDate}
        tripType={tripType}
        language={language}
        onSearch={handleSearch}
      />

      {/* Main Content */}
      <div className="max-w-7xl mx-auto p-4">
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          {/* Left Column - Results List */}
          <div className="lg:col-span-2 space-y-3 max-h-[calc(100vh-140px)] overflow-y-auto pr-2" ref={listRef}>
            {destinations.map((dest: Flight) => (
              <div
                key={dest.destination}
                ref={(el) => { itemRefs.current[dest.destination] = el; }}
                className={`bg-white rounded-xl shadow-sm hover:shadow-md transition-all cursor-pointer border-2 ${
                  hoveredDestination === dest.destination 
                    ? 'border-blue-500 shadow-lg scale-[1.02]' 
                    : 'border-transparent'
                }`}
                onMouseEnter={() => setHoveredDestination(dest.destination)}
                onMouseLeave={() => setHoveredDestination(null)}
              >
                <div className="flex">
                  {/* Image */}
                  <div className="w-32 h-32 flex-shrink-0">
                    <img 
                      src={dest.imageUrl} 
                      alt={dest.city}
                      className="w-full h-full object-cover rounded-l-xl"
                    />
                  </div>
                  
                  {/* Content */}
                  <div className="flex-1 p-4">
                    <div className="flex justify-between items-start mb-2">
                      <div>
                        <h3 className="font-bold text-lg text-gray-900">{dest.city} ({dest.destination})</h3>
                        <p className="text-sm text-gray-600">{dest.destinationName || dest.country}</p>
                      </div>
                      <div className="text-right">
                        <p className="text-2xl font-bold text-green-600">
                          €{tripType === 'one-way' ? dest.price : dest.totalPrice}
                        </p>
                      </div>
                    </div>

                    {tripType === 'one-way' ? (
                      <div className="text-sm text-gray-600 mb-3">
                        <p className="flex items-center gap-1">
                          <Plane className="w-3 h-3" />
                          {dest.departureTime && format(new Date(dest.departureTime), 'HH:mm')} • {dest.flightNumber}
                        </p>
                        {dest.hasMultipleFlights && (
                          <p className="text-xs text-blue-600 mt-1">+ more flights available</p>
                        )}
                      </div>
                    ) : (
                      <div className="text-sm text-gray-600 mb-3 space-y-1">
                        <p className="flex items-center gap-1">
                          <span className="text-gray-500">{t.out}:</span> 
                          {dest.outboundFlight?.departureTime && format(new Date(dest.outboundFlight.departureTime), 'HH:mm')} 
                          • €{dest.outboundPrice}
                        </p>
                        <p className="flex items-center gap-1">
                          <span className="text-gray-500">{t.ret}:</span> 
                          {dest.returnFlight?.departureTime && format(new Date(dest.returnFlight.departureTime), 'HH:mm')} 
                          • €{dest.returnPrice}
                        </p>
                        {dest.hasMultipleFlights && (
                          <p className="text-xs text-blue-600">+ more flights available</p>
                        )}
                      </div>
                    )}

                    <a
                      href={dest.bookingUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-2 bg-blue-600 text-white px-3 py-1.5 rounded-lg text-sm hover:bg-blue-700 transition-colors"
                      onClick={(e) => e.stopPropagation()}
                    >
                      {t.bookOnRyanair}
                      <ExternalLink className="w-3 h-3" />
                    </a>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Right Column - Map */}
          <div className="lg:col-span-3 bg-white rounded-xl shadow-md p-4 h-[calc(100vh-140px)] sticky top-4">
            <SimpleDestinationMap
              origin={origin}
              destinations={destinations}
              hoveredDestination={hoveredDestination}
              onDestinationHover={handleDestinationHover}
              onDestinationClick={handleDestinationClick}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default function ResultsPage() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <ResultsContent />
    </Suspense>
  );
}