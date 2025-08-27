'use client';

import { useState, useEffect, useRef } from 'react';
import { Calendar, MapPin } from 'lucide-react';
import { format } from 'date-fns';
import RyanairAirports from '@/data/ryanair-airports.json';
import { translations, Language } from '@/lib/translations';

interface QuickSearchProps {
  origin: string;
  departureDate: string;
  returnDate: string;
  tripType: 'one-way' | 'round-trip';
  language: Language;
  onSearch: (params: {
    origin: string;
    departureDate: string;
    returnDate: string;
    tripType: 'one-way' | 'round-trip';
  }) => void;
}

export default function QuickSearch({
  origin: initialOrigin,
  departureDate: initialDepartureDate,
  returnDate: initialReturnDate,
  tripType: initialTripType,
  language,
  onSearch
}: QuickSearchProps) {
  const [origin, setOrigin] = useState(initialOrigin);
  const [departureDate, setDepartureDate] = useState(initialDepartureDate);
  const [returnDate, setReturnDate] = useState(initialReturnDate);
  const [tripType, setTripType] = useState(initialTripType);
  const [searchQuery, setSearchQuery] = useState('');
  const [showDropdown, setShowDropdown] = useState(false);
  const [filteredAirports, setFilteredAirports] = useState(RyanairAirports.airports);
  const [minDate, setMinDate] = useState('');
  const returnDateRef = useRef<HTMLInputElement>(null);
  
  const t = translations[language];

  // Get display name for airport
  const getAirportDisplay = (code: string) => {
    const airport = RyanairAirports.airports.find(a => a.code === code);
    if (airport) {
      return `${airport.city} (${airport.code})`;
    }
    return code;
  };

  // Initialize search query with proper display
  useEffect(() => {
    if (initialOrigin) {
      setSearchQuery(getAirportDisplay(initialOrigin));
      setOrigin(initialOrigin);
    }
  }, [initialOrigin]);

  // Set min date on client side only to avoid hydration issues
  useEffect(() => {
    setMinDate(format(new Date(), 'yyyy-MM-dd'));
  }, []);

  // Filter airports based on search
  useEffect(() => {
    const query = searchQuery.replace(/\s*\([A-Z]{3}\)\s*$/, ''); // Remove code in parentheses for search
    if (query && query !== origin) {
      const filtered = RyanairAirports.airports.filter(
        airport => 
          airport.code.toLowerCase().includes(query.toLowerCase()) ||
          airport.city.toLowerCase().includes(query.toLowerCase()) ||
          airport.country.toLowerCase().includes(query.toLowerCase()) ||
          airport.name.toLowerCase().includes(query.toLowerCase())
      );
      setFilteredAirports(filtered);
    } else {
      setFilteredAirports(RyanairAirports.airports.slice(0, 5));
    }
  }, [searchQuery, origin]);

  // Auto-trigger search when parameters change
  useEffect(() => {
    // Only trigger if values actually changed from initial
    const hasChanged = 
      origin !== initialOrigin || 
      departureDate !== initialDepartureDate || 
      returnDate !== initialReturnDate || 
      tripType !== initialTripType;
    
    if (hasChanged && origin && departureDate && (tripType === 'one-way' || returnDate)) {
      const timer = setTimeout(() => {
        onSearch({ origin, departureDate, returnDate, tripType });
      }, 800); // Slightly longer delay to avoid too many requests
      
      return () => clearTimeout(timer);
    }
  }, [origin, departureDate, returnDate, tripType, initialOrigin, initialDepartureDate, initialReturnDate, initialTripType, onSearch]);

  const selectAirport = (airportCode: string) => {
    setOrigin(airportCode);
    setSearchQuery(getAirportDisplay(airportCode));
    setShowDropdown(false);
  };

  return (
    <div className="bg-white border-b shadow-sm">
      <div className="max-w-7xl mx-auto px-4 py-2">
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-2 items-end">
          {/* Trip Type Toggle - Super Compact */}
          <div className="col-span-2 md:col-span-1">
            <label className="block text-xs font-medium text-gray-600 mb-1">Type</label>
            <div className="flex bg-gray-100 rounded-lg p-0.5">
              <button
                onClick={() => setTripType('one-way')}
                className={`flex-1 px-2 py-1 rounded text-xs font-medium transition-all ${
                  tripType === 'one-way'
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-600 hover:text-gray-800'
                }`}
              >
                {t.oneWay}
              </button>
              <button
                onClick={() => setTripType('round-trip')}
                className={`flex-1 px-2 py-1 rounded text-xs font-medium transition-all ${
                  tripType === 'round-trip'
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-600 hover:text-gray-800'
                }`}
              >
                {t.roundTrip}
              </button>
            </div>
          </div>

          {/* Origin Airport */}
          <div className={`relative col-span-2 ${tripType === 'round-trip' ? 'lg:col-span-1' : 'md:col-span-1'}`}>
            <label className="block text-xs font-medium text-gray-600 mb-1">
              <MapPin className="inline w-3 h-3 mr-1" />
              {t.origin}
            </label>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setShowDropdown(true);
              }}
              onFocus={() => setShowDropdown(true)}
              onBlur={() => setTimeout(() => setShowDropdown(false), 200)}
              placeholder={t.searchPlaceholder}
              className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none"
            />
            
            {showDropdown && (
              <div className="absolute z-20 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg max-h-40 overflow-y-auto">
                {filteredAirports.map(airport => (
                  <button
                    key={airport.code}
                    onClick={() => selectAirport(airport.code)}
                    className="w-full px-2 py-1.5 text-left hover:bg-blue-50 transition-colors text-xs"
                  >
                    <div className="font-semibold">{airport.city} ({airport.code})</div>
                    <div className="text-xs text-gray-500">{airport.country}</div>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Departure Date */}
          <div className={tripType === 'round-trip' ? 'col-span-1' : 'col-span-2 md:col-span-1'}>
            <label className="block text-xs font-medium text-gray-600 mb-1">
              <Calendar className="inline w-3 h-3 mr-1" />
              {t.departureDate}
            </label>
            <input
              type="date"
              value={departureDate}
              onChange={(e) => {
                setDepartureDate(e.target.value);
                // Auto-open return date picker for round-trip
                if (tripType === 'round-trip' && e.target.value && returnDateRef.current) {
                  setTimeout(() => {
                    returnDateRef.current?.focus();
                    returnDateRef.current?.showPicker?.();
                  }, 100);
                }
              }}
              min={minDate}
              className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none"
            />
          </div>

          {/* Return Date */}
          {tripType === 'round-trip' && (
            <div className="col-span-1">
              <label className="block text-xs font-medium text-gray-600 mb-1">
                <Calendar className="inline w-3 h-3 mr-1" />
                {t.returnDate}
              </label>
              <input
                ref={returnDateRef}
                type="date"
                value={returnDate}
                onChange={(e) => setReturnDate(e.target.value)}
                min={departureDate || minDate}
                className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none"
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}