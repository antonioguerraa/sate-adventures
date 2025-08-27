'use client';

import { useState, useEffect, useRef } from 'react';
import { format } from 'date-fns';
import { Plane, Calendar, MapPin, ToggleLeft, ToggleRight } from 'lucide-react';
import { useRouter } from 'next/navigation';
import RyanairAirports from '@/data/ryanair-airports.json';
import { translations, Language } from '@/lib/translations';
import LanguageSelector from '@/components/LanguageSelector';

export default function Home() {
  const router = useRouter();
  const [origin, setOrigin] = useState('');
  const [filteredAirports, setFilteredAirports] = useState(RyanairAirports.airports);
  const [showDropdown, setShowDropdown] = useState(false);
  const [departureDate, setDepartureDate] = useState('');
  const [returnDate, setReturnDate] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [tripType, setTripType] = useState<'round-trip' | 'one-way'>('round-trip');
  const [language, setLanguage] = useState<Language>('en');
  const [minDate, setMinDate] = useState('');
  const returnDateRef = useRef<HTMLInputElement>(null);
  
  const t = translations[language];

  // Set min date on client side only to avoid hydration issues
  useEffect(() => {
    setMinDate(format(new Date(), 'yyyy-MM-dd'));
  }, []);

  // Filter airports based on search
  useEffect(() => {
    if (searchQuery) {
      const filtered = RyanairAirports.airports.filter(
        airport => 
          airport.code.toLowerCase().includes(searchQuery.toLowerCase()) ||
          airport.city.toLowerCase().includes(searchQuery.toLowerCase()) ||
          airport.country.toLowerCase().includes(searchQuery.toLowerCase())
      );
      setFilteredAirports(filtered);
    } else {
      setFilteredAirports(RyanairAirports.airports);
    }
  }, [searchQuery]);

  const handleSearch = () => {
    if (!origin || !departureDate) return;
    if (tripType === 'round-trip' && !returnDate) return;

    // Navigate to results page with search parameters
    const params = new URLSearchParams({
      tripType,
      origin,
      departureDate,
      ...(tripType === 'round-trip' && { returnDate })
    });

    router.push(`/results?${params.toString()}`);
  };

  const selectAirport = (airportCode: string, airportCity: string) => {
    setOrigin(airportCode);
    setSearchQuery(`${airportCity} (${airportCode})`);
    setShowDropdown(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-600 via-blue-500 to-indigo-600">
      {/* Language Selector */}
      <div className="absolute top-4 right-4 z-10">
        <LanguageSelector 
          currentLanguage={language} 
          onLanguageChange={setLanguage}
          isWhite={true}
        />
      </div>
      
      {/* Hero Section */}
      <div className="pt-20 pb-32">
        <div className="max-w-4xl mx-auto px-4 text-center">
          <div className="mb-8">
            <img 
              src="/logo.svg" 
              alt="Cool Donkey" 
              className="w-24 h-24 mx-auto mb-4 drop-shadow-2xl hover:scale-110 transition-transform"
            />
            <h1 className="text-5xl md:text-6xl font-bold text-white mb-4">
              {t.title}
            </h1>
            <p className="text-xl text-blue-100">
              {t.subtitle}
            </p>
          </div>

          {/* Search Form */}
          <div className="bg-white rounded-2xl shadow-2xl p-6 md:p-8 space-y-5 max-w-3xl mx-auto">
            {/* Trip Type Toggle */}
            <div className="flex justify-center mb-6">
              <div className="inline-flex bg-gray-100 rounded-lg p-1">
                <button
                  onClick={() => setTripType('one-way')}
                  className={`px-6 py-2 rounded-md font-medium transition-all ${
                    tripType === 'one-way'
                      ? 'bg-blue-600 text-white shadow-md'
                      : 'text-gray-600 hover:text-gray-800'
                  }`}
                >
                  {t.oneWay}
                </button>
                <button
                  onClick={() => setTripType('round-trip')}
                  className={`px-6 py-2 rounded-md font-medium transition-all ${
                    tripType === 'round-trip'
                      ? 'bg-blue-600 text-white shadow-md'
                      : 'text-gray-600 hover:text-gray-800'
                  }`}
                >
                  {t.roundTrip}
                </button>
              </div>
            </div>

            {/* Airport Input */}
            <div className="relative">
              <label className="block text-left text-sm font-medium text-gray-700 mb-2">
                <MapPin className="inline w-4 h-4 mr-1" />
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
                placeholder={t.searchPlaceholder}
                className="w-full px-4 py-3 text-lg border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none"
              />
              
              {/* Airport Dropdown */}
              {showDropdown && searchQuery && (
                <div className="absolute z-10 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg max-h-60 overflow-y-auto">
                  {filteredAirports.map(airport => (
                    <button
                      key={airport.code}
                      onClick={() => selectAirport(airport.code, airport.city)}
                      className="w-full px-4 py-3 text-left hover:bg-blue-50 transition-colors border-b border-gray-100 last:border-b-0"
                    >
                      <div className="font-semibold">{airport.city} ({airport.code})</div>
                      <div className="text-sm text-gray-600">{airport.name}, {airport.country}</div>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Date Inputs */}
            <div className="space-y-4">
              <div>
                <label className="block text-left text-sm font-medium text-gray-700 mb-2">
                  <Calendar className="inline w-4 h-4 mr-1" />
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
                  className="w-full px-3 py-2 text-base border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none"
                />
              </div>
              {tripType === 'round-trip' && (
                <div>
                  <label className="block text-left text-sm font-medium text-gray-700 mb-2">
                    <Calendar className="inline w-4 h-4 mr-1" />
                    {t.returnDate}
                  </label>
                  <input
                    ref={returnDateRef}
                    type="date"
                    value={returnDate}
                    onChange={(e) => setReturnDate(e.target.value)}
                    min={departureDate || minDate}
                    className="w-full px-3 py-2 text-base border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none"
                  />
                </div>
              )}
            </div>

            {/* Search Button */}
            <button
              onClick={handleSearch}
              disabled={!origin || !departureDate || (tripType === 'round-trip' && !returnDate)}
              className="w-full py-4 bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-bold text-lg rounded-lg hover:from-blue-700 hover:to-indigo-700 disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed transition-all transform hover:scale-105 active:scale-95"
            >
              {tripType === 'one-way' ? t.searchButtonOneWay : t.searchButtonRoundTrip}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}