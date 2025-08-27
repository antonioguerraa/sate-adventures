'use client';

import { useEffect, useRef } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

interface Destination {
  destination: string;
  city: string;
  country: string;
  price?: number;
  totalPrice?: number;
  coordinates?: [number, number];
}

interface DestinationMapProps {
  origin: string;
  destinations: Destination[];
  selectedDestination: string | null;
  onDestinationClick: (destination: string) => void;
}

// Fix for default markers
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
});

export default function DestinationMap({ origin, destinations, selectedDestination, onDestinationClick }: DestinationMapProps) {
  const mapRef = useRef<L.Map | null>(null);
  const markersRef = useRef<{ [key: string]: L.Marker }>({});

  useEffect(() => {
    if (!mapRef.current) {
      // Initialize map
      mapRef.current = L.map('map').setView([48.8566, 2.3522], 5); // Center on Europe

      // Add tile layer
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
      }).addTo(mapRef.current);
    }

    // Clear existing markers
    Object.values(markersRef.current).forEach(marker => marker.remove());
    markersRef.current = {};

    // Add markers for destinations
    destinations.forEach(dest => {
      if (dest.coordinates) {
        const isSelected = selectedDestination === dest.destination;
        
        // Create custom icon
        const icon = L.divIcon({
          className: 'custom-marker',
          html: `
            <div class="${isSelected ? 'selected-marker' : 'normal-marker'}">
              <div class="marker-price">€${dest.price || dest.totalPrice}</div>
              <div class="marker-city">${dest.destination}</div>
            </div>
          `,
          iconSize: [80, 40],
          iconAnchor: [40, 40],
        });

        const marker = L.marker(dest.coordinates, { icon })
          .addTo(mapRef.current!)
          .on('click', () => onDestinationClick(dest.destination));

        markersRef.current[dest.destination] = marker;

        // Add popup
        marker.bindPopup(`
          <div class="text-center">
            <strong>${dest.city}</strong><br/>
            ${dest.country}<br/>
            <span class="text-green-600 font-bold">€${dest.price || dest.totalPrice}</span>
          </div>
        `);
      }
    });

    // Fit bounds to show all markers
    if (destinations.length > 0 && Object.keys(markersRef.current).length > 0) {
      const group = L.featureGroup(Object.values(markersRef.current));
      mapRef.current.fitBounds(group.getBounds().pad(0.1));
    }
  }, [destinations, selectedDestination, onDestinationClick]);

  // Highlight selected destination
  useEffect(() => {
    if (selectedDestination && markersRef.current[selectedDestination]) {
      markersRef.current[selectedDestination].openPopup();
    }
  }, [selectedDestination]);

  return (
    <>
      <style jsx global>{`
        .custom-marker {
          background: none;
          border: none;
        }
        .normal-marker {
          background: white;
          border: 2px solid #3b82f6;
          border-radius: 8px;
          padding: 4px 8px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.2);
          cursor: pointer;
          transition: all 0.2s;
        }
        .normal-marker:hover {
          transform: scale(1.1);
          box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }
        .selected-marker {
          background: #3b82f6;
          color: white;
          border: 2px solid #1e40af;
          border-radius: 8px;
          padding: 4px 8px;
          box-shadow: 0 4px 8px rgba(0,0,0,0.3);
          cursor: pointer;
          transform: scale(1.1);
        }
        .marker-price {
          font-weight: bold;
          font-size: 14px;
          white-space: nowrap;
        }
        .marker-city {
          font-size: 11px;
          opacity: 0.9;
          white-space: nowrap;
        }
      `}</style>
      <div id="map" className="w-full h-full rounded-lg" />
    </>
  );
}