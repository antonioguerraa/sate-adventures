'use client';

import { useEffect, useRef, useState } from 'react';
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

interface SimpleDestinationMapProps {
  origin: string;
  destinations: Destination[];
  hoveredDestination: string | null;
  onDestinationHover: (destination: string | null) => void;
  onDestinationClick: (destination: string) => void;
}

// Fix for default markers
delete (L.Icon.Default.prototype as any)._getIconUrl;

// Extended coordinates for all Ryanair airports
const airportCoordinates: Record<string, [number, number]> = {
  'MAD': [40.4168, -3.7038],
  'BCN': [41.3851, 2.1734],
  'SVQ': [37.3891, -5.9845],
  'VLC': [39.4699, -0.3763],
  'AGP': [36.7213, -4.4214],
  'ALC': [38.3452, -0.4810],
  'PMI': [39.5696, 2.6502],
  'IBZ': [38.9067, 1.4821],
  'DUB': [53.3498, -6.2603],
  'STN': [51.8787, 0.2350],
  'LGW': [51.1537, -0.1821],
  'LTN': [51.8763, -0.3717],
  'MAN': [53.4808, -2.2426],
  'EDI': [55.9533, -3.1883],
  'BER': [52.5200, 13.4050],
  'MUC': [48.1351, 11.5820],
  'FRA': [50.1109, 8.6821],
  'CGN': [50.9375, 6.9603],
  'HAM': [53.5511, 9.9937],
  'AMS': [52.3676, 4.9041],
  'BRU': [50.8503, 4.3517],
  'CRL': [50.4592, 4.4538],
  'CDG': [49.0097, 2.5479],
  'BVA': [49.4544, 2.1130],
  'FCO': [41.9028, 12.4964],
  'CIA': [41.7994, 12.5950],
  'MXP': [45.4642, 9.1900],
  'BGY': [45.6739, 9.7040],
  'VCE': [45.4408, 12.3155],
  'TSF': [45.6484, 12.1944],
  'NAP': [40.8518, 14.2681],
  'BLQ': [44.5308, 11.2889],
  'PSA': [43.6839, 10.3926],
  'CAG': [39.2515, 9.0543],
  'PMO': [38.1759, 13.0910],
  'CTA': [37.4668, 15.0664],
  'BRI': [40.8986, 17.0803],
  'TRN': [45.2008, 7.6497],
  'VRN': [45.3957, 10.8885],
  'TRS': [45.8275, 13.4722],
  'ATH': [37.9364, 23.9445],
  'SKG': [40.5197, 22.9709],
  'CHQ': [35.5396, 24.1497],
  'CFU': [39.6019, 19.9117],
  'RHO': [36.4054, 28.0862],
  'KGS': [36.7933, 27.0917],
  'LIS': [38.7813, -9.1359],
  'OPO': [41.2481, -8.6814],
  'FAO': [37.0144, -7.9658],
  'FNC': [32.6978, -16.7745],
  'VIE': [48.1103, 16.5697],
  'PRG': [50.1008, 14.2600],
  'BTS': [48.1703, 17.2127],
  'BUD': [47.4298, 19.2611],
  'WAW': [52.1657, 20.9671],
  'WMI': [52.4511, 20.6519],
  'KRK': [50.0777, 19.7848],
  'KTW': [50.4743, 19.0800],
  'POZ': [52.4210, 16.8263],
  'WRO': [51.1027, 16.8858],
  'GDN': [54.3775, 18.4661],
  'RZE': [50.1100, 22.0190],
  'CPH': [55.6181, 12.6561],
  'ARN': [59.6519, 17.9186],
  'GOT': [57.6628, 12.2798],
  'OSL': [60.1939, 11.1004],
  'HEL': [60.3172, 24.9633],
  'TLL': [59.4133, 24.8328],
  'RIX': [56.9236, 23.9711],
  'VNO': [54.6341, 25.2858],
  'KUN': [54.9638, 24.0848],
  'SOF': [42.6951, 23.4064],
  'BOJ': [42.5697, 27.5153],
  'OTP': [44.5711, 26.0858],
  'CLJ': [46.7852, 23.6862],
  'IAS': [47.1785, 27.6206],
  'ZAG': [45.7429, 16.0688],
  'SPU': [43.5389, 16.2980],
  'DBV': [42.5614, 18.2682],
  'ZAD': [44.1083, 15.3467],
  'LUX': [49.6233, 6.2044],
  'ZRH': [47.4647, 8.5492],
  'GVA': [46.2381, 6.1089],
  'BSL': [47.5896, 7.5299],
  'BRE': [53.0475, 8.7867],
  'NUE': [49.4987, 11.0680],
  'FKB': [48.7794, 8.0805],
  'FMM': [47.9889, 10.2396],
  'HHN': [49.9487, 7.2639],
  'EIN': [51.4500, 5.3745],
  'MRS': [43.4393, 5.2214],
  'NCE': [43.6584, 7.2158],
  'TLS': [43.6294, 1.3638],
  'BIQ': [43.4684, -1.5311],
  'NTE': [47.1569, -1.6078],
  'LRH': [46.1791, -1.1953],
  'BVE': [45.1505, 1.4692],
  'RDZ': [44.4079, 2.4827],
  'CCF': [43.2022, 2.3064],
  'FNI': [44.0763, 4.2892],
  'LDE': [43.1787, 0.0064],
  'BRS': [51.3827, -2.7191],
  'BOH': [50.7800, -1.8425],
  'BHX': [52.4539, -1.7480],
  'LPL': [53.3336, -2.8497],
  'LBA': [53.8659, -1.6605],
  'NCL': [55.0375, -1.6917],
  'EMA': [52.8311, -1.3278],
  'CWL': [51.3967, -3.3433],
  'GLA': [55.8719, -4.4331],
  'NQY': [50.4406, -4.9954],
  'ORK': [51.8413, -8.4911],
  'KIR': [52.1872, -9.5238],
  'TFS': [28.0445, -16.5725],
  'TFN': [28.4827, -16.3415],
  'ACE': [28.9455, -13.6052],
  'FUE': [28.4527, -13.8638],
  'LPA': [27.9319, -15.3866],
  'RAK': [31.6069, -8.0363],
  'AGA': [30.3250, -9.4131],
  'RBA': [34.0515, -6.7516],
  'TLV': [32.0114, 34.8867],
  'DLM': [36.7128, 28.7929],
  'AYT': [36.8987, 30.8005],
  'KSC': [48.6632, 21.2411],
  'SDR': [43.3965, -3.2056],
  'RMU': [37.8030, -0.8122],
  'REU': [41.1476, 1.1672],
  'GRO': [41.9011, 2.7606],
  'MAH': [39.8626, 4.2185],
  'SUF': [38.2824, 15.6470],
  'OLB': [40.8987, 9.5176],
  'PLQ': [55.9729, 21.0937]
};

export default function SimpleDestinationMap({ 
  origin, 
  destinations, 
  hoveredDestination, 
  onDestinationHover,
  onDestinationClick 
}: SimpleDestinationMapProps) {
  const mapRef = useRef<L.Map | null>(null);
  const markersRef = useRef<{ [key: string]: L.CircleMarker }>({});
  const flightPathRef = useRef<L.Polyline | null>(null);
  const tooltipRef = useRef<L.Tooltip | null>(null);

  const originCoords = airportCoordinates[origin] || [48.8566, 2.3522];

  useEffect(() => {
    if (!mapRef.current) {
      // Initialize map with simple styling
      mapRef.current = L.map('map', {
        zoomControl: false,
        attributionControl: true
      }).setView([48.8566, 2.3522], 4);

      // Add CartoDB Voyager tile layer for better colors and light blue water
      L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
        attribution: '© OpenStreetMap contributors © CARTO',
        subdomains: 'abcd',
        maxZoom: 19
      }).addTo(mapRef.current);

      // Add zoom control to top right
      L.control.zoom({
        position: 'topright'
      }).addTo(mapRef.current);
    }

    // Clear existing markers
    Object.values(markersRef.current).forEach(marker => marker.remove());
    markersRef.current = {};

    // Add origin marker
    if (originCoords) {
      L.circleMarker(originCoords, {
        radius: 8,
        fillColor: '#3b82f6',
        color: '#1e40af',
        weight: 2,
        opacity: 1,
        fillOpacity: 1
      }).addTo(mapRef.current!);
    }

    // Add destination markers
    destinations.forEach(dest => {
      const coords = airportCoordinates[dest.destination] || dest.coordinates;
      if (coords) {
        const marker = L.circleMarker(coords, {
          radius: 6,
          fillColor: '#64748b',
          color: '#ffffff',
          weight: 2,
          opacity: 1,
          fillOpacity: 1,
          className: 'destination-marker'
        })
        .addTo(mapRef.current!)
        .on('mouseover', function(e) {
          onDestinationHover(dest.destination);
        })
        .on('mouseout', function(e) {
          onDestinationHover(null);
        })
        .on('click', () => onDestinationClick(dest.destination));

        markersRef.current[dest.destination] = marker;
      }
    });

    // Fit bounds to show all markers
    if (destinations.length > 0) {
      const bounds = L.latLngBounds([originCoords]);
      destinations.forEach(dest => {
        const coords = airportCoordinates[dest.destination] || dest.coordinates;
        if (coords) bounds.extend(coords);
      });
      mapRef.current.fitBounds(bounds, { padding: [50, 50] });
    }
  }, [destinations, origin, onDestinationHover, onDestinationClick, originCoords]);

  // Handle hover effects
  useEffect(() => {
    // Remove previous flight path
    if (flightPathRef.current) {
      flightPathRef.current.remove();
      flightPathRef.current = null;
    }

    // Remove previous tooltip
    if (tooltipRef.current) {
      tooltipRef.current.remove();
      tooltipRef.current = null;
    }

    // Reset all markers
    Object.values(markersRef.current).forEach(marker => {
      marker.setStyle({
        radius: 6,
        fillColor: '#64748b',
        color: '#ffffff',
        weight: 2,
        fillOpacity: 1
      });
    });

    if (hoveredDestination && markersRef.current[hoveredDestination]) {
      const marker = markersRef.current[hoveredDestination];
      const destCoords = airportCoordinates[hoveredDestination];
      
      // Highlight the hovered marker without changing size (which causes movement)
      marker.setStyle({
        radius: 6, // Keep same radius to prevent movement
        fillColor: '#ef4444', // Red color
        color: '#ffffff',
        weight: 3,
        fillOpacity: 1
      });

      // Draw flight path
      if (originCoords && destCoords && mapRef.current) {
        flightPathRef.current = L.polyline([originCoords, destCoords], {
          color: '#3b82f6',
          weight: 2,
          opacity: 0.8,
          dashArray: '10, 10',
          className: 'flight-path'
        }).addTo(mapRef.current);
      }

      // Show tooltip with destination info
      const dest = destinations.find(d => d.destination === hoveredDestination);
      if (dest && destCoords && mapRef.current) {
        tooltipRef.current = L.tooltip({
          permanent: true,
          direction: 'top',
          offset: [0, -10],
          className: 'custom-tooltip'
        })
        .setLatLng(destCoords)
        .setContent(`
          <div style="text-align: center; padding: 4px;">
            <strong>${dest.city}</strong><br/>
            <span style="color: #059669; font-size: 16px; font-weight: bold;">
              €${dest.price || dest.totalPrice}
            </span>
          </div>
        `)
        .addTo(mapRef.current);
      }
    }
  }, [hoveredDestination, destinations, originCoords]);

  return (
    <>
      <style jsx global>{`
        #map {
          border-radius: 12px;
          overflow: hidden;
        }
        .leaflet-container {
          font-family: inherit;
          background: #a8daef;
        }
        .destination-marker {
          cursor: pointer;
          transition: all 0.2s ease;
        }
        .flight-path {
          animation: dash 20s linear infinite;
        }
        @keyframes dash {
          to {
            stroke-dashoffset: -1000;
          }
        }
        .custom-tooltip {
          background: white;
          border: 2px solid #3b82f6;
          border-radius: 8px;
          box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
          font-size: 14px;
          padding: 0;
        }
        .custom-tooltip::before {
          border-top-color: #3b82f6;
        }
        .leaflet-control-zoom {
          border: none !important;
          box-shadow: 0 2px 6px rgba(0,0,0,0.1) !important;
        }
        .leaflet-control-zoom a {
          background: white !important;
          color: #333 !important;
        }
      `}</style>
      <div id="map" className="w-full h-full" />
    </>
  );
}