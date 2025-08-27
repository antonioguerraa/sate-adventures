# Ryanair Anywhere Search - Complete Implementation

## Overview
This application provides a clean, focused interface for Ryanair's "fly anywhere" functionality. Users can search for round-trip flights from any Ryanair airport to all available destinations, sorted by total price.

## Features

### 🎯 Core Functionality
- **Anywhere Search**: Search flights from any origin to ALL destinations
- **Round-Trip Pricing**: Shows both outbound and return flights with total price
- **Smart Sorting**: Results automatically sorted by total trip price
- **Airport Autocomplete**: Quick search through 70+ Ryanair airports
- **Destination Images**: Visual representation for major destinations
- **Clean UI**: Minimalist design focused on the search experience

### 🛠 Technical Implementation

#### Frontend (`app/page.tsx`)
- Clean, single-page interface
- Airport autocomplete with search filtering
- Date pickers for outbound and return dates
- Animated loading states
- Card-based results display with flight details
- Responsive design for all screen sizes

#### Backend API (`app/api/explore/anywhere/route.ts`)
- POST endpoint for round-trip searches
- Integrates with Ryanair Python API
- Enriches data with airport information
- Adds destination images
- Returns sorted results by price

#### Python Integration
- `ryanair-search.py`: Basic flight search functionality
- `test-return-anywhere.py`: Round-trip anywhere search
- Uses official `ryanair-py` library
- Groups flights by destination
- Finds cheapest options

#### Data (`data/ryanair-airports.json`)
- Complete list of 70+ Ryanair airports
- Includes airport codes, names, cities, and countries
- Used for autocomplete and validation

## User Experience Flow

1. **Search Input**
   - User types airport code or city name
   - Autocomplete dropdown shows matching airports
   - User selects departure and return dates

2. **Search Animation**
   - Full-screen loading overlay
   - Animated plane icon
   - Progress messaging

3. **Results Display**
   - Grid of destination cards
   - Each card shows:
     - Destination image (when available)
     - City and country
     - Outbound flight details and price
     - Return flight details and price
     - Total trip price highlighted
   - Sorted by total price (cheapest first)

## API Endpoints

### POST `/api/explore/anywhere`
Request:
```json
{
  "origin": "ALC",
  "outboundDate": "2025-09-15",
  "returnDate": "2025-09-22"
}
```

Response:
```json
{
  "origin": "ALC",
  "originName": "Alicante, Spain",
  "destinations": [
    {
      "destination": "MAD",
      "city": "Madrid",
      "country": "Spain",
      "outboundPrice": 25.99,
      "returnPrice": 35.99,
      "totalPrice": 61.98,
      "outboundFlight": {
        "flightNumber": "FR1234",
        "departureTime": "2025-09-15T10:30:00Z",
        "arrivalTime": "2025-09-15T11:45:00Z"
      },
      "returnFlight": {
        "flightNumber": "FR5678",
        "departureTime": "2025-09-22T18:00:00Z",
        "arrivalTime": "2025-09-22T19:15:00Z"
      },
      "imageUrl": "https://images.unsplash.com/..."
    }
  ],
  "totalFound": 45
}
```

## Installation & Setup

1. **Install Python dependencies**:
   ```bash
   pip install ryanair-py
   ```

2. **Install Node dependencies**:
   ```bash
   npm install
   ```

3. **Run development server**:
   ```bash
   npm run dev
   ```

4. **Access the application**:
   Open http://localhost:3000

## Key Components

### UI Components
- **Airport Search**: Filterable dropdown with all Ryanair airports
- **Date Pickers**: HTML5 date inputs with validation
- **Search Button**: Animated loading state
- **Result Cards**: Image, flight details, and pricing
- **Loading Overlay**: Full-screen animated loader

### Styling
- Gradient backgrounds (blue to indigo)
- Card-based layout with shadows
- Hover effects and transitions
- Responsive grid system
- Clean typography

### Data Flow
1. User selects origin and dates
2. Frontend calls `/api/explore/anywhere`
3. API executes Python script
4. Python queries Ryanair API
5. Results processed and enriched
6. Sorted data returned to frontend
7. UI renders destination cards

## Performance Optimizations
- Results limited to top 50 destinations
- Images lazy-loaded from CDN
- Autocomplete filters locally
- Minimal API calls

## Future Enhancements
- [ ] Add price filtering
- [ ] Include flexible date search
- [ ] Add favorite destinations
- [ ] Implement booking deeplinks
- [ ] Add map visualization
- [ ] Include weather data
- [ ] Add travel recommendations

## Technologies Used
- **Frontend**: Next.js 15, React, TypeScript
- **Styling**: Tailwind CSS
- **Icons**: Lucide React
- **Backend**: Next.js API Routes
- **Python**: ryanair-py library
- **Date handling**: date-fns

## Notes
- Prices are in EUR
- Flight times are estimates for some routes
- Images use Unsplash for demo purposes
- Limited to Ryanair's network of airports