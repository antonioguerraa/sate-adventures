# Flight Search Engine

A modern web application for discovering cheap Ryanair flights from any origin to destinations across Europe. Built with Next.js 15, TypeScript, and real-time flight data integration.

## Features

- **One-way & Round-trip Search**: Search for single or return flights to anywhere
- **Interactive Destination Map**: Visualize all available destinations on an interactive map
- **Real-time Pricing**: Live flight prices directly from Ryanair API
- **Multi-language Support**: Interface available in multiple languages
- **Smart Airport Search**: Auto-complete search with all Ryanair airports
- **Direct Booking Links**: Click through to Ryanair to complete your booking
- **Responsive Design**: Optimized for desktop and mobile devices

## How It Works

1. **Select Trip Type**: Choose between one-way or round-trip
2. **Choose Origin**: Type to search from 200+ Ryanair airports
3. **Select Dates**: Pick your departure (and return) dates
4. **Search**: Click search to find flights to all available destinations
5. **Explore Results**: 
   - View destinations on an interactive map
   - See prices and flight times
   - Click to book directly on Ryanair

## Tech Stack

- **Frontend**: Next.js 15.4.5, React 19, TypeScript
- **Maps**: Leaflet with React-Leaflet
- **API Integration**: Python scripts using ryanair-py library
- **Styling**: Tailwind CSS 4
- **Icons**: Lucide React
- **Date Handling**: date-fns

## Getting Started

### Prerequisites

- Node.js 18+
- Python 3.x
- npm or yarn

### Installation

1. Clone the repository:
```bash
cd /workspaces/sate-adventures/code/flight-search-engine
```

2. Install Node dependencies:
```bash
npm install
```

3. Install Python dependencies:
```bash
pip install ryanair-py
```

4. Start the development server:
```bash
npm run dev
```

5. Open [http://localhost:3000](http://localhost:3000) in your browser

## Project Structure

```
flight-search-engine/
├── app/                    # Next.js app directory
│   ├── api/               # API routes
│   │   └── explore/       # Flight search endpoints
│   │       ├── anywhere/  # Round-trip search
│   │       └── one-way/   # One-way search
│   ├── results/           # Results page
│   └── page.tsx           # Home page
├── components/            # React components
│   ├── SimpleDestinationMap.tsx  # Interactive map
│   ├── QuickSearch.tsx           # Search bar
│   └── LanguageSelector.tsx     # Language switcher
├── lib/                   # Utilities
│   ├── providers/         # Flight data providers
│   └── translations.ts    # i18n support
├── data/                  # Static data
│   └── ryanair-airports.json
├── public/
│   └── airport-images/    # Destination images
└── Python Scripts
    ├── search-one-way-anywhere.py
    └── test-return-anywhere.py
```

## Data Sources

- **Airports**: 200+ Ryanair airports with IATA codes, cities, and countries
- **Flight Data**: Real-time from Ryanair API via ryanair-py
- **Images**: Curated destination images for major airports
- **Translations**: Multi-language support for UI elements

## API Integration

The application uses Python scripts to interface with Ryanair's API:

1. **One-way Search**: Finds all destinations with available flights
2. **Round-trip Search**: Discovers return flight combinations
3. **Price Aggregation**: Shows cheapest available options
4. **Real-time Data**: Fetches current availability and pricing

## API Endpoints

### POST /api/explore/one-way
Search one-way flights to anywhere:
```json
{
  "origin": "MAD",
  "departureDate": "2024-03-15"
}
```

### POST /api/explore/anywhere
Search round-trip flights to anywhere:
```json
{
  "origin": "BCN",
  "outboundDate": "2024-03-15",
  "returnDate": "2024-03-22"
}
```

## Environment Variables

Create a `.env.local` file:
```bash
# No environment variables required for basic functionality
# The app uses Python scripts to fetch data directly
```

## Development

```bash
# Run development server with Turbopack
npm run dev

# Build for production
npm run build

# Run production server
npm start

# Lint code
npm run lint

# Run tests
npm test
```

## Deployment

The application can be deployed to any Node.js hosting platform:

1. **Vercel** (Recommended):
   - Push to GitHub
   - Import project in Vercel
   - Deploy automatically

2. **Requirements**:
   - Node.js 18+
   - Python 3.x with ryanair-py
   - 512MB+ RAM

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
