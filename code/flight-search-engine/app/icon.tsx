import { ImageResponse } from 'next/og'

export const runtime = 'edge'

export const size = {
  width: 32,
  height: 32,
}

export const contentType = 'image/png'

export default function Icon() {
  return new ImageResponse(
    (
      <svg
        width="32"
        height="32"
        viewBox="0 0 100 100"
        xmlns="http://www.w3.org/2000/svg"
      >
        {/* Donkey head shape */}
        <ellipse cx="50" cy="55" rx="20" ry="25" fill="#8b7355"/>
        {/* Ears */}
        <ellipse cx="38" cy="35" rx="8" ry="15" fill="#8b7355" transform="rotate(-20 38 35)"/>
        <ellipse cx="62" cy="35" rx="8" ry="15" fill="#8b7355" transform="rotate(20 62 35)"/>
        {/* Sunglasses */}
        <path d="M30 48 L70 48" stroke="black" strokeWidth="2" fill="none"/>
        <rect x="32" y="45" width="15" height="10" rx="3" fill="black" opacity="0.8"/>
        <rect x="53" y="45" width="15" height="10" rx="3" fill="black" opacity="0.8"/>
        {/* Nose */}
        <ellipse cx="50" cy="65" rx="12" ry="8" fill="#6b5d4f"/>
        <circle cx="46" cy="65" r="2" fill="black"/>
        <circle cx="54" cy="65" r="2" fill="black"/>
        {/* Cigarette */}
        <rect x="62" y="62" width="20" height="4" rx="1" fill="white"/>
        <rect x="78" y="62" width="4" height="4" fill="#ff6b35"/>
        {/* Smoke */}
        <circle cx="85" cy="60" r="3" fill="gray" opacity="0.5"/>
        <circle cx="87" cy="56" r="2.5" fill="gray" opacity="0.4"/>
        <circle cx="88" cy="52" r="2" fill="gray" opacity="0.3"/>
      </svg>
    ),
    {
      ...size,
    }
  )
}