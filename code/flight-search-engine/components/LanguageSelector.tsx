'use client';

import { useState, useRef, useEffect } from 'react';
import { languages, Language } from '@/lib/translations';

interface LanguageSelectorProps {
  currentLanguage: Language;
  onLanguageChange: (lang: Language) => void;
  isWhite?: boolean;
}

export default function LanguageSelector({ currentLanguage, onLanguageChange, isWhite = false }: LanguageSelectorProps) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  
  const currentLang = languages.find(l => l.code === currentLanguage) || languages[0];

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-colors ${
          isWhite 
            ? 'bg-white/20 hover:bg-white/30 text-white backdrop-blur-sm' 
            : 'bg-white hover:bg-gray-50 text-gray-800 shadow-md'
        }`}
      >
        <span className="text-xl">{currentLang.flag}</span>
        <span className="font-medium">{currentLang.name}</span>
        <svg 
          className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`} 
          fill="none" 
          stroke="currentColor" 
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {isOpen && (
        <div className={`absolute right-0 mt-2 w-48 rounded-lg shadow-lg overflow-hidden z-50 ${
          isWhite ? 'bg-white' : 'bg-white'
        }`}>
          {languages.map((lang) => (
            <button
              key={lang.code}
              onClick={() => {
                onLanguageChange(lang.code);
                setIsOpen(false);
              }}
              className={`flex items-center gap-3 w-full px-4 py-3 text-left hover:bg-gray-50 transition-colors ${
                lang.code === currentLanguage ? 'bg-blue-50 text-blue-600' : 'text-gray-800'
              }`}
            >
              <span className="text-xl">{lang.flag}</span>
              <span className="font-medium">{lang.name}</span>
              {lang.code === currentLanguage && (
                <svg className="w-4 h-4 ml-auto" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}