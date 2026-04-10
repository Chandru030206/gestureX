import React from 'react';

const AslHelloSVG = () => (
    <svg width="120" height="120" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
        {/* Right Hand - Palm Facing Viewer */}
        <path 
            d="M70 150 L65 110 Q60 100 70 95 L85 90 
               Q90 50 100 50 Q110 50 115 90
               Q120 40 130 40 Q140 40 145 90
               Q150 45 160 45 Q170 45 170 100
               Q175 60 185 60 Q195 60 190 120
               L180 160 Q150 180 80 170 Z" 
            fill="#e8c9a0" 
            stroke="#b8860b" 
            strokeWidth="2" 
            transform="rotate(-15 100 110)"
        />
        
        {/* Wave Motion Lines */}
        <path d="M170 70 Q185 75 180 90" fill="none" stroke="#b8860b" strokeWidth="2" strokeLinecap="round" />
        <path d="M180 50 Q195 60 190 80" fill="none" stroke="#b8860b" strokeWidth="2" strokeLinecap="round" opacity="0.6" />
        <path d="M190 30 Q205 45 200 70" fill="none" stroke="#b8860b" strokeWidth="2" strokeLinecap="round" opacity="0.3" />
    </svg>
);

export default AslHelloSVG;
