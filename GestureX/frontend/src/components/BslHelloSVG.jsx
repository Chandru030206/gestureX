import React from 'react';

const BslHelloSVG = () => (
    <svg width="120" height="120" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
        {/* Left Hand */}
        <g transform="translate(60, 100) scale(0.6) rotate(15)">
            <path 
                d="M0 80 L-10 40 Q-15 30 -5 20 L10 10 
                   Q10 -40 25 -40 Q40 -40 40 10
                   Q50 -50 65 -50 Q80 -50 80 10
                   Q90 -45 105 -45 Q120 -45 120 20
                   Q135 -30 150 -30 Q165 -30 160 40
                   L150 90 Q120 120 20 110 Z" 
                fill="#e8c9a0" 
                stroke="#b8860b" 
                strokeWidth="3"
            />
        </g>

        {/* Right Hand (Mirrored) */}
        <g transform="translate(140, 100) scale(-0.6, 0.6) rotate(15)">
            <path 
                d="M0 80 L-10 40 Q-15 30 -5 20 L10 10 
                   Q10 -40 25 -40 Q40 -40 40 10
                   Q50 -50 65 -50 Q80 -50 80 10
                   Q90 -45 105 -45 Q120 -45 120 20
                   Q135 -30 150 -30 Q165 -30 160 40
                   L150 90 Q120 120 20 110 Z" 
                fill="#e8c9a0" 
                stroke="#b8860b" 
                strokeWidth="3"
            />
        </g>
    </svg>
);

export default BslHelloSVG;
