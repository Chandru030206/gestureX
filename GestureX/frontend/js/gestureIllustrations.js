/**
 * GestureX Duo - SVG Illustration Library
 * Style: Minimalist Line Art, Pastel Colors
 * Task: Human figure demonstrations of sign language
 */

const COLORS = {
    skin: "#F5E6D8",
    outline: "#C8A882",
    clothing: "#C8D8A0",
    hair: "#B8A898",
    text: "#555555"
};

const BASE_FIGURE = (sign, armPathLeft, armPathRight) => `
<svg viewBox="0 0 200 240" xmlns="http://www.w3.org/2000/svg">
    <!-- Background -->
    <rect width="200" height="240" fill="#FFFFFF" />
    
    <!-- Torso -->
    <path d="M60 200 Q100 210 140 200 L145 130 Q100 120 55 130 Z" fill="${COLORS.clothing}" stroke="${COLORS.outline}" stroke-width="2" />
    
    <!-- Head & Hair -->
    <ellipse cx="100" cy="70" rx="35" ry="40" fill="${COLORS.skin}" stroke="${COLORS.outline}" stroke-width="2" />
    <path d="M65 70 Q65 30 100 30 Q135 30 135 70 Q135 50 100 50 Q65 50 65 70" fill="${COLORS.hair}" />
    
    <!-- Face details -->
    <path d="M90 75 Q100 80 110 75" fill="none" stroke="${COLORS.outline}" stroke-width="1.5" /> <!-- Mouth -->
    
    <!-- ARMS & HANDS (Sign Specific) -->
    ${armPathLeft || ''}
    ${armPathRight || ''}

    <!-- Word Label -->
    <text x="100" y="230" font-family="Arial, sans-serif" font-size="14" fill="${COLORS.text}" text-anchor="middle" font-weight="bold">${sign.toUpperCase()}</text>
</svg>
`;

export const GESTURE_ILLUSTRATIONS = {
    HELLO: BASE_FIGURE("HELLO", 
        `<path d="M140 160 Q170 140 160 80 L180 60" fill="none" stroke="${COLORS.outline}" stroke-width="6" stroke-linecap="round" /> <!-- Wave Arm -->
         <circle cx="170" cy="50" r="12" fill="${COLORS.skin}" stroke="${COLORS.outline}" stroke-width="2" /> <!-- Open Palm -->
         <path d="M155 30 Q170 20 185 30" fill="none" stroke="${COLORS.outline}" stroke-dasharray="4,4" /> <!-- Wave Motion -->`,
        `<path d="M60 160 Q30 180 40 200" fill="none" stroke="${COLORS.outline}" stroke-width="6" stroke-linecap="round" />`
    ),

    THANK_YOU: BASE_FIGURE("THANK YOU",
        `<path d="M100 110 Q120 140 150 140" fill="none" stroke="${COLORS.outline}" stroke-width="6" stroke-linecap="round" /> <!-- Chin to Outward -->
         <circle cx="100" cy="100" r="10" fill="${COLORS.skin}" stroke="${COLORS.outline}" stroke-width="2" /> <!-- Hand at chin -->
         <path d="M110 100 Q140 100 160 120" fill="none" stroke="${COLORS.outline}" stroke-dasharray="4,4" />`,
        `<path d="M60 160 Q40 180 40 200" fill="none" stroke="${COLORS.outline}" stroke-width="6" stroke-linecap="round" />`
    ),

    SORRY: BASE_FIGURE("SORRY",
        `<path d="M100 155 Q120 155 120 170" fill="none" stroke="${COLORS.outline}" stroke-width="6" stroke-linecap="round" /> 
         <circle cx="100" cy="165" r="15" fill="${COLORS.skin}" stroke="${COLORS.outline}" stroke-width="2" /> <!-- Fist -->
         <path d="M80 165 Q100 145 120 165 Q100 185 80 165" fill="none" stroke="${COLORS.outline}" stroke-dasharray="3,3" /> <!-- Circle motion -->`,
        `<path d="M60 160 Q40 180 40 200" fill="none" stroke="${COLORS.outline}" stroke-width="6" stroke-linecap="round" />`
    ),

    YES: BASE_FIGURE("YES",
        `<path d="M140 160 Q160 180 160 200" fill="none" stroke="${COLORS.outline}" stroke-width="6" stroke-linecap="round" />
         <circle cx="100" cy="115" r="12" fill="${COLORS.skin}" stroke="${COLORS.outline}" stroke-width="2" /> <!-- Fist nodding -->
         <path d="M100 100 L100 130" fill="none" stroke="${COLORS.outline}" stroke-dasharray="2,2" />`,
        `<path d="M60 160 Q40 180 40 200" fill="none" stroke="${COLORS.outline}" stroke-width="6" stroke-linecap="round" />`
    ),

    NO: BASE_FIGURE("NO",
        `<path d="M140 160 Q160 140 150 110" fill="none" stroke="${COLORS.outline}" stroke-width="6" stroke-linecap="round" />
         <circle cx="150" cy="100" r="10" fill="${COLORS.skin}" stroke="${COLORS.outline}" stroke-width="2" /> <!-- Two fingers -->
         <path d="M165 90 Q175 100 165 110" fill="none" stroke="${COLORS.outline}" stroke-dasharray="2,2" />`,
        `<path d="M60 160 Q40 180 40 200" fill="none" stroke="${COLORS.outline}" stroke-width="6" stroke-linecap="round" />`
    ),

    STOP: BASE_FIGURE("STOP",
        `<path d="M140 160 Q170 140 170 90" fill="none" stroke="${COLORS.outline}" stroke-width="6" stroke-linecap="round" />
         <rect x="160" y="60" width="20" height="30" rx="5" fill="${COLORS.skin}" stroke="${COLORS.outline}" stroke-width="2" /> <!-- Flat Palm -->`,
        `<path d="M60 160 Q40 180 40 200" fill="none" stroke="${COLORS.outline}" stroke-width="6" stroke-linecap="round" />`
    ),

    PLEASE: BASE_FIGURE("PLEASE",
        `<path d="M100 150 Q130 150 130 170" fill="none" stroke="${COLORS.outline}" stroke-width="6" stroke-linecap="round" />
         <circle cx="100" cy="160" r="18" fill="${COLORS.skin}" stroke="${COLORS.outline}" stroke-width="2" /> <!-- Flat hand on chest -->`,
        `<path d="M60 160 Q40 180 40 200" fill="none" stroke="${COLORS.outline}" stroke-width="6" stroke-linecap="round" />`
    ),

    HELP: BASE_FIGURE("HELP",
        `<path d="M100 180 Q100 150 100 120" fill="none" stroke="${COLORS.outline}" stroke-width="8" stroke-linecap="round" /> <!-- Lifting motion -->
         <circle cx="100" cy="130" r="12" fill="${COLORS.skin}" stroke="${COLORS.outline}" stroke-width="2" /> <!-- Fist on palm -->
         <rect x="80" y="145" width="40" height="10" rx="5" fill="${COLORS.skin}" stroke="${COLORS.outline}" stroke-width="2" />`,
        `<path d="M60 180 Q40 200 40 220" fill="none" stroke="${COLORS.outline}" stroke-width="6" stroke-linecap="round" />`
    ),

    GOOD: BASE_FIGURE("GOOD",
        `<path d="M100 110 Q140 130 140 180" fill="none" stroke="${COLORS.outline}" stroke-width="6" stroke-linecap="round" />
         <circle cx="100" cy="100" r="10" fill="${COLORS.skin}" stroke="${COLORS.outline}" stroke-width="2" />
         <path d="M110 110 L140 150" fill="none" stroke="${COLORS.outline}" stroke-dasharray="3,3" />`,
        `<path d="M60 160 Q40 180 40 200" fill="none" stroke="${COLORS.outline}" stroke-width="6" stroke-linecap="round" />`
    ),

    NAME: BASE_FIGURE("NAME",
        `<path d="M120 140 Q110 140 100 140" fill="none" stroke="${COLORS.outline}" stroke-width="8" stroke-linecap="round" /> <!-- Tapping -->
         <circle cx="90" cy="140" r="10" fill="${COLORS.skin}" stroke="${COLORS.outline}" stroke-width="2" />
         <circle cx="110" cy="140" r="10" fill="${COLORS.skin}" stroke="${COLORS.outline}" stroke-width="2" />`,
        `<path d="M80 160 Q70 180 70 200" fill="none" stroke="${COLORS.outline}" stroke-width="6" stroke-linecap="round" />`
    )
};

// BSL Dual-Hand Variants
export const BSL_GESTURE_ILLUSTRATIONS = {
    ...GESTURE_ILLUSTRATIONS,
    HELLO: BASE_FIGURE("HELLO (BSL)",
        `<path d="M160 150 Q180 120 180 80" fill="none" stroke="${COLORS.outline}" stroke-width="6" stroke-linecap="round" /> 
         <circle cx="180" cy="70" r="12" fill="${COLORS.skin}" stroke="${COLORS.outline}" stroke-width="2" />`,
        `<path d="M40 150 Q20 120 20 80" fill="none" stroke="${COLORS.outline}" stroke-width="6" stroke-linecap="round" />
         <circle cx="20" cy="70" r="12" fill="${COLORS.skin}" stroke="${COLORS.outline}" stroke-width="2" />`
    ),
    THANK_YOU: BASE_FIGURE("THANK YOU (BSL)",
        `<circle cx="120" cy="100" r="10" fill="${COLORS.skin}" stroke="${COLORS.outline}" stroke-width="2" />`,
        `<circle cx="80" cy="100" r="10" fill="${COLORS.skin}" stroke="${COLORS.outline}" stroke-width="2" />`
         // BSL TODO: Add full bilateral arm paths
    ),
    YES: BASE_FIGURE("YES (BSL)",
        `<circle cx="120" cy="115" r="10" fill="${COLORS.skin}" stroke="${COLORS.outline}" stroke-width="2" />`,
        `<circle cx="80" cy="115" r="10" fill="${COLORS.skin}" stroke="${COLORS.outline}" stroke-width="2" />`
         // BSL TODO: Add two-handed nodding sync
    )
};
