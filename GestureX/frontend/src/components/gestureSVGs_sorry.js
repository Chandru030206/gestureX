import React from 'react';

const COLORS = { skin: "#e8c9a0", stroke: "#b8860b" };

const SVG_WRAP = (content) => (
    <svg viewBox="0 0 160 160" xmlns="http://www.w3.org/2000/svg" style={{ width: '120px', height: '120px' }}>
        {content}
    </svg>
);

export const ASL_SORRY = () => SVG_WRAP(
    <g>
        <circle cx="80" cy="80" r="22" fill={COLORS.skin} stroke={COLORS.stroke} strokeWidth="1.5" /> <!-- Fist -->
        <path d="M50 80 Q80 50 110 80 Q80 110 50 80" fill="none" stroke={COLORS.stroke} strokeWidth="1" strokeDasharray="3,3" /> <!-- Circle motion -->
    </g>
);

export const BSL_SORRY = () => SVG_WRAP(
    <g>
        <path d="M60 70 L100 70 L105 100 L55 100 Z" fill={COLORS.skin} stroke={COLORS.stroke} strokeWidth="1.5" /> <!-- Open hand -->
        <path d="M80 60 L80 110" fill="none" stroke={COLORS.stroke} strokeWidth="2" strokeDasharray="3,3" /> <!-- Pat -->
    </g>
);

export const JSL_SORRY = () => SVG_WRAP(
    <g>
        <path d="M75 90 L85 60 L80 50 Z" fill={COLORS.skin} stroke={COLORS.stroke} strokeWidth="1.5" /> <!-- Hands pressed -->
        <path d="M85 90 L75 60 L80 50 Z" fill={COLORS.skin} stroke={COLORS.stroke} strokeWidth="1.5" />
        <path d="M60 40 Q80 20 100 40" fill="none" stroke={COLORS.stroke} strokeWidth="1.5" /> <!-- Head Bow -->
    </g>
);

export const AUSLAN_SORRY = () => SVG_WRAP(
    <g>
        <path d="M60 80 Q80 70 100 80 L80 110 Z" fill={COLORS.skin} stroke={COLORS.stroke} strokeWidth="1.5" /> <!-- Flat hand -->
        <circle cx="80" cy="80" r="15" fill="none" stroke={COLORS.stroke} strokeWidth="1" strokeDasharray="2,2" /> <!-- Small circle -->
    </g>
);

export const LSF_SORRY = () => SVG_WRAP(
    <g>
        <rect x="65" y="60" width="30" height="15" rx="3" fill={COLORS.skin} stroke={COLORS.stroke} strokeWidth="1.5" /> <!-- Shoulder tap -->
        <path d="M50 30 Q80 15 110 30" fill="none" stroke={COLORS.stroke} strokeWidth="1.5" /> <!-- Bow -->
    </g>
);

export const DGS_SORRY = () => SVG_WRAP(
    <g>
        <circle cx="80" cy="60" r="18" fill={COLORS.skin} stroke={COLORS.stroke} strokeWidth="1.5" /> <!-- Upper fist -->
        <path d="M80 55 V85" fill="none" stroke={COLORS.stroke} strokeWidth="2" strokeDasharray="2,2" /> <!-- Tap -->
    </g>
);

export const LIBRAS_SORRY = () => SVG_WRAP(
    <g>
        <rect x="60" y="80" width="40" height="12" rx="4" fill={COLORS.skin} stroke={COLORS.stroke} strokeWidth="1.5" /> <!-- Hand on chest -->
        <path d="M70 90 V105 M90 90 V105" fill="none" stroke={COLORS.stroke} strokeWidth="1" strokeDasharray="2,2" /> <!-- Two taps -->
    </g>
);

export const CSL_SORRY = () => SVG_WRAP(
    <g>
        <ellipse cx="73" cy="80" rx="12" ry="12" fill={COLORS.skin} stroke={COLORS.stroke} strokeWidth="1.5" /> <!-- Clasp 1 -->
        <ellipse cx="87" cy="80" rx="12" ry="12" fill={COLORS.skin} stroke={COLORS.stroke} strokeWidth="1.5" /> <!-- Clasp 2 -->
        <path d="M70 40 Q80 30 90 40" fill="none" stroke={COLORS.stroke} strokeWidth="1.5" /> <!-- Bow -->
    </g>
);

export const KSL_SORRY = () => JSL_SORRY(); // Using shared bowing logic
export const ISL_SORRY = () => SVG_WRAP(<path d="M80 70 Q100 50 120 70 L80 110 Z" fill={COLORS.skin} stroke={COLORS.stroke} strokeWidth="1.5" />); // Existing ISL placeholder
