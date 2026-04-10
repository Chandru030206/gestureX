import React, { useState } from 'react';
import { gestureLibrary } from '../gestureLibrary';
import * as SORRY_SVGS from './gestureSVGs_sorry';

const GestureCard = ({ language, word, accuracy }) => {
    const [hasImgError, setHasImgError] = useState(false);
    
    // Lookup data
    const normalizedWord = (word || "HELLO").toUpperCase();
    const regionData = gestureLibrary[language]?.[normalizedWord] || null;

    // Component Resolution
    const renderContent = () => {
        if (!regionData) return <SimpleFallback word={word} />;

        // 1. Image Path Check
        if (regionData.imagePath && !hasImgError) {
            return (
                <img 
                    src={regionData.imagePath} 
                    alt={regionData.gestureName}
                    style={{ maxWidth: '100%', maxHeight: '100%' }}
                    onError={() => setHasImgError(true)}
                />
            );
        }

        // 2. SVG Fallback Component Check
        const SVGComp = SORRY_SVGS[regionData.svgFallback];
        if (SVGComp) {
            return <SVGComp />;
        }

        // 3. Last Resort
        return <SimpleFallback word={word} />;
    };

    return (
        <div style={styles.card}>
            {/* Header / Accuracy */}
            <div style={styles.accuracyPill}>
                {accuracy}% ACCURACY
            </div>

            {/* Illustration Canvas */}
            <div style={styles.canvas}>
                {renderContent()}
            </div>

            {/* Metadata Labels */}
            <div style={styles.labelGroup}>
                <div style={styles.name}>{regionData?.gestureName || word}</div>
                <div style={styles.desc}>{regionData?.description || `Standard ${language} Variation`}</div>
            </div>
        </div>
    );
};

const SimpleFallback = ({ word }) => (
    <div style={styles.fallbackBox}>
        <span style={styles.fallbackText}>{(word || "G")[0].toUpperCase()}</span>
    </div>
);

const styles = {
    card: {
        background: '#000',
        borderRadius: '12px',
        overflow: 'hidden',
        border: '1px solid #222',
        width: '100%',
        display: 'flex',
        flexDirection: 'column'
    },
    canvas: {
        background: '#f5f0e4', // The Premium Cream
        height: '180px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        position: 'relative'
    },
    accuracyPill: {
        position: 'absolute',
        top: '12px',
        left: '12px',
        zIndex: 10,
        background: 'rgba(0,0,0,0.6)',
        border: '1px solid #00E5C8',
        color: '#00E5C8',
        borderRadius: '20px',
        padding: '3px 8px',
        fontSize: '0.65rem',
        fontWeight: '900'
    },
    labelGroup: {
        padding: '12px',
        textAlign: 'center'
    },
    name: {
        color: '#FFF',
        fontSize: '13px',
        fontWeight: 'bold',
        marginBottom: '2px'
    },
    desc: {
        color: '#00E5C8',
        fontSize: '11px',
        opacity: 0.7
    },
    fallbackBox: {
        width: '60px',
        height: '60px',
        borderRadius: '50%',
        border: '2px dashed #ccc',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
    },
    fallbackText: {
        fontSize: '24px',
        color: '#ccc',
        fontWeight: 'bold'
    }
};

export default GestureCard;
