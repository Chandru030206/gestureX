/**
 * # CONFIGURE THIS: gestureLibrary.js
 * Master registry for 10 regions x 10 words.
 */

export const gestureLibrary = {
    "ASL": {
        "SORRY": { gestureName: "Fist Circle on Chest", description: "ASL for Sorry", imagePath: "/gestures/asl/sorry.svg", handType: "single", svgFallback: "ASL_SORRY" },
        "HELLO": { gestureName: "Open Palm Wave", description: "ASL for Hello", imagePath: "/gestures/asl/hello.svg", handType: "single", svgFallback: "GENERIC_HELLO" }
    },
    "BSL": {
        "SORRY": { gestureName: "Chest Pat", description: "BSL for Sorry", imagePath: "/gestures/bsl/sorry.svg", handType: "single", svgFallback: "BSL_SORRY" },
        "HELLO": { gestureName: "Two-Hand Wave", description: "BSL for Hello", imagePath: "/gestures/bsl/hello.svg", handType: "dual", svgFallback: "BSL_HELLO" }
    },
    "ISL": {
        "SORRY": { gestureName: "Heart Touch", description: "ISL for Sorry", imagePath: "/gestures/isl/sorry.svg", handType: "single", svgFallback: "ISL_SORRY" },
        "HELLO": { gestureName: "Namaste", description: "ISL for Hello", imagePath: "/gestures/isl/hello.svg", handType: "single",  svgFallback: "ISL_HELLO" }
    },
    "JSL": {
        "SORRY": { gestureName: "Pressed Hands Bow", description: "JSL for Sorry", imagePath: "/gestures/jsl/sorry.svg", handType: "dual", svgFallback: "JSL_SORRY" }
    },
    "AUSLAN": {
        "SORRY": { gestureName: "Small Circle on Chest", description: "AUSLAN for Sorry", imagePath: "/gestures/auslan/sorry.svg", handType: "single", svgFallback: "AUSLAN_SORRY" }
    },
    "LSF": {
        "SORRY": { gestureName: "Shoulder Tap Bow", description: "LSF for Sorry", imagePath: "/gestures/lsf/sorry.svg", handType: "single", svgFallback: "LSF_SORRY" }
    },
    "DGS": {
        "SORRY": { gestureName: "Upper Chest Tap", description: "DGS for Sorry", imagePath: "/gestures/dgs/sorry.svg", handType: "single", svgFallback: "DGS_SORRY" }
    },
    "LIBRAS": {
        "SORRY": { gestureName: "Two Small Taps", description: "LIBRAS for Sorry", imagePath: "/gestures/libras/sorry.svg", handType: "single", svgFallback: "LIBRAS_SORRY" }
    },
    "KSL": {
        "SORRY": { gestureName: "Deep Formal Bow", description: "KSL for Sorry", imagePath: "/gestures/ksl/sorry.svg", handType: "dual", svgFallback: "KSL_SORRY" }
    },
    "CSL": {
        "SORRY": { gestureName: "Clasped Forward Bow", description: "CSL for Sorry", imagePath: "/gestures/csl/sorry.svg", handType: "dual", svgFallback: "CSL_SORRY" }
    }
};
