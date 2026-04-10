/**
 * # CONFIGURE THIS: Path to your gestures folder
 * This map connects predicted words to illustrations and labels.
 */
import AslHelloSVG from './components/AslHelloSVG';
import BslHelloSVG from './components/BslHelloSVG';

export const gestureMap = {
    "HELLO": {
        "ASL": {
            illustration: AslHelloSVG,
            gestureName: "Open Palm Wave",
            description: "(ASL for Hello)",
            bgColor: "#f5f0e4"
        },
        "BSL": {
            illustration: BslHelloSVG,
            gestureName: "Two-Hand Wave",
            description: "(BSL for Hello)",
            bgColor: "#f5f0e4"
        },
        "ISL": {
            imagePath: "public/gestures/ISL_HELLO.png",
            gestureName: "Namaste Joined Palms",
            description: "(ISL for Hello)",
            bgColor: "#f5f0e4"
        }
    }
};
