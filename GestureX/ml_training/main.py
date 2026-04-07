"""
Main Interface - Gesture ↔ Speech Communication System

Provides a menu-based interface to select between:
1. Gesture → Speech: Camera detects gestures and speaks them
2. Speech → Gesture: User speaks/types and sees gesture visualization

Usage:
    python main.py
"""

import os
import sys


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_banner():
    """Print application banner."""
    print("\n" + "="*60)
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║      🤟  GESTURE ↔ SPEECH COMMUNICATION SYSTEM  🗣️    ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    print("="*60)


def print_menu():
    """Print main menu options."""
    print("""
    ┌─────────────────────────────────────────────────────────┐
    │                    SELECT AN OPTION                     │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │    [1]  🖐️ → 🔊  GESTURE to SPEECH                      │
    │         Show hand gestures to camera,                   │
    │         system speaks the meaning                       │
    │                                                         │
    │    [2]  🔊 → 🖐️  SPEECH to GESTURE                      │
    │         Speak or type words,                            │
    │         system shows the gesture                        │
    │                                                         │
    │    [3]  📖  VIEW SUPPORTED GESTURES                     │
    │                                                         │
    │    [4]  🚀  RUN STREAMLIT APP (Full UI)                 │
    │                                                         │
    │    [Q]  ❌  QUIT                                        │
    │                                                         │
    └─────────────────────────────────────────────────────────┘
    """)


def print_gestures():
    """Print supported gestures."""
    print("""
    ┌─────────────────────────────────────────────────────────┐
    │                 SUPPORTED GESTURES                      │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │   GESTURE          │  MEANING          │  WORDS         │
    │  ──────────────────┼───────────────────┼───────────────│
    │   ✊ Closed_Fist    │  Stop/Wait        │  stop, wait    │
    │   ✋ Open_Palm      │  Hello/Goodbye    │  hello, hi, bye│
    │   ☝️ Pointing_Up    │  One moment       │  wait, one     │
    │   👎 Thumb_Down     │  No/Disagree      │  no, bad       │
    │   👍 Thumb_Up       │  Yes/OK/Good      │  yes, ok, good │
    │   ✌️ Victory        │  Peace/Victory    │  peace, two    │
    │   🤟 ILoveYou       │  I Love You       │  love, i love  │
    │                                                         │
    └─────────────────────────────────────────────────────────┘
    """)
    input("\n    Press Enter to continue...")


def run_gesture_to_speech():
    """Run gesture to speech mode."""
    try:
        from gesture_to_speech import run_gesture_to_speech as g2s
        clear_screen()
        g2s()
    except ImportError as e:
        print(f"\nError importing module: {e}")
        print("Make sure gesture_to_speech.py exists.")
        input("\nPress Enter to continue...")
    except Exception as e:
        print(f"\nError: {e}")
        input("\nPress Enter to continue...")


def run_speech_to_gesture():
    """Run speech to gesture mode."""
    try:
        from speech_to_gesture import run_speech_to_gesture as s2g
        clear_screen()
        s2g()
    except ImportError as e:
        print(f"\nError importing module: {e}")
        print("Make sure speech_to_gesture.py exists.")
        input("\nPress Enter to continue...")
    except Exception as e:
        print(f"\nError: {e}")
        input("\nPress Enter to continue...")


def run_streamlit():
    """Launch Streamlit app."""
    print("\n    Starting Streamlit app...")
    print("    Press Ctrl+C in terminal to stop.\n")
    os.system("streamlit run app.py")


def main():
    """Main application loop."""
    while True:
        clear_screen()
        print_banner()
        print_menu()
        
        choice = input("    Enter your choice [1/2/3/4/Q]: ").strip().lower()
        
        if choice == '1':
            run_gesture_to_speech()
        elif choice == '2':
            run_speech_to_gesture()
        elif choice == '3':
            clear_screen()
            print_banner()
            print_gestures()
        elif choice == '4':
            run_streamlit()
        elif choice == 'q' or choice == 'quit' or choice == 'exit':
            clear_screen()
            print("\n    👋 Goodbye! Thank you for using Gesture ↔ Speech System\n")
            break
        else:
            print("\n    ⚠️  Invalid choice. Please enter 1, 2, 3, 4, or Q.")
            input("    Press Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n    👋 Program interrupted. Goodbye!\n")
        sys.exit(0)
