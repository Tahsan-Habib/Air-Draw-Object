# Magic Objects — Air Writing App

Write words in the air using just your index finger (via webcam), and watch
them turn into floating 3D-style objects on screen.

---

## 1. What You Need

- **Python 3.9 or newer** installed on your computer
- A **webcam**
- The full project folder, containing:
  - `main.py`
  - `requirements.txt`
  - `images/` folder (apple.png, pen.png, egg.png)

---

## 2. Installation (one-time setup)

Open Command Prompt (Windows) or Terminal (Mac/Linux) inside the project
folder, then run these one at a time:

```
python -m venv venv
venv\Scripts\activate          (Windows)
source venv/bin/activate       (Mac/Linux)

pip install -r requirements.txt
```

The install will take a few minutes and download ~1GB — this is normal,
it includes the AI model that powers the handwriting recognition.

---

## 3. Running the App

With your virtual environment activated, run:

```
python main.py
```

A fullscreen camera window will open. **Click once anywhere inside that
window with your mouse** before using any keyboard shortcut — this gives
the window keyboard focus. If R or Esc don't seem to respond, this click
is almost always the fix.

---

## 4. How to Use It — Controls

### ✍️ Writing Mode (default when you start)

| Action | How |
|---|---|
| **Write** | Point with your **index finger only**, keep your thumb tucked in against your palm, and draw letters in the air |
| **Pause (lift pen)** | Stick your **thumb out to the side** while still pointing — this lifts the "pen" so you can reposition your hand without drawing a line. Great for letters like "P" where you need a gap between strokes |
| **Resume writing** | Tuck your thumb back in |
| **Finish the word / trigger recognition** | **Open all 5 fingers** (show your open palm). After about 1-2 seconds, the app reads what you wrote and — if it recognizes a valid word — the matching object appears floating on screen |

**Tip:** Write in **UPPERCASE-style block letters** — bigger, blockier
strokes are much easier for the recognition to read correctly than small
or cursive writing.

### 🎯 Object Mode (after a word is recognized)

| Action | How |
|---|---|
| **Move the object** | Move your index finger — the object follows your hand |
| **Write another word** | Press **`R`** on your keyboard — this clears the screen and returns you to Writing Mode |

### 🎨 Pen Customization (works anytime)

| Action | How |
|---|---|
| **Change pen size** | Pinch your **thumb and index finger together**, hold the pinch on the horizontal bar at the top of the screen, then move your hand left (smaller) or right (bigger) |
| **Change pen color** | Pinch on the **color circle icon** (top-right of screen) to open a small color palette, then pinch on any color swatch to select it. Neon Blue is the default |

### ⌨️ Keyboard Shortcuts

| Key | Action |
|---|---|
| `R` | Return to writing mode (clears the object and canvas) |
| `Esc` | Close the app |

---

## 5. Currently Recognized Words

This is a **testing/demo build** with only 3 objects set up:

- **apple**
- **pen**
- **egg**

More words and objects can easily be added later by:
1. Adding a transparent PNG of the new object to the `images/` folder
2. Adding the word to the `VALID_WORDS` dictionary near the top of `main.py`
3. Loading the new image into `OBJECT_IMAGES` the same way apple/pen/egg are loaded

---

## 6. Troubleshooting

- **Keyboard shortcuts not responding** → Click once on the camera window
  with your mouse first, then try again. The window needs focus to receive
  key presses.
- **Word not recognized** → Try writing bigger, in uppercase block letters,
  with clear gaps between letters (use the thumb-out pause between letters
  if needed).
- **Nothing happens when I open my hand** → Make sure all 5 fingers are
  clearly spread and visible to the camera, and hold the pose for a moment.

---

## 7. About This Project

Built as a computer vision portfolio project using MediaPipe (hand
tracking), EasyOCR (handwriting recognition), and OpenCV (rendering).
