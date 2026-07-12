import cv2
import mediapipe as mp
import easyocr
import numpy as np
import threading
import queue
import difflib
from time import time
from collections import deque

# ============================================
# CONFIG
# ============================================

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

THICKNESS_MIN = 6
THICKNESS_MAX = 60
DEFAULT_THICKNESS = 25
DRAW_THICKNESS = float(DEFAULT_THICKNESS)   # float so the slider can change it smoothly

DRAW_GLOW = 18
GLOW_DOWNSCALE = 0.5

SMOOTHING = 0.25

# thumb-tip <-> index-tip distance (relative to hand size) below this = "pinching"
# used ONLY for UI clicks (slider / color palette), not for writing
PINCH_THRESHOLD_RATIO = 0.35

# thumb-tip <-> index-MCP(knuckle) distance (relative to hand size) above this = "thumb out"
# thumb out = pause writing. thumb tucked in = write. TUNE THIS if it feels too sensitive/insensitive.
THUMB_OUT_RATIO = 0.85

# frames a pinch must dwell on a color swatch/icon before it "clicks" (debounce so it
# doesn't fire by accident while passing through)
UI_CLICK_HOLD_FRAMES = 5

# frames the open-hand (5 fingers) pose must hold before OCR fires
# kept short on purpose so OCR feels instant, per your request
FIVE_FINGER_HOLD_FRAMES = 3

FUZZY_MATCH_CUTOFF = 0.55

OBJECT_SIZE = 320   # bigger, so it reads as "floating on your hand" not a small icon

DEBUG_SHOW_OCR_CROP = False

# Neon color palette (BGR order, since OpenCV uses BGR not RGB)
NEON_COLORS = {
    "blue":  (255, 217, 4),
    "pink":  (147, 20, 255),
    "green": (20, 255, 57),
}
DEFAULT_COLOR_KEY = "blue"

# ============================================
# MEDIAPIPE
# ============================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

draw = mp.solutions.drawing_utils

# ============================================
# OCR (runs in a background thread so it never blocks the camera loop)
# ============================================

reader = easyocr.Reader(['en'], gpu=False)

VALID_WORDS = {
    "apple": "apple",
    "pen": "pen",
    "egg": "egg",
}

ocr_input_queue = queue.Queue()
ocr_result_queue = queue.Queue()


def fuzzy_match(text_nospace):
    """Snap a messy OCR guess to the closest valid word instead of
    requiring an exact match."""
    candidate_keys = list(VALID_WORDS.keys())
    candidate_nospace = [c.replace(" ", "") for c in candidate_keys]
    close = difflib.get_close_matches(
        text_nospace, candidate_nospace, n=1, cutoff=FUZZY_MATCH_CUTOFF
    )
    if not close:
        return ""
    matched_key = candidate_keys[candidate_nospace.index(close[0])]
    return VALID_WORDS[matched_key]


def recognize_word(image):
    result = reader.readtext(image, detail=0, paragraph=False)
    if len(result) == 0:
        return ""
    text = " ".join(result).lower()
    text = text.replace("-", " ").replace("_", " ").strip()
    if text in VALID_WORDS:
        return VALID_WORDS[text]
    text_nospace = text.replace(" ", "")
    if text_nospace in VALID_WORDS:
        return VALID_WORDS[text_nospace]
    return fuzzy_match(text_nospace)


def ocr_worker():
    while True:
        img = ocr_input_queue.get()
        if img is None:
            break
        word = recognize_word(img)
        ocr_result_queue.put(word)


ocr_thread = threading.Thread(target=ocr_worker, daemon=True)
ocr_thread.start()

ocr_busy = False

# ============================================
# CAMERA
# ============================================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

cv2.namedWindow("Magic Objects", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Magic Objects", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
try:
    # Pull the window to the front so it actually has keyboard focus.
    # (Not supported on every OpenCV build/backend - safe to ignore if it errors.)
    cv2.setWindowProperty("Magic Objects", cv2.WND_PROP_TOPMOST, 1)
except cv2.error:
    pass

# ============================================
# LOAD PNG OBJECTS
# ============================================

OBJECT_IMAGES = {
    "apple": cv2.imread("images/apple.png", cv2.IMREAD_UNCHANGED),
    "pen": cv2.imread("images/pen.png", cv2.IMREAD_UNCHANGED),
    "egg": cv2.imread("images/egg.png", cv2.IMREAD_UNCHANGED),   # <-- replaced ice cream
}

for key, img in OBJECT_IMAGES.items():
    if img is not None:
        # Some PNGs (like your apple.png) have no alpha channel -- only 3 channels (BGR).
        # overlay_png() needs a 4th (alpha) channel to know what's transparent, so if it's
        # missing we add one here, treating the whole image as fully opaque.
        if img.shape[2] == 3:
            alpha_channel = np.full(img.shape[:2], 255, dtype=img.dtype)
            img = cv2.merge([img[:, :, 0], img[:, :, 1], img[:, :, 2], alpha_channel])

        h, w = img.shape[:2]
        scale = OBJECT_SIZE / w
        OBJECT_IMAGES[key] = cv2.resize(
            img, (OBJECT_SIZE, int(h * scale)), interpolation=cv2.INTER_AREA
        )

print("Apple:", OBJECT_IMAGES["apple"] is not None)
print("Pen:", OBJECT_IMAGES["pen"] is not None)
print("Egg:", OBJECT_IMAGES["egg"] is not None)

# ============================================
# STATE
# ============================================

MODE_WRITE = "write"
MODE_OBJECT = "object"
mode = MODE_WRITE

canvas = None
points = deque(maxlen=5000)
drawing = False
recognized_word = ""
current_object = None
pen_color_key = DEFAULT_COLOR_KEY

object_x = 0
object_y = 0

smooth_x = None
smooth_y = None

frame_count = 0
five_finger_counter = 0

# UI layout (computed once from first frame size)
ui_layout_ready = False
slider_x1 = slider_x2 = slider_y = 0
color_icon_center = (0, 0)
color_icon_radius = 26
swatch_positions = {}
swatch_radius = 24
color_menu_open = False

ui_hover_target = None
ui_hover_counter = 0
ui_action_armed = True   # False right after a click fires, True again once pinch releases

# ============================================
# HELPERS
# ============================================

def overlay_png(background, png, center_x, center_y):
    if png is None:
        return
    h, w = png.shape[:2]
    x = int(center_x - w / 2)
    y = int(center_y - h / 2)
    # clamp instead of hiding the object when it nears the edge
    x = max(0, min(x, background.shape[1] - w - 1))
    y = max(0, min(y, background.shape[0] - h - 1))

    alpha = png[:, :, 3] / 255.0
    rgb = png[:, :, :3]
    roi = background[y:y+h, x:x+w]
    for c in range(3):
        roi[:, :, c] = alpha * rgb[:, :, c] + (1 - alpha) * roi[:, :, c]
    background[y:y+h, x:x+w] = roi


def smooth_point(x, y):
    global smooth_x, smooth_y
    if smooth_x is None:
        smooth_x, smooth_y = x, y
    smooth_x = int(smooth_x + (x - smooth_x) * SMOOTHING)
    smooth_y = int(smooth_y + (y - smooth_y) * SMOOTHING)
    return smooth_x, smooth_y


def hand_size(hand, width, height):
    wrist = hand.landmark[0]
    middle_mcp = hand.landmark[9]
    wx, wy = wrist.x * width, wrist.y * height
    mx, my = middle_mcp.x * width, middle_mcp.y * height
    return np.hypot(mx - wx, my - wy)


def create_word_image(points, width, height):
    img = np.zeros((height, width), dtype=np.uint8)
    for i in range(1, len(points)):
        if points[i-1] is None or points[i] is None:
            continue
        cv2.line(img, points[i-1], points[i], 255, 35, cv2.LINE_AA)
    ys, xs = np.where(img > 0)
    if len(xs) == 0:
        return None
    pad = 40
    x1 = max(xs.min() - pad, 0)
    y1 = max(ys.min() - pad, 0)
    x2 = min(xs.max() + pad, width)
    y2 = min(ys.max() + pad, height)
    crop = img[y1:y2, x1:x2]
    # Smaller crop = EasyOCR processes far fewer pixels = noticeably faster inference.
    # 480x200 is still plenty of detail for short words; INTER_LINEAR is cheaper than CUBIC.
    crop = cv2.resize(crop, (480, 200), interpolation=cv2.INTER_LINEAR)
    kernel = np.ones((5, 5), np.uint8)
    crop = cv2.dilate(crop, kernel, 1)
    crop = cv2.erode(crop, kernel, 1)
    return crop


def setup_ui_layout(w, h):
    global slider_x1, slider_x2, slider_y, color_icon_center
    slider_y = 60
    SLIDER_LENGTH = 260  # medium-size line instead of stretching near full width
    slider_x1 = w // 2 - SLIDER_LENGTH // 2
    slider_x2 = w // 2 + SLIDER_LENGTH // 2
    color_icon_center = (slider_x2 + 55, slider_y)  # sits just beside the line's right end


def thickness_to_slider_x(thickness):
    t = (thickness - THICKNESS_MIN) / (THICKNESS_MAX - THICKNESS_MIN)
    return int(slider_x1 + t * (slider_x2 - slider_x1))


def slider_x_to_thickness(x):
    t = (x - slider_x1) / max(1, (slider_x2 - slider_x1))
    t = max(0.0, min(1.0, t))
    return THICKNESS_MIN + t * (THICKNESS_MAX - THICKNESS_MIN)


def draw_ui(frame):
    # pen-size slider bar
    cv2.line(frame, (slider_x1, slider_y), (slider_x2, slider_y), (90, 90, 90), 6)
    handle_x = thickness_to_slider_x(DRAW_THICKNESS)
    cv2.circle(frame, (handle_x, slider_y), 14, NEON_COLORS[pen_color_key], -1)
    cv2.circle(frame, (handle_x, slider_y), 16, (255, 255, 255), 2)

    # color icon (shows current color)
    cv2.circle(frame, color_icon_center, color_icon_radius, NEON_COLORS[pen_color_key], -1)
    cv2.circle(frame, color_icon_center, color_icon_radius, (255, 255, 255), 2)

    if color_menu_open:
        keys = list(NEON_COLORS.keys())
        for i, k in enumerate(keys):
            pos = (
                color_icon_center[0],
                color_icon_center[1] + color_icon_radius + 40 + i * (swatch_radius * 2 + 16),
            )
            swatch_positions[k] = pos
            cv2.circle(frame, pos, swatch_radius, NEON_COLORS[k], -1)
            cv2.circle(frame, pos, swatch_radius, (255, 255, 255), 2)


def dist(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])


# ============================================
# MAIN LOOP
# ============================================

while True:
    frame_count += 1
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]
    if not ui_layout_ready:
        setup_ui_layout(w, h)
        ui_layout_ready = True

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    trigger_ocr = False

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        landmarks = hand.landmark

        index_tip = landmarks[8]
        index_pip = landmarks[6]
        index_mcp = landmarks[5]
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        ring_tip = landmarks[16]
        ring_pip = landmarks[14]
        pinky_tip = landmarks[20]
        pinky_pip = landmarks[18]
        thumb_tip = landmarks[4]

        x = int(index_tip.x * w)
        y = int(index_tip.y * h)
        x, y = smooth_point(x, y)

        scale = hand_size(hand, w, h)

        tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

        # pinch = thumb tip touching index TIP -> used only for UI clicks
        pinch_dist = np.hypot(tx - x, ty - y)
        is_pinching = scale > 0 and (pinch_dist / scale) < PINCH_THRESHOLD_RATIO
        pinch_point = ((tx + x) // 2, (ty + y) // 2)

        # thumb "out" = thumb tip far from index KNUCKLE -> pause writing
        mcp_x, mcp_y = int(index_mcp.x * w), int(index_mcp.y * h)
        thumb_mcp_dist = np.hypot(tx - mcp_x, ty - mcp_y)
        thumb_out = scale > 0 and (thumb_mcp_dist / scale) > THUMB_OUT_RATIO

        index_up = index_tip.y < index_pip.y
        middle_up = middle_tip.y < middle_pip.y
        ring_up = ring_tip.y < ring_pip.y
        pinky_up = pinky_tip.y < pinky_pip.y

        open_hand = index_up and middle_up and ring_up and pinky_up

        # ---------------- UI interaction: slider + color palette ----------------
        handled_by_ui = False

        if is_pinching:
            px, py = pinch_point
            near_slider = abs(py - slider_y) < 45 and (slider_x1 - 30) < px < (slider_x2 + 30)
            near_icon = dist((px, py), color_icon_center) < color_icon_radius + 15

            if near_slider and not color_menu_open:
                target_thickness = slider_x_to_thickness(px)
                DRAW_THICKNESS = DRAW_THICKNESS + (target_thickness - DRAW_THICKNESS) * 0.3
                handled_by_ui = True
                ui_hover_target = None

            elif near_icon and not color_menu_open:
                if ui_hover_target != "icon":
                    ui_hover_target = "icon"
                    ui_hover_counter = 0
                ui_hover_counter += 1
                if ui_hover_counter >= UI_CLICK_HOLD_FRAMES and ui_action_armed:
                    color_menu_open = True
                    ui_action_armed = False
                handled_by_ui = True

            elif color_menu_open:
                hit_key = None
                for k, pos in swatch_positions.items():
                    if dist((px, py), pos) < swatch_radius + 15:
                        hit_key = k
                        break
                if hit_key:
                    if ui_hover_target != hit_key:
                        ui_hover_target = hit_key
                        ui_hover_counter = 0
                    ui_hover_counter += 1
                    if ui_hover_counter >= UI_CLICK_HOLD_FRAMES and ui_action_armed:
                        pen_color_key = hit_key
                        color_menu_open = False
                        ui_action_armed = False
                handled_by_ui = True
        else:
            ui_hover_target = None
            ui_hover_counter = 0
            ui_action_armed = True

        # ---------------- OCR trigger: open hand (5 fingers) ----------------
        if mode == MODE_WRITE and open_hand and not handled_by_ui:
            five_finger_counter += 1
        else:
            five_finger_counter = 0

        trigger_ocr = (
            mode == MODE_WRITE
            and five_finger_counter == FIVE_FINGER_HOLD_FRAMES
            and len(points) > 10
            and not ocr_busy
        )

        # ---------------- WRITE MODE ----------------
        if mode == MODE_WRITE and not handled_by_ui:
            if index_up and not thumb_out and not open_hand:
                drawing = True
                cv2.circle(frame, (x, y), 14, (255, 255, 255), -1)
                cv2.circle(frame, (x, y), 24, NEON_COLORS[pen_color_key], 3)

                if len(points) == 0 or points[-1] is None:
                    points.append((x, y))
                else:
                    lx, ly = points[-1]
                    distance = np.hypot(x - lx, y - ly)
                    if distance > 4:
                        cv2.line(canvas, (lx, ly), (x, y), NEON_COLORS[pen_color_key],
                                  int(DRAW_THICKNESS), cv2.LINE_AA)
                        points.append((x, y))

            elif index_up and thumb_out:
                # PAUSED: thumb out -> pen lifted, reposition freely without drawing a line
                if drawing:
                    points.append(None)   # breaks the stroke so it won't connect
                drawing = False
                cv2.circle(frame, (x, y), 14, (0, 0, 255), 2)   # red ring = pen is up

            else:
                if drawing:
                    points.append(None)
                drawing = False

        # ---------------- OBJECT MODE: object follows index fingertip ----------------
        if mode == MODE_OBJECT:
            object_x += int((x - object_x) * 0.35)
            object_y += int((y - object_y) * 0.35)

    else:
        smooth_x = None
        smooth_y = None
        if drawing:
            points.append(None)
        drawing = False
        five_finger_counter = 0
        ui_hover_target = None
        ui_hover_counter = 0

    # ============================================
    # NEON RENDER
    # ============================================

    small = cv2.resize(canvas, None, fx=GLOW_DOWNSCALE, fy=GLOW_DOWNSCALE,
                        interpolation=cv2.INTER_LINEAR)
    small_glow = cv2.GaussianBlur(small, (0, 0), DRAW_GLOW * GLOW_DOWNSCALE)
    glow = cv2.resize(small_glow, (canvas.shape[1], canvas.shape[0]),
                       interpolation=cv2.INTER_LINEAR)

    frame = cv2.addWeighted(frame, 1.0, glow, 0.95, 0)

    mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) > 0
    frame[mask] = (255, 255, 255)   # bright white core, colored neon glow around it

    # ============================================
    # OCR TRIGGER + RESULT (non-blocking)
    # ============================================

    if trigger_ocr:
        img = create_word_image(points, frame.shape[1], frame.shape[0])
        if img is not None:
            if DEBUG_SHOW_OCR_CROP:
                cv2.imshow("OCR", img)
            ocr_input_queue.put(img)
            ocr_busy = True
        points.clear()
        canvas[:] = 0

    try:
        word = ocr_result_queue.get_nowait()
        ocr_busy = False
        if word != "" and word in OBJECT_IMAGES and OBJECT_IMAGES[word] is not None:
            recognized_word = word
            current_object = OBJECT_IMAGES[word]
            object_x = frame.shape[1] // 2
            object_y = frame.shape[0] // 2
            mode = MODE_OBJECT
            print("Detected:", word)
        else:
            print("Word not recognized")
    except queue.Empty:
        pass

    # ============================================
    # DRAW OBJECT (follows index finger until R is pressed)
    # ============================================

    if mode == MODE_OBJECT and current_object is not None:
        overlay_png(frame, current_object, object_x, object_y)

    # ============================================
    # UI OVERLAY
    # ============================================

    draw_ui(frame)

    # ============================================
    # HUD (kept minimal on purpose)
    # ============================================

    if mode == MODE_WRITE:
        cv2.putText(frame, "MODE : WRITE", (25, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    else:
        cv2.putText(frame, f"OBJECT : {recognized_word.upper()}  (R = write again)",
                    (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if ocr_busy:
        cv2.putText(frame, "Reading...", (25, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    # ============================================
    # SHOW FRAME
    # ============================================

    cv2.imshow("Magic Objects", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27 or key == ord("q") or key == ord("Q"):
        break
    elif key == ord("c") or key == ord("C"):
        canvas[:] = 0
        points.clear()
    elif key == ord("r") or key == ord("R"):
        mode = MODE_WRITE
        canvas[:] = 0
        points.clear()
        recognized_word = ""
        current_object = None

# ============================================
# CLEANUP
# ============================================

ocr_input_queue.put(None)
cap.release()
cv2.destroyAllWindows()